// silero_vad.h — single-header Silero VAD inference on ggml.
// Ported from whisper.cpp's Silero VAD implementation.
//
// Exposes:
//   - silero_vad_load / silero_vad_free: load/free a Silero VAD model from a
//     whisper.cpp-style ggml file (produced by scripts/convert-silero-vad-to-ggml.py).
//   - silero_vad_segments / silero_vad_segments_from_model: run VAD on a
//     16 kHz mono float waveform and return speech segments [start_ms, end_ms].
//   - vad_inc_*: incremental (streaming) VAD that keeps LSTM state across
//     calls for continuous segment detection.
//
// The model file format is the same one used by whisper.cpp's
// models/convert-silero-vad-to-ggml.py (magic 0x67676d6c), so the same
// converted .bin can be used by both projects.
#pragma once

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

namespace silero_vad_impl {

// Default VAD parameters (mirror whisper.cpp's whisper_vad_default_params).
struct vad_params {
    float threshold               = 0.5f;
    int   min_speech_duration_ms  = 250;
    int   min_silence_duration_ms = 100;
    float max_speech_duration_s   = 1e9f;  // effectively unbounded
    int   speech_pad_ms           = 30;
};

// Hyperparameters describing the Silero VAD architecture.
struct vad_hparams {
    int32_t n_encoder_layers  = 4;
    int32_t encoder_in_channels[4]  = {129, 128, 64, 64};
    int32_t encoder_out_channels[4] = {128, 64, 64, 128};
    int32_t kernel_sizes[4]         = {3, 3, 3, 3};
    int32_t lstm_input_size   = 128;
    int32_t lstm_hidden_size  = 128;
    int32_t final_conv_in     = 128;
    int32_t final_conv_out    = 1;
};

// Tensor name constants (must match whisper.cpp's convert-silero-vad-to-ggml.py).
static const char * const TENSOR_STFT_BASIS        = "_model.stft.forward_basis_buffer";
static const char * const TENSOR_ENC_0_WEIGHT      = "_model.encoder.0.reparam_conv.weight";
static const char * const TENSOR_ENC_0_BIAS        = "_model.encoder.0.reparam_conv.bias";
static const char * const TENSOR_ENC_1_WEIGHT      = "_model.encoder.1.reparam_conv.weight";
static const char * const TENSOR_ENC_1_BIAS        = "_model.encoder.1.reparam_conv.bias";
static const char * const TENSOR_ENC_2_WEIGHT      = "_model.encoder.2.reparam_conv.weight";
static const char * const TENSOR_ENC_2_BIAS        = "_model.encoder.2.reparam_conv.bias";
static const char * const TENSOR_ENC_3_WEIGHT      = "_model.encoder.3.reparam_conv.weight";
static const char * const TENSOR_ENC_3_BIAS        = "_model.encoder.3.reparam_conv.bias";
static const char * const TENSOR_LSTM_WEIGHT_IH    = "_model.decoder.rnn.weight_ih";
static const char * const TENSOR_LSTM_WEIGHT_HH    = "_model.decoder.rnn.weight_hh";
static const char * const TENSOR_LSTM_BIAS_IH      = "_model.decoder.rnn.bias_ih";
static const char * const TENSOR_LSTM_BIAS_HH      = "_model.decoder.rnn.bias_hh";
static const char * const TENSOR_FINAL_CONV_WEIGHT = "_model.decoder.decoder.2.weight";
static const char * const TENSOR_FINAL_CONV_BIAS   = "_model.decoder.decoder.2.bias";

// Model container: owns the ggml context + tensor map for the loaded weights.
struct vad {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    std::map<std::string, ggml_tensor *> t;
    vad_hparams hparams;
    int n_window  = 512;   // samples per VAD window (32ms at 16kHz)
    int n_context = 64;    // reflective padding on each side

    ggml_tensor * g(const std::string & n) const {
        auto it = t.find(n);
        if (it == t.end()) {
            fprintf(stderr, "silero_vad: missing tensor %s\n", n.c_str());
            return nullptr;
        }
        return it->second;
    }
};

static inline int samples_to_ms(int samples) {
    // 16 kHz -> 1 sample = 1/16 ms
    return (int)((int64_t)samples * 1000 / 16000);
}

// Linear layer helper: y = w @ x + b  (b optional).
static inline ggml_tensor * lin(ggml_context * c, ggml_tensor * w,
                                ggml_tensor * b, ggml_tensor * x) {
    ggml_tensor * y = ggml_mul_mat(c, w, x);
    return b ? ggml_add(c, y, b) : y;
}

// ---- File loading ----

static inline bool read_safe(std::ifstream & fin, void * dst, size_t n) {
    fin.read((char *)dst, (std::streamsize)n);
    return fin.good();
}
template <typename T>
static inline bool read_safe(std::ifstream & fin, T & v) {
    fin.read((char *)&v, sizeof(T));
    return fin.good();
}

// The ggml file format is little-endian. On big-endian hosts this would need
// real byte-swapping; whisper.cpp's BYTESWAP_TENSOR handles that. We assume a
// little-endian host (x86/ARM little-endian), matching all common targets.
static inline void maybe_byteswap_tensor(ggml_tensor * /*t*/) {}

// Load a Silero VAD model from a whisper.cpp-style ggml file into `m`.
// Returns true on success. On failure, prints an error and returns false.
inline bool silero_vad_load(vad & m, const std::string & path) {
    std::ifstream fin(path, std::ios::binary);
    if (!fin) {
        fprintf(stderr, "silero_vad: cannot open %s\n", path.c_str());
        return false;
    }

    uint32_t magic = 0;
    if (!read_safe(fin, magic) || magic != 0x67676d6c) {
        fprintf(stderr, "silero_vad: invalid model file (bad magic)\n");
        return false;
    }

    // model type string
    int32_t str_len = 0;
    if (!read_safe(fin, str_len) || str_len <= 0 || str_len > 256) {
        fprintf(stderr, "silero_vad: bad model type length\n");
        return false;
    }
    {
        std::vector<char> buf((size_t)str_len + 1, 0);
        if (!read_safe(fin, buf.data(), (size_t)str_len)) return false;
        // model_type = "silero-16k"; not stored.
    }
    // version (major, minor, patch)
    int32_t major = 0, minor = 0, patch = 0;
    if (!read_safe(fin, major) || !read_safe(fin, minor) || !read_safe(fin, patch)) return false;

    // window + context
    if (!read_safe(fin, m.n_window) || !read_safe(fin, m.n_context)) return false;
    if (m.n_window <= 0 || m.n_window > 65536 || m.n_context < 0) {
        fprintf(stderr, "silero_vad: bad window/context (%d/%d)\n", m.n_window, m.n_context);
        return false;
    }

    // hparams
    auto & hp = m.hparams;
    if (!read_safe(fin, hp.n_encoder_layers)) return false;
    if (hp.n_encoder_layers != 4) {
        fprintf(stderr, "silero_vad: unexpected n_encoder_layers=%d (expected 4)\n",
                (int)hp.n_encoder_layers);
        return false;
    }
    for (int i = 0; i < hp.n_encoder_layers; i++) {
        if (!read_safe(fin, hp.encoder_in_channels[i])) return false;
        if (!read_safe(fin, hp.encoder_out_channels[i])) return false;
        if (!read_safe(fin, hp.kernel_sizes[i])) return false;
    }
    if (!read_safe(fin, hp.lstm_input_size)) return false;
    if (!read_safe(fin, hp.lstm_hidden_size)) return false;
    if (!read_safe(fin, hp.final_conv_in)) return false;
    if (!read_safe(fin, hp.final_conv_out)) return false;

    // Build tensor metadata in a no-alloc context, then allocate one backend
    // buffer for all weights and read data into it.
    const size_t n_tensors = 1 + 4 * 2 + 4 + 2;  // stft + enc + lstm + final
    ggml_init_params ip = { n_tensors * ggml_tensor_overhead(), nullptr, true };
    m.ctx = ggml_init(ip);
    if (!m.ctx) {
        fprintf(stderr, "silero_vad: failed to init ggml context\n");
        return false;
    }

    auto mk1 = [&](const char * name, ggml_type type, int n0) {
        ggml_tensor * t = ggml_new_tensor_1d(m.ctx, type, n0);
        m.t[name] = t;
        return t;
    };
    auto mk2 = [&](const char * name, ggml_type type, int n0, int n1) {
        ggml_tensor * t = ggml_new_tensor_2d(m.ctx, type, n0, n1);
        m.t[name] = t;
        return t;
    };
    auto mk3 = [&](const char * name, ggml_type type, int n0, int n1, int n2) {
        ggml_tensor * t = ggml_new_tensor_3d(m.ctx, type, n0, n1, n2);
        m.t[name] = t;
        return t;
    };

    // STFT basis: ggml dims [256, 1, 258] (torch [258, 1, 256] written reversed).
    mk3(TENSOR_STFT_BASIS, GGML_TYPE_F16, 256, 1, 258);

    // Encoder conv weights: ggml dims [kernel, in_ch, out_ch]. Biases: [out_ch].
    mk3(TENSOR_ENC_0_WEIGHT, GGML_TYPE_F16,
        hp.kernel_sizes[0], hp.encoder_in_channels[0], hp.encoder_out_channels[0]);
    mk1(TENSOR_ENC_0_BIAS, GGML_TYPE_F32, hp.encoder_out_channels[0]);
    mk3(TENSOR_ENC_1_WEIGHT, GGML_TYPE_F16,
        hp.kernel_sizes[1], hp.encoder_in_channels[1], hp.encoder_out_channels[1]);
    mk1(TENSOR_ENC_1_BIAS, GGML_TYPE_F32, hp.encoder_out_channels[1]);
    mk3(TENSOR_ENC_2_WEIGHT, GGML_TYPE_F16,
        hp.kernel_sizes[2], hp.encoder_in_channels[2], hp.encoder_out_channels[2]);
    mk1(TENSOR_ENC_2_BIAS, GGML_TYPE_F32, hp.encoder_out_channels[2]);
    mk3(TENSOR_ENC_3_WEIGHT, GGML_TYPE_F16,
        hp.kernel_sizes[3], hp.encoder_in_channels[3], hp.encoder_out_channels[3]);
    mk1(TENSOR_ENC_3_BIAS, GGML_TYPE_F32, hp.encoder_out_channels[3]);

    // LSTM weights: ggml [hidden, 4*hidden]. Biases: [4*hidden].
    const int hstate_dim = hp.lstm_hidden_size * 4;
    mk2(TENSOR_LSTM_WEIGHT_IH, GGML_TYPE_F32, hp.lstm_hidden_size, hstate_dim);
    mk2(TENSOR_LSTM_WEIGHT_HH, GGML_TYPE_F32, hp.lstm_hidden_size, hstate_dim);
    mk1(TENSOR_LSTM_BIAS_IH, GGML_TYPE_F32, hstate_dim);
    mk1(TENSOR_LSTM_BIAS_HH, GGML_TYPE_F32, hstate_dim);

    // Final conv: weight [final_conv_in, 1], bias [1].
    mk2(TENSOR_FINAL_CONV_WEIGHT, GGML_TYPE_F16, hp.final_conv_in, 1);
    mk1(TENSOR_FINAL_CONV_BIAS, GGML_TYPE_F32, 1);

    // Allocate one CPU buffer for all tensors.
    {
        ggml_backend_t tmp_backend = ggml_backend_cpu_init();
        if (!tmp_backend) {
            fprintf(stderr, "silero_vad: failed to init CPU backend for alloc\n");
            return false;
        }
        m.buffer = ggml_backend_alloc_ctx_tensors(m.ctx, tmp_backend);
        ggml_backend_free(tmp_backend);
    }
    if (!m.buffer) {
        fprintf(stderr, "silero_vad: failed to allocate weight buffer\n");
        return false;
    }

    // Read tensors one by one. The file lists each as:
    //   int32 n_dims, int32 name_len, int32 ttype, int32 ne[n_dims], char name[name_len], data
    int loaded = 0;
    while (true) {
        int32_t n_dims = 0;
        if (!read_safe(fin, n_dims)) break;
        if (fin.eof()) break;
        int32_t name_len = 0, ttype = 0;
        if (!read_safe(fin, name_len) || !read_safe(fin, ttype)) break;
        if (name_len <= 0 || name_len > 256 || n_dims < 0 || n_dims > 3) {
            fprintf(stderr, "silero_vad: bad tensor header (n_dims=%d name_len=%d)\n",
                    (int)n_dims, (int)name_len);
            return false;
        }

        int32_t ne[3] = {1, 1, 1};
        int32_t nelements = 1;
        for (int i = 0; i < n_dims; i++) {
            if (!read_safe(fin, ne[i])) return false;
            nelements *= ne[i];
        }
        // n_dims == 0 means a scalar (single element); the tensor metadata
        // for such tensors is created as a 1-D tensor of length 1, so the
        // ne[0]=1 default above already matches.
        std::vector<char> namebuf((size_t)name_len + 1, 0);
        if (!read_safe(fin, namebuf.data(), (size_t)name_len)) return false;
        std::string name(namebuf.data(), (size_t)name_len);

        auto it = m.t.find(name);
        if (it == m.t.end()) {
            fprintf(stderr, "silero_vad: unknown tensor '%s' in model file\n", name.c_str());
            return false;
        }
        ggml_tensor * t = it->second;
        if ((int)ggml_nelements(t) != nelements) {
            fprintf(stderr, "silero_vad: tensor '%s' size mismatch (file=%d model=%d)\n",
                    name.c_str(), (int)nelements, (int)ggml_nelements(t));
            return false;
        }
        for (int i = 0; i < n_dims; i++) {
            if (t->ne[i] != (int64_t)ne[i]) {
                fprintf(stderr, "silero_vad: tensor '%s' dim %d mismatch (file=%d model=%lld)\n",
                        name.c_str(), i, (int)ne[i], (long long)t->ne[i]);
                return false;
            }
        }
        (void)ttype;  // tensor type already fixed at creation time

        if (ggml_backend_buffer_is_host(t->buffer)) {
            if (!read_safe(fin, t->data, ggml_nbytes(t))) return false;
        } else {
            std::vector<char> tmp(ggml_nbytes(t));
            if (!read_safe(fin, tmp.data(), tmp.size())) return false;
            ggml_backend_tensor_set(t, tmp.data(), 0, ggml_nbytes(t));
        }
        maybe_byteswap_tensor(t);
        loaded++;
    }

    if (loaded != (int)m.t.size()) {
        fprintf(stderr, "silero_vad: expected %zu tensors, got %d\n", m.t.size(), loaded);
        return false;
    }

    fprintf(stderr, "silero_vad: loaded %s (window=%d context=%d lstm_hidden=%d tensors=%d)\n",
            path.c_str(), (int)m.n_window, (int)m.n_context,
            (int)hp.lstm_hidden_size, loaded);
    return true;
}

inline void silero_vad_free(vad & m) {
    if (m.buffer) {
        ggml_backend_buffer_free(m.buffer);
        m.buffer = nullptr;
    }
    if (m.ctx) {
        ggml_free(m.ctx);
        m.ctx = nullptr;
    }
    m.t.clear();
}

// ---- Graph builders ----

// STFT layer: reflect-pad input by n_context on each side, conv_1d with the
// stft_forward_basis, then compute magnitude of the first cutoff bins.
static inline ggml_tensor * build_stft_layer(ggml_context * ctx0,
                                             const vad & m, ggml_tensor * cur) {
    ggml_tensor * padded = ggml_pad_reflect_1d(ctx0, cur, m.n_context, m.n_context);
    // ggml_conv_1d(ctx, a=kernel, b=data, s0=stride, p0=pad, d0=dilation).
    // whisper.cpp passes lstm_input_size as the stride (STFT hop).
    ggml_tensor * stft = ggml_conv_1d(ctx0, m.g(TENSOR_STFT_BASIS), padded,
                                      m.hparams.lstm_input_size, 0, 1);
    int cutoff = (int)(m.g(TENSOR_STFT_BASIS)->ne[2] / 2);
    ggml_tensor * real_part = ggml_view_2d(ctx0, stft, 4, cutoff, stft->nb[1], 0);
    ggml_tensor * img_part  = ggml_view_2d(ctx0, stft, 4, cutoff, stft->nb[1],
                                           (size_t)cutoff * stft->nb[1]);
    ggml_tensor * real_sq = ggml_mul(ctx0, real_part, real_part);
    ggml_tensor * img_sq  = ggml_mul(ctx0, img_part,  img_part);
    ggml_tensor * sum_sq  = ggml_add(ctx0, real_sq, img_sq);
    return ggml_sqrt(ctx0, sum_sq);
}

// 4-layer Conv1D + ReLU encoder.
static inline ggml_tensor * build_encoder_layer(ggml_context * ctx0,
                                                const vad & m, ggml_tensor * cur) {
    cur = ggml_conv_1d(ctx0, m.g(TENSOR_ENC_0_WEIGHT), cur, 1, 1, 1);
    cur = ggml_add(ctx0, cur, ggml_reshape_3d(ctx0, m.g(TENSOR_ENC_0_BIAS), 1, 128, 1));
    cur = ggml_relu(ctx0, cur);
    cur = ggml_conv_1d(ctx0, m.g(TENSOR_ENC_1_WEIGHT), cur, 2, 1, 1);
    cur = ggml_add(ctx0, cur, ggml_reshape_3d(ctx0, m.g(TENSOR_ENC_1_BIAS), 1, 64, 1));
    cur = ggml_relu(ctx0, cur);
    cur = ggml_conv_1d(ctx0, m.g(TENSOR_ENC_2_WEIGHT), cur, 2, 1, 1);
    cur = ggml_add(ctx0, cur, ggml_reshape_3d(ctx0, m.g(TENSOR_ENC_2_BIAS), 1, 64, 1));
    cur = ggml_relu(ctx0, cur);
    cur = ggml_conv_1d(ctx0, m.g(TENSOR_ENC_3_WEIGHT), cur, 1, 1, 1);
    cur = ggml_add(ctx0, cur, ggml_reshape_3d(ctx0, m.g(TENSOR_ENC_3_BIAS), 1, 128, 1));
    cur = ggml_relu(ctx0, cur);
    return cur;
}

// LSTM step. Updates h_state/c_state (persistent tensors in state_ctx) and
// returns the new hidden state. The ggml_cpy nodes write the new h/c back into
// the persistent state tensors so callers can read them after compute.
static inline ggml_tensor * build_lstm_layer(ggml_context * ctx0,
                                             const vad & m,
                                             ggml_tensor * cur,
                                             ggml_tensor * h_state,
                                             ggml_tensor * c_state,
                                             ggml_cgraph * gf) {
    const int hdim = m.hparams.lstm_hidden_size;

    ggml_tensor * x_t = ggml_transpose(ctx0, cur);

    ggml_tensor * inp_gate = ggml_mul_mat(ctx0, m.g(TENSOR_LSTM_WEIGHT_IH), x_t);
    inp_gate = ggml_add(ctx0, inp_gate, m.g(TENSOR_LSTM_BIAS_IH));
    ggml_tensor * hid_gate = ggml_mul_mat(ctx0, m.g(TENSOR_LSTM_WEIGHT_HH), h_state);
    hid_gate = ggml_add(ctx0, hid_gate, m.g(TENSOR_LSTM_BIAS_HH));
    ggml_tensor * out_gate = ggml_add(ctx0, inp_gate, hid_gate);

    const size_t hdim_size = ggml_row_size(out_gate->type, hdim);
    ggml_tensor * i_t = ggml_sigmoid(ctx0, ggml_view_1d(ctx0, out_gate, hdim, 0 * hdim_size));
    ggml_tensor * f_t = ggml_sigmoid(ctx0, ggml_view_1d(ctx0, out_gate, hdim, 1 * hdim_size));
    ggml_tensor * g_t = ggml_tanh  (ctx0, ggml_view_1d(ctx0, out_gate, hdim, 2 * hdim_size));
    ggml_tensor * o_t = ggml_sigmoid(ctx0, ggml_view_1d(ctx0, out_gate, hdim, 3 * hdim_size));

    ggml_tensor * c_out = ggml_add(ctx0,
        ggml_mul(ctx0, f_t, c_state),
        ggml_mul(ctx0, i_t, g_t));
    ggml_build_forward_expand(gf, ggml_cpy(ctx0, c_out, c_state));

    ggml_tensor * out = ggml_mul(ctx0, o_t, ggml_tanh(ctx0, c_out));
    ggml_build_forward_expand(gf, ggml_cpy(ctx0, out, h_state));
    return out;
}

// Per-window VAD graph. The frame input is [n_window, 1]; the LSTM state
// tensors (h_state, c_state) are marked as graph inputs so the gallocr
// allocates them in its compute buffer. The LSTM step writes the new state
// back into h_state/c_state via ggml_cpy nodes, so after compute the caller
// can read the updated state out of those tensors. Because we re-allocate
// the graph every window (graph structure is identical and tiny), the
// caller must set h_state/c_state from host-side state before each compute.
struct vad_graph {
    ggml_context * ctx = nullptr;
    ggml_cgraph * gf = nullptr;
    ggml_tensor * frame = nullptr;
    ggml_tensor * prob = nullptr;
    ggml_tensor * h_state = nullptr;
    ggml_tensor * c_state = nullptr;
};

static inline void build_window_graph(vad_graph & g, const vad & m,
                                      size_t graph_size) {
    if (g.ctx) ggml_free(g.ctx);
    ggml_init_params ip = { graph_size, nullptr, true };
    g.ctx = ggml_init(ip);
    g.gf = ggml_new_graph(g.ctx);

    g.frame = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, (int64_t)m.n_window, 1);
    ggml_set_name(g.frame, "frame");
    ggml_set_input(g.frame);

    const int hdim = m.hparams.lstm_hidden_size;
    g.h_state = ggml_new_tensor_1d(g.ctx, GGML_TYPE_F32, hdim);
    ggml_set_name(g.h_state, "h_state");
    g.c_state = ggml_new_tensor_1d(g.ctx, GGML_TYPE_F32, hdim);
    ggml_set_name(g.c_state, "c_state");
    // Mark as inputs: gallocr allocates them and the caller sets them before
    // compute. The LSTM step's ggml_cpy nodes write the new state into these
    // tensors, which the caller reads back after compute.
    ggml_set_input(g.h_state);
    ggml_set_input(g.c_state);

    ggml_tensor * cur = build_stft_layer(g.ctx, m, g.frame);
    cur = build_encoder_layer(g.ctx, m, cur);
    // Take the first time step: [:, :, 0]  -> [1, 128].
    cur = ggml_view_2d(g.ctx, cur, 1, 128, cur->nb[1], 0);
    cur = build_lstm_layer(g.ctx, m, cur, g.h_state, g.c_state, g.gf);
    cur = ggml_relu(g.ctx, cur);
    cur = ggml_conv_1d(g.ctx, m.g(TENSOR_FINAL_CONV_WEIGHT), cur, 1, 0, 1);
    cur = ggml_add(g.ctx, cur, m.g(TENSOR_FINAL_CONV_BIAS));
    cur = ggml_sigmoid(g.ctx, cur);
    g.prob = cur;
    ggml_set_name(g.prob, "prob");
    ggml_set_output(g.prob);
    ggml_build_forward_expand(g.gf, g.prob);
}

// Run the model on a single window of n_window samples, updating LSTM state.
// `window` must contain exactly n_window samples (zero-pad the last chunk).
// Returns the speech probability for this window, or -1 on compute failure.
static inline float run_window(vad & m, vad_graph & g,
                               ggml_gallocr_t gallocr,
                               ggml_backend_t backend,
                               const float * window,
                               const float * h_in, const float * c_in,
                               float * h_out, float * c_out,
                               int n_threads) {
    ggml_gallocr_alloc_graph(gallocr, g.gf);
    ggml_backend_tensor_set(g.frame, window, 0, (size_t)m.n_window * sizeof(float));
    ggml_backend_tensor_set(g.h_state, h_in, 0,
                            (size_t)m.hparams.lstm_hidden_size * sizeof(float));
    ggml_backend_tensor_set(g.c_state, c_in, 0,
                            (size_t)m.hparams.lstm_hidden_size * sizeof(float));

    if (n_threads > 0) {
        ggml_backend_cpu_set_n_threads(backend, n_threads);
    }
    bool ok = ggml_backend_graph_compute(backend, g.gf) == GGML_STATUS_SUCCESS;
    if (!ok) return -1.0f;

    float prob = 0.0f;
    ggml_backend_tensor_get(g.prob, &prob, 0, sizeof(float));
    if (h_out) ggml_backend_tensor_get(g.h_state, h_out, 0,
                                       (size_t)m.hparams.lstm_hidden_size * sizeof(float));
    if (c_out) ggml_backend_tensor_get(g.c_state, c_out, 0,
                                       (size_t)m.hparams.lstm_hidden_size * sizeof(float));
    return prob;
}

// ---- Segment state machine (ported from whisper.cpp's
//      whisper_vad_segments_from_probs) ----

// Converts a probability sequence into speech segments [start_ms, end_ms].
// `probs[i]` corresponds to window i covering samples [i*n_window, (i+1)*n_window).
//
// `finalize_open` controls how a speech run that is still in progress at the
// end of the probability sequence is handled:
//   - true  (batch mode): finalize it as a segment ending at audio_length_samples.
//     This matches whisper.cpp, which sees a complete clip.
//   - false (streaming mode): leave it out of `segs` so the caller can report
//     it separately as an in-progress (open) segment. Without this, a still-
//     talking speaker would be reported as a completed segment on every feed,
//     causing premature cutoff and making min_silence_duration_ms ineffective.
inline void probs_to_segments(const std::vector<float> & probs,
                              const vad_params & vp,
                              int n_window,
                              std::vector<std::pair<int,int>> & segs,
                              bool finalize_open = true) {
    segs.clear();
    const int n_probs = (int)probs.size();
    const int sample_rate = 16000;

    const int min_silence_samples     = sample_rate * vp.min_silence_duration_ms / 1000;
    const int min_speech_samples      = sample_rate * vp.min_speech_duration_ms  / 1000;
    const int speech_pad_samples      = sample_rate * vp.speech_pad_ms           / 1000;
    const int audio_length_samples    = n_probs * n_window;
    const int min_silence_samples_at_max_speech = sample_rate * 98 / 1000;

    int max_speech_samples;
    if (vp.max_speech_duration_s > 100000.0f) {
        max_speech_samples = INT_MAX / 2;
    } else {
        int64_t tmp = (int64_t)sample_rate * (int64_t)vp.max_speech_duration_s
                      - n_window - 2 * speech_pad_samples;
        max_speech_samples = (tmp > INT_MAX) ? INT_MAX / 2 : (int)tmp;
        if (max_speech_samples < 0) max_speech_samples = INT_MAX / 2;
    }

    float neg_threshold = vp.threshold - 0.15f;
    if (neg_threshold < 0.01f) neg_threshold = 0.01f;

    struct seg { int start; int end; };
    std::vector<seg> speeches;
    speeches.reserve(256);

    bool is_speech_segment = false;
    int  temp_end          = 0;
    int  prev_end          = 0;
    int  next_start        = 0;
    int  curr_speech_start = 0;
    bool has_curr_speech   = false;

    for (int i = 0; i < n_probs; i++) {
        float curr_prob   = probs[i];
        int   curr_sample = n_window * i;

        if ((curr_prob >= vp.threshold) && temp_end) {
            temp_end = 0;
            if (next_start < prev_end) next_start = curr_sample;
        }
        if ((curr_prob >= vp.threshold) && !is_speech_segment) {
            is_speech_segment = true;
            curr_speech_start = curr_sample;
            has_curr_speech = true;
            continue;
        }
        if (is_speech_segment && (curr_sample - curr_speech_start) > max_speech_samples) {
            if (prev_end) {
                speeches.push_back({ curr_speech_start, prev_end });
                has_curr_speech = true;
                if (next_start < prev_end) {
                    is_speech_segment = false;
                    has_curr_speech = false;
                } else {
                    curr_speech_start = next_start;
                }
                prev_end = next_start = temp_end = 0;
            } else {
                speeches.push_back({ curr_speech_start, curr_sample });
                prev_end = next_start = temp_end = 0;
                is_speech_segment = false;
                has_curr_speech = false;
                continue;
            }
        }
        if ((curr_prob < neg_threshold) && is_speech_segment) {
            if (!temp_end) temp_end = curr_sample;
            if ((curr_sample - temp_end) > min_silence_samples_at_max_speech) {
                prev_end = temp_end;
            }
            if ((curr_sample - temp_end) < min_silence_samples) {
                continue;
            } else {
                if ((temp_end - curr_speech_start) > min_speech_samples) {
                    speeches.push_back({ curr_speech_start, temp_end });
                }
                prev_end = next_start = temp_end = 0;
                is_speech_segment = false;
                has_curr_speech = false;
                continue;
            }
        }
    }

    // Finalize a still-in-progress speech run at the end of the clip. In batch
    // mode this is correct (the clip has ended). In streaming mode we skip this
    // so the run is reported as an open segment instead of a completed one.
    if (finalize_open && has_curr_speech &&
        (audio_length_samples - curr_speech_start) > min_speech_samples) {
        speeches.push_back({ curr_speech_start, audio_length_samples });
    }

    // Merge adjacent segments with small gaps (<200ms).
    if (speeches.size() > 1) {
        const int max_merge_gap_samples = sample_rate * 200 / 1000;
        for (int i = 0; i < (int)speeches.size() - 1; i++) {
            if (speeches[i + 1].start - speeches[i].end < max_merge_gap_samples) {
                speeches[i].end = speeches[i + 1].end;
                speeches.erase(speeches.begin() + i + 1);
                i--;
            }
        }
    }

    // Drop segments shorter than min speech duration.
    for (int i = 0; i < (int)speeches.size(); i++) {
        if (speeches[i].end - speeches[i].start < min_speech_samples) {
            speeches.erase(speeches.begin() + i);
            i--;
        }
    }

    // Apply speech padding and convert to [start_ms, end_ms].
    segs.reserve(speeches.size());
    for (int i = 0; i < (int)speeches.size(); i++) {
        if (i == 0) {
            speeches[i].start = (speeches[i].start > speech_pad_samples)
                                ? (speeches[i].start - speech_pad_samples) : 0;
        }
        if (i < (int)speeches.size() - 1) {
            int silence_duration = speeches[i + 1].start - speeches[i].end;
            if (silence_duration < 2 * speech_pad_samples) {
                speeches[i].end += silence_duration / 2;
                speeches[i + 1].start = (speeches[i + 1].start > silence_duration / 2)
                                        ? (speeches[i + 1].start - silence_duration / 2) : 0;
            } else {
                speeches[i].end = (speeches[i].end + speech_pad_samples < audio_length_samples)
                                 ? (speeches[i].end + speech_pad_samples) : audio_length_samples;
                speeches[i + 1].start = (speeches[i + 1].start > speech_pad_samples)
                                        ? (speeches[i + 1].start - speech_pad_samples) : 0;
            }
        } else {
            speeches[i].end = (speeches[i].end + speech_pad_samples < audio_length_samples)
                             ? (speeches[i].end + speech_pad_samples) : audio_length_samples;
        }
        segs.push_back({ samples_to_ms(speeches[i].start), samples_to_ms(speeches[i].end) });
    }
}

// ---- Batch (non-streaming) API ----

// Run Silero VAD on a 16k mono float waveform using a pre-loaded model and
// the full set of VAD parameters (threshold, min speech/silence durations,
// max speech duration, speech padding).
inline bool silero_vad_segments_from_model(vad & m, const std::vector<float> & wav,
                                           const vad_params & vp,
                                           std::vector<std::pair<int,int>> & segs,
                                           int nthreads = 8) {
    segs.clear();
    const int n_window = m.n_window;
    const int n_samples = (int)wav.size();
    if (n_samples <= 0) return true;

    int n_chunks = n_samples / n_window;
    if (n_samples % n_window != 0) n_chunks += 1;

    std::vector<float> probs((size_t)n_chunks, 0.0f);
    std::vector<float> window((size_t)n_window, 0.0f);

    std::vector<float> h_state((size_t)m.hparams.lstm_hidden_size, 0.0f);
    std::vector<float> c_state((size_t)m.hparams.lstm_hidden_size, 0.0f);
    std::vector<float> h_out((size_t)m.hparams.lstm_hidden_size, 0.0f);
    std::vector<float> c_out((size_t)m.hparams.lstm_hidden_size, 0.0f);

    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) {
        fprintf(stderr, "silero_vad: failed to init CPU backend\n");
        return false;
    }
    ggml_gallocr_t gallocr = ggml_gallocr_new(ggml_backend_cpu_buffer_type());

    vad_graph g;
    build_window_graph(g, m, (size_t)4 * 1024 * 1024);

    for (int i = 0; i < n_chunks; i++) {
        int idx_start = i * n_window;
        int idx_end = std::min(idx_start + n_window, n_samples);
        std::fill(window.begin(), window.end(), 0.0f);
        std::copy(wav.data() + idx_start, wav.data() + idx_end, window.begin());

        float prob = run_window(m, g, gallocr, backend, window.data(),
                                h_state.data(), c_state.data(),
                                h_out.data(), c_out.data(), nthreads);
        if (prob < 0.0f) {
            fprintf(stderr, "silero_vad: graph compute failed at chunk %d\n", i);
            break;
        }
        probs[i] = prob;
        std::swap(h_state, h_out);
        std::swap(c_state, c_out);
    }

    probs_to_segments(probs, vp, n_window, segs);

    if (g.ctx) ggml_free(g.ctx);
    ggml_gallocr_free(gallocr);
    ggml_backend_free(backend);
    return true;
}

// Legacy overload: max_seg_ms caps a single segment (e.g. 30000); <=0 = unbounded.
// All other parameters use whisper.cpp defaults.
inline bool silero_vad_segments_from_model(vad & m, const std::vector<float> & wav,
                                           int max_seg_ms,
                                           std::vector<std::pair<int,int>> & segs,
                                           int nthreads = 8) {
    vad_params vp;
    if (max_seg_ms > 0) vp.max_speech_duration_s = max_seg_ms / 1000.0f;
    return silero_vad_segments_from_model(m, wav, vp, segs, nthreads);
}

// Load + run Silero VAD on a 16k mono float waveform with full VAD params.
inline bool silero_vad_segments(const std::string & path,
                                const std::vector<float> & wav,
                                const vad_params & vp,
                                std::vector<std::pair<int,int>> & segs,
                                int nthreads = 8) {
    vad m;
    if (!silero_vad_load(m, path)) return false;
    bool ok = silero_vad_segments_from_model(m, wav, vp, segs, nthreads);
    silero_vad_free(m);
    return ok;
}

// Legacy convenience wrapper using only max_seg_ms.
inline bool silero_vad_segments(const std::string & path,
                                const std::vector<float> & wav,
                                int max_seg_ms,
                                std::vector<std::pair<int,int>> & segs,
                                int nthreads = 8) {
    vad m;
    if (!silero_vad_load(m, path)) return false;
    bool ok = silero_vad_segments_from_model(m, wav, max_seg_ms, segs, nthreads);
    silero_vad_free(m);
    return ok;
}

// ===== Incremental (streaming) VAD =====
// Buffers incoming PCM, runs one window at a time keeping LSTM state, appends
// per-window probabilities, and re-runs the segment state machine each call so
// the merge/pad post-processing stays globally consistent.

struct vad_inc_state {
    ggml_backend_t backend = nullptr;
    ggml_gallocr_t gallocr = nullptr;

    vad_graph graph;
    static constexpr size_t kGraphSize = (size_t)4 * 1024 * 1024;

    std::vector<float> h_state;
    std::vector<float> c_state;
    std::vector<float> h_out;
    std::vector<float> c_out;

    std::vector<float> pcm_buf;        // unprocessed samples (< n_window)
    std::vector<float> probs;          // per-window probabilities
    int n_window = 512;

    vad_params vp;
    std::vector<std::pair<int,int>> completed_segments;
    std::pair<int,int> open_segment = {-1, -1};

    bool initialized = false;
};

// Initialize incremental VAD state with a pre-loaded model and the full
// set of VAD parameters. The params are stored in the state and used by the
// segment state machine on every feed.
inline bool vad_inc_init(vad_inc_state * s, vad & m, const vad_params & vp) {
    s->backend = ggml_backend_cpu_init();
    if (!s->backend) {
        fprintf(stderr, "silero_vad: vad_inc_init: failed to init CPU backend\n");
        return false;
    }
    s->gallocr = ggml_gallocr_new(ggml_backend_cpu_buffer_type());

    s->n_window = m.n_window;
    s->h_state.assign((size_t)m.hparams.lstm_hidden_size, 0.0f);
    s->c_state.assign((size_t)m.hparams.lstm_hidden_size, 0.0f);
    s->h_out.assign((size_t)m.hparams.lstm_hidden_size, 0.0f);
    s->c_out.assign((size_t)m.hparams.lstm_hidden_size, 0.0f);

    build_window_graph(s->graph, m, vad_inc_state::kGraphSize);

    s->vp = vp;
    s->initialized = true;
    return true;
}

// Legacy overload: only max_seg_ms is configurable; other params use defaults.
inline bool vad_inc_init(vad_inc_state * s, vad & m, int max_seg_ms = 30000) {
    vad_params vp;
    if (max_seg_ms > 0) vp.max_speech_duration_s = max_seg_ms / 1000.0f;
    return vad_inc_init(s, m, vp);
}

// Feed new audio samples. Returns the number of new windows processed (>=0),
// or -1 on error.
inline int vad_inc_feed(vad_inc_state * s, vad & m,
                        const float * pcm, int n_samples, int nthreads = 8) {
    if (!s->initialized || !pcm || n_samples <= 0) return -1;

    s->pcm_buf.insert(s->pcm_buf.end(), pcm, pcm + n_samples);

    int new_windows = 0;
    while ((int)s->pcm_buf.size() >= s->n_window) {
        std::vector<float> window(s->pcm_buf.begin(),
                                  s->pcm_buf.begin() + s->n_window);
        s->pcm_buf.erase(s->pcm_buf.begin(),
                         s->pcm_buf.begin() + s->n_window);

        float prob = run_window(m, s->graph, s->gallocr, s->backend,
                                window.data(),
                                s->h_state.data(), s->c_state.data(),
                                s->h_out.data(), s->c_out.data(), nthreads);
        if (prob < 0.0f) {
            fprintf(stderr, "silero_vad: vad_inc_feed: compute failed\n");
            return -1;
        }
        s->probs.push_back(prob);
        std::swap(s->h_state, s->h_out);
        std::swap(s->c_state, s->c_out);
        new_windows++;
    }

    // Re-run the segment state machine over the full accumulated probability
    // vector so the merge/padding post-processing stays globally consistent.
    // finalize_open=false: a speech run still in progress at the end must NOT
    // be reported as a completed segment (that would cut off a still-talking
    // speaker and make min_silence_duration_ms ineffective). It is reported
    // separately as the open segment below.
    std::vector<std::pair<int,int>> segs;
    probs_to_segments(s->probs, s->vp, s->n_window, segs, /*finalize_open=*/false);
    s->completed_segments = segs;

    // Derive the (still-open) speech segment by scanning the probability
    // sequence for an in-progress speech run that hasn't been finalized by a
    // long-enough silence. This mirrors the state machine's
    // `curr_speech_start` at the end of the consumed sequence.
    const float neg_threshold = [t = s->vp.threshold]() {
        float nt = t - 0.15f;
        return nt < 0.01f ? 0.01f : nt;
    }();
    const int min_silence_samples = 16000 * s->vp.min_silence_duration_ms / 1000;
    const int total_samples = (int)s->probs.size() * s->n_window;

    bool in_speech = false;
    int open_start_sample = -1;
    {
        bool is_seg = false;
        int temp_end = 0;
        int curr_start = 0;
        for (int i = 0; i < (int)s->probs.size(); i++) {
            const float p = s->probs[i];
            const int cs = s->n_window * i;
            if (p >= s->vp.threshold && !is_seg) {
                is_seg = true;
                curr_start = cs;
                temp_end = 0;
            } else if (p < neg_threshold && is_seg) {
                if (!temp_end) temp_end = cs;
                if ((cs - temp_end) >= min_silence_samples) {
                    is_seg = false;
                    temp_end = 0;
                }
            } else if (p >= s->vp.threshold && temp_end) {
                temp_end = 0;
            }
        }
        if (is_seg) {
            in_speech = true;
            open_start_sample = curr_start;
        }
    }
    if (in_speech) {
        // Only report an open segment if it starts strictly after the last
        // finalized segment's end. Otherwise the last finalized segment
        // already covers the trailing speech (probs_to_segments finalizes a
        // trailing speech run at audio_length_samples), and reporting an
        // overlapping open segment would duplicate the audio for callers
        // that append open_segment to get_segments().
        int last_end_ms = -1;
        if (!s->completed_segments.empty()) {
            last_end_ms = s->completed_segments.back().second;
        }
        int open_start_ms = samples_to_ms(open_start_sample);
        if (open_start_ms > last_end_ms) {
            s->open_segment = { open_start_ms, samples_to_ms(total_samples) };
        } else {
            s->open_segment = {-1, -1};
        }
    } else {
        s->open_segment = {-1, -1};
    }

    return new_windows;
}

inline void vad_inc_get_segments(vad_inc_state * s,
                                std::vector<std::pair<int,int>> & segs) {
    segs = s->completed_segments;
}

inline bool vad_inc_get_open_segment(vad_inc_state * s, int & start_ms, int & end_ms) {
    if (s->open_segment.first < 0) return false;
    start_ms = s->open_segment.first;
    end_ms   = s->open_segment.second;
    return true;
}

inline void vad_inc_reset(vad_inc_state * s) {
    s->pcm_buf.clear();
    s->probs.clear();
    std::fill(s->h_state.begin(), s->h_state.end(), 0.0f);
    std::fill(s->c_state.begin(), s->c_state.end(), 0.0f);
    s->completed_segments.clear();
    s->open_segment = {-1, -1};
}

inline void vad_inc_free(vad_inc_state * s) {
    if (s->graph.ctx) {
        ggml_free(s->graph.ctx);
        s->graph.ctx = nullptr;
    }
    if (s->gallocr) {
        ggml_gallocr_free(s->gallocr);
        s->gallocr = nullptr;
    }
    if (s->backend) {
        ggml_backend_free(s->backend);
        s->backend = nullptr;
    }
    s->initialized = false;
}

} // namespace silero_vad_impl
