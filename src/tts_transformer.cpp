#include "tts_transformer.h"
#include "transformer/transformer_state_internal.h"
#include "ggml-backend.h"

namespace qwen3_tts {

TTSTransformer::TTSTransformer()
    : impl_(std::make_unique<tts_transformer_private>()) {
}

TTSTransformer::~TTSTransformer() {
    default_session_.reset();
    unload_model();
}

const tts_transformer_config & TTSTransformer::get_config() const {
    return impl_->model.config;
}

std::unique_ptr<TTSTransformerSession> TTSTransformer::create_session() {
    auto session = std::make_unique<TTSTransformerSession>();
    auto & st = session->state_;

    st.backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU, nullptr);
    if (!st.backend) {
        st.backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    }
    if (!st.backend) {
        st.backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_ACCEL, nullptr);
    }
    if (!st.backend) {
        st.backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    }
    if (!st.backend) {
        error_msg_ = "Failed to create session backend";
        return nullptr;
    }
    session->own_backend_ = true;

    st.backend_cpu = impl_->state.backend_cpu;

    std::vector<ggml_backend_t> backends;
    backends.push_back(st.backend);
    if (st.backend_cpu) {
        backends.push_back(st.backend_cpu);
    }
    st.sched = ggml_backend_sched_new(backends.data(), nullptr, (int) backends.size(), QWEN3_TTS_MAX_NODES, false, true);
    if (!st.sched) {
        error_msg_ = "Failed to create session scheduler";
        return nullptr;
    }

    const size_t meta_size = ggml_tensor_overhead() * QWEN3_TTS_MAX_NODES + ggml_graph_overhead();
    st.compute_meta.resize(meta_size);
    st.code_pred_compute_meta.resize(15);
    for (int i = 0; i < 15; ++i) {
        st.code_pred_compute_meta[i].resize(meta_size);
    }

    session->initialized_ = true;
    return session;
}

void TTSTransformer::ensure_default_session() {
    if (!default_session_) {
        default_session_ = create_session();
    }
}

bool TTSTransformer::forward(const int32_t * tokens, int32_t n_tokens, int32_t n_past,
                             std::vector<float> & output) {
    return forward_text(tokens, n_tokens, nullptr, n_past, output);
}

bool TTSTransformer::forward_with_audio(const int32_t * tokens, int32_t n_tokens,
                                        const float * audio_embd, int32_t n_audio,
                                        int32_t audio_start_pos, int32_t n_past,
                                        std::vector<float> & output) {
    (void) audio_embd;
    (void) n_audio;
    (void) audio_start_pos;
    return forward_text(tokens, n_tokens, nullptr, n_past, output);
}

void free_transformer_model(tts_transformer_model & model) {
    if (model.buffer) {
        ggml_backend_buffer_free(model.buffer);
        model.buffer = nullptr;
    }
    if (model.ctx) {
        ggml_free(model.ctx);
        model.ctx = nullptr;
    }
    model.tensors.clear();
    model.layers.clear();
    model.code_pred_layers.clear();
    model.code_pred_small_to_mtp_weight = nullptr;
    model.code_pred_small_to_mtp_bias = nullptr;
    model.code_pred_output_norm = nullptr;
    model.code_pred_embd.clear();
    model.code_pred_head.clear();
}

void free_tts_kv_cache(tts_kv_cache & cache) {
    if (cache.buffer) {
        ggml_backend_buffer_free(cache.buffer);
        cache.buffer = nullptr;
    }
    if (cache.ctx) {
        ggml_free(cache.ctx);
        cache.ctx = nullptr;
    }
    cache.k_cache.clear();
    cache.v_cache.clear();
    cache.n_ctx = 0;
    cache.n_used = 0;
}

} // namespace qwen3_tts
