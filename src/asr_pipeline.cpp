// asr_pipeline.cpp — Implementation of the unified ASR/VAD API.
// Delegates to the funasr single-header libraries (funasr_vad.h, sensevoice_asr.h, paraformer_asr.h).

#include "asr_pipeline.h"
#include "funasr_vad.h"
#include "sensevoice_asr.h"
#include "paraformer_asr.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>

namespace qwen3_tts {

static int64_t now_us() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

std::vector<float> resample_to_16k(const float* samples, int n_samples, int source_rate) {
    if (source_rate == 16000) {
        return std::vector<float>(samples, samples + n_samples);
    }
    if (n_samples <= 0 || source_rate <= 0) return {};

    const int target_rate = 16000;
    int out_len = (int)((int64_t)n_samples * target_rate / source_rate);
    std::vector<float> out(out_len);
    double ratio = (double)source_rate / target_rate;
    for (int i = 0; i < out_len; i++) {
        double src_pos = i * ratio;
        int idx = (int)src_pos;
        float frac = (float)(src_pos - idx);
        if (idx + 1 < n_samples) {
            out[i] = samples[idx] * (1.0f - frac) + samples[idx + 1] * frac;
        } else if (idx < n_samples) {
            out[i] = samples[idx];
        }
    }
    return out;
}

static int64_t now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

// ===== VAD =====

bool run_vad(const std::string& vad_gguf,
             const std::vector<float>& pcm_16k,
             std::vector<vad_segment>& segments,
             int maxseg_ms) {
    std::vector<std::pair<int,int>> raw;
    if (!funasr_vad_segments(vad_gguf, pcm_16k, maxseg_ms, raw)) {
        return false;
    }
    segments.clear();
    segments.reserve(raw.size());
    for (auto& p : raw) {
        segments.push_back({p.first, p.second});
    }
    return true;
}

// ===== SenseVoice =====

asr_result transcribe_sensevoice_with_model(const sensevoice::model& m,
                                            const std::vector<int>& qtok,
                                            const std::vector<std::string>& vocab,
                                            const std::vector<float>& pcm_16k,
                                            const asr_params& params) {
    asr_result result;
    int64_t t0 = now_ms();

    bool emit_ids = params.output_ids || vocab.empty();
    float* emb = (float*)((sensevoice::model&)m).g("embed.weight")->data;
    int nq = (int)qtok.size();
    auto mel_fb = sensevoice::mel_filterbank();

    auto run_and_collect = [&](const std::vector<float>& fb, int T) {
        auto ids = sensevoice::run_seg((sensevoice::model&)m, qtok.data(), nq, emb, fb, T, params.n_threads);
        if (emit_ids) {
            result.token_ids.insert(result.token_ids.end(), ids.begin(), ids.end());
        } else {
            std::string t = sensevoice::detok_sv(ids, vocab, params.keep_tags);
            if (!result.text.empty() && !t.empty()) result.text += " ";
            result.text += t;
        }
    };

    int t = 0;
    int64_t tf0 = now_us();
    auto fb = sensevoice::compute_fbank(pcm_16k, t, &mel_fb);
    int64_t tf1 = now_us();
    fprintf(stderr, "asr_pipeline: pcm_samples=%zu fbank_T=%d fbank_time=%.1fms nq=%d\n",
            pcm_16k.size(), t, (tf1-tf0)/1000.0, nq);
    if (t > 0) run_and_collect(fb, t);

    result.t_total_ms = now_ms() - t0;
    result.success = true;
    return result;
}

asr_result transcribe_sensevoice(const std::string& model_gguf,
                                 const std::vector<float>& pcm_16k,
                                 const asr_params& params) {
    asr_result result;
    int64_t t0 = now_ms();

    sensevoice::model m;
    std::vector<int> qtok;
    std::vector<std::string> vocab;
    if (!sensevoice::sensevoice_load(model_gguf, m, qtok, vocab)) {
        result.error_msg = "failed to load SenseVoice model: " + model_gguf;
        return result;
    }

    // Handle VAD segmentation if requested
    if (!params.vad_model_path.empty()) {
        std::vector<vad_segment> segments;
        if (!run_vad(params.vad_model_path, pcm_16k, segments, params.vad_maxseg)) {
            result.error_msg = "VAD failed";
            if (m.ctx_w) ggml_free(m.ctx_w);
            return result;
        }

        auto mel_fb = sensevoice::mel_filterbank();
        bool emit_ids = params.output_ids || vocab.empty();
        float* emb = (float*)m.g("embed.weight")->data;
        int nq = (int)qtok.size();

        for (auto& s : segments) {
            int off = (int)((int64_t)s.start_ms * 16000 / 1000);
            int end = (int)((int64_t)s.end_ms * 16000 / 1000);
            if (end > (int)pcm_16k.size()) end = (int)pcm_16k.size();
            if (end - off < sensevoice::WINLEN) continue;
            std::vector<float> seg(pcm_16k.begin() + off, pcm_16k.begin() + end);
            int t = 0;
            auto fb = sensevoice::compute_fbank(seg, t, &mel_fb);
            if (t > 0) {
                result.segments.push_back(s);
                auto ids = sensevoice::run_seg(m, qtok.data(), nq, emb, fb, t, params.n_threads);
                if (emit_ids) {
                    result.token_ids.insert(result.token_ids.end(), ids.begin(), ids.end());
                } else {
                    std::string txt = sensevoice::detok_sv(ids, vocab, params.keep_tags);
                    if (!result.text.empty() && !txt.empty()) result.text += " ";
                    result.text += txt;
                }
            }
        }
    } else {
        result = transcribe_sensevoice_with_model(m, qtok, vocab, pcm_16k, params);
    }

    if (m.ctx_w) ggml_free(m.ctx_w);
    result.t_total_ms = now_ms() - t0;
    return result;
}

// ===== Paraformer =====

asr_result transcribe_paraformer_with_model(const paraformer::model& m,
                                            const std::vector<std::string>& vocab,
                                            const std::vector<float>& pcm_16k,
                                            const asr_params& params) {
    asr_result result;
    int64_t t0 = now_ms();

    bool emit_ids = params.output_ids || vocab.empty();
    float* cmvn_shift = (float*)((paraformer::model&)m).g("cmvn.shift")->data;
    float* cmvn_scale = (float*)((paraformer::model&)m).g("cmvn.scale")->data;

    int T = 0;
    auto fb = paraformer::compute_fbank_raw(pcm_16k, T);
    if (T < 1) {
        result.error_msg = "fbank computation failed";
        return result;
    }

    auto ids = paraformer::paraformer_run_seg((paraformer::model&)m, fb, T, cmvn_shift, cmvn_scale, params.n_threads);
    if (emit_ids) {
        result.token_ids.insert(result.token_ids.end(), ids.begin(), ids.end());
    } else {
        result.text = paraformer::detok_pf(ids, vocab);
    }

    result.t_total_ms = now_ms() - t0;
    result.success = true;
    return result;
}

asr_result transcribe_paraformer(const std::string& model_gguf,
                                 const std::vector<float>& pcm_16k,
                                 const asr_params& params) {
    asr_result result;
    int64_t t0 = now_ms();

    paraformer::model m;
    std::vector<std::string> vocab;
    if (!paraformer::paraformer_load(model_gguf, m, vocab)) {
        result.error_msg = "failed to load Paraformer model: " + model_gguf;
        return result;
    }

    // Handle VAD segmentation if requested
    if (!params.vad_model_path.empty()) {
        std::vector<vad_segment> segments;
        if (!run_vad(params.vad_model_path, pcm_16k, segments, params.vad_maxseg)) {
            result.error_msg = "VAD failed";
            if (m.ctx_w) ggml_free(m.ctx_w);
            return result;
        }

        bool emit_ids = params.output_ids || vocab.empty();
        float* cmvn_shift = (float*)m.g("cmvn.shift")->data;
        float* cmvn_scale = (float*)m.g("cmvn.scale")->data;

        for (auto& s : segments) {
            int off = (int)((int64_t)s.start_ms * 16000 / 1000);
            int end = (int)((int64_t)s.end_ms * 16000 / 1000);
            if (end > (int)pcm_16k.size()) end = (int)pcm_16k.size();
            if (end - off < paraformer::WINLEN) continue;
            std::vector<float> seg(pcm_16k.begin() + off, pcm_16k.begin() + end);
            result.segments.push_back(s);
            int T = 0;
            auto fb = paraformer::compute_fbank_raw(seg, T);
            if (T > 0) {
                auto ids = paraformer::paraformer_run_seg(m, fb, T, cmvn_shift, cmvn_scale, params.n_threads);
                if (emit_ids) {
                    result.token_ids.insert(result.token_ids.end(), ids.begin(), ids.end());
                } else {
                    std::string txt = paraformer::detok_pf(ids, vocab);
                    if (!result.text.empty() && !txt.empty()) result.text += " ";
                    result.text += txt;
                }
            }
        }
    } else {
        result = transcribe_paraformer_with_model(m, vocab, pcm_16k, params);
    }

    if (m.ctx_w) ggml_free(m.ctx_w);
    result.t_total_ms = now_ms() - t0;
    return result;
}

} // namespace qwen3_tts
