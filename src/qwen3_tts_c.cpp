#include "qwen3_tts_c.h"
#include "qwen3_tts.h"
#include "pipeline/pipeline_internal.h"
#include "transformer/transformer_state_internal.h"
#include "sensevoice_asr.h"
#include "paraformer_asr.h"
#include "funasr_vad.h"
#include <cstring>
#include <cstdlib>
#include <vector>
#include <string>
#include <memory>

#ifdef _WIN32
#define strdup _strdup
#endif

struct qwen3_tts_context {
    qwen3_tts::Qwen3TTS tts;
    qwen3_tts_progress_callback progress_callback = nullptr;
    void* user_data = nullptr;

    // Cached ASR models
    std::string cached_asr_path;
    std::unique_ptr<sensevoice::model> cached_sensevoice;
    std::vector<int> cached_sensevoice_qtok;
    std::vector<std::string> cached_sensevoice_vocab;

    std::unique_ptr<paraformer::model> cached_paraformer;
    std::vector<std::string> cached_paraformer_vocab;

    // Cached VAD model
    std::string cached_vad_path;
    std::unique_ptr<funasr_vad_impl::vad> cached_vad;
};

struct qwen3_tts_session {
    std::unique_ptr<qwen3_tts::Qwen3TTS::Session> session;
};

static int32_t to_model_kind(const std::string & model_type) {
    if (model_type == "base") return QWEN3_TTS_MODEL_KIND_BASE;
    if (model_type == "custom_voice") return QWEN3_TTS_MODEL_KIND_CUSTOM_VOICE;
    if (model_type == "voice_design") return QWEN3_TTS_MODEL_KIND_VOICE_DESIGN;
    return QWEN3_TTS_MODEL_KIND_UNKNOWN;
}

static qwen3_tts::tts_params convert_params(qwen3_tts_params_t params) {
    qwen3_tts::tts_params p;
    p.max_audio_tokens = params.max_audio_tokens;
    p.temperature = params.temperature;
    p.top_p = params.top_p;
    p.top_k = params.top_k;
    p.n_threads = params.n_threads;
    p.print_progress = params.print_progress != 0;
    p.print_timing = params.print_timing != 0;
    p.repetition_penalty = params.repetition_penalty;
    p.language_id = params.language_id;
    if (params.instruction) {
        p.instruction = params.instruction;
    }
    if (params.speaker) {
        p.speaker = params.speaker;
    }
    return p;
}

static qwen3_tts::tts_streaming_params convert_streaming_params(qwen3_tts_streaming_params_t params) {
    qwen3_tts::tts_streaming_params p;
    p.generation = convert_params(params.generation);
    p.chunk_sec = params.chunk_sec > 0.0f ? params.chunk_sec : p.chunk_sec;
    p.left_context_sec = params.left_context_sec >= 0.0f ? params.left_context_sec : p.left_context_sec;
    p.collect_audio = params.collect_audio != 0;
    return p;
}

static qwen3_tts_result_t convert_result(const qwen3_tts::tts_result& res) {
    qwen3_tts_result_t r;
    r.audio_len = static_cast<int32_t>(res.audio.size());
    if (r.audio_len > 0) {
        r.audio = (float*)malloc(r.audio_len * sizeof(float));
        std::memcpy(r.audio, res.audio.data(), r.audio_len * sizeof(float));
    } else {
        r.audio = nullptr;
    }
    r.sample_rate = res.sample_rate;
    r.success = res.success ? 1 : 0;
    if (!res.error_msg.empty()) {
        r.error_msg = strdup(res.error_msg.c_str());
    } else {
        r.error_msg = nullptr;
    }
    r.t_total_ms = res.t_total_ms;
    return r;
}

qwen3_tts_context_t* qwen3_tts_init() {
    return new qwen3_tts_context();
}

void qwen3_tts_free(qwen3_tts_context_t* ctx) {
    delete ctx;
}

int32_t qwen3_tts_load_models(qwen3_tts_context_t* ctx, const char* model_dir) {
    return qwen3_tts_load_models_with_name(ctx, model_dir, nullptr);
}

int32_t qwen3_tts_load_models_with_name(
    qwen3_tts_context_t* ctx,
    const char* model_dir,
    const char* model_name
) {
    if (!ctx || !model_dir) return 0;
    return ctx->tts.load_models(model_dir, model_name ? model_name : "") ? 1 : 0;
}

qwen3_tts_result_t qwen3_tts_synthesize(
    qwen3_tts_context_t* ctx, 
    const char* text, 
    qwen3_tts_params_t params
) {
    if (!ctx || !text) {
        qwen3_tts_result_t res = {0};
        res.success = 0;
        res.error_msg = strdup("Invalid context or text");
        return res;
    }
    auto result = ctx->tts.synthesize(text, convert_params(params));
    return convert_result(result);
}

qwen3_tts_result_t qwen3_tts_synthesize_with_voice(
    qwen3_tts_context_t* ctx,
    const char* text,
    const char* reference_audio,
    const char* reference_text,
    qwen3_tts_params_t params
) {
    if (!ctx || !text || !reference_audio) {
        qwen3_tts_result_t res = {0};
        res.success = 0;
        res.error_msg = strdup("Invalid context, text, or reference audio");
        return res;
    }
    auto p = convert_params(params);
    if (reference_text) {
        p.reference_text = reference_text;
    }
    auto result = ctx->tts.synthesize_with_voice(text, reference_audio, p);
    return convert_result(result);
}

qwen3_tts_result_t qwen3_tts_synthesize_with_speaker_embedding(
    qwen3_tts_context_t* ctx,
    const char* text,
    const char* speaker_embedding_file,
    qwen3_tts_params_t params
) {
    if (!ctx || !text || !speaker_embedding_file) {
        qwen3_tts_result_t res = {0};
        res.success = 0;
        res.error_msg = strdup("Invalid context, text, or speaker embedding file");
        return res;
    }

    std::vector<float> speaker_embedding;
    if (!qwen3_tts::load_speaker_embedding_file(speaker_embedding_file, speaker_embedding)) {
        qwen3_tts_result_t res = {0};
        res.success = 0;
        res.error_msg = strdup("Failed to load speaker embedding file");
        return res;
    }

    auto result = ctx->tts.synthesize_with_speaker_embedding(text, speaker_embedding, convert_params(params));
    return convert_result(result);
}

qwen3_tts_result_t qwen3_tts_synthesize_streaming(
    qwen3_tts_context_t* ctx,
    const char* text,
    qwen3_tts_streaming_params_t params,
    qwen3_tts_audio_chunk_callback callback,
    void* user_data
) {
    if (!ctx || !text || !callback) {
        qwen3_tts_result_t res = {0};
        res.success = 0;
        res.error_msg = strdup("Invalid context, text, or streaming callback");
        return res;
    }

    qwen3_tts::tts_audio_chunk_callback_t cb =
        [callback, user_data](const float* samples, int32_t n_samples, int32_t sample_rate) {
            return callback(samples, n_samples, sample_rate, user_data) != 0;
        };
    auto result = ctx->tts.synthesize_streaming(text, cb, convert_streaming_params(params));
    return convert_result(result);
}

qwen3_tts_result_t qwen3_tts_synthesize_with_voice_streaming(
    qwen3_tts_context_t* ctx,
    const char* text,
    const char* reference_audio,
    const char* reference_text,
    qwen3_tts_streaming_params_t params,
    qwen3_tts_audio_chunk_callback callback,
    void* user_data
) {
    if (!ctx || !text || !reference_audio || !callback) {
        qwen3_tts_result_t res = {0};
        res.success = 0;
        res.error_msg = strdup("Invalid context, text, reference audio, or streaming callback");
        return res;
    }

    qwen3_tts::tts_audio_chunk_callback_t cb =
        [callback, user_data](const float* samples, int32_t n_samples, int32_t sample_rate) {
            return callback(samples, n_samples, sample_rate, user_data) != 0;
        };
    auto sp = convert_streaming_params(params);
    if (reference_text) {
        sp.generation.reference_text = reference_text;
    }
    auto result = ctx->tts.synthesize_with_voice_streaming(text, reference_audio, cb, sp);
    return convert_result(result);
}

qwen3_tts_result_t qwen3_tts_synthesize_with_speaker_embedding_streaming(
    qwen3_tts_context_t* ctx,
    const char* text,
    const char* speaker_embedding_file,
    qwen3_tts_streaming_params_t params,
    qwen3_tts_audio_chunk_callback callback,
    void* user_data
) {
    if (!ctx || !text || !speaker_embedding_file || !callback) {
        qwen3_tts_result_t res = {0};
        res.success = 0;
        res.error_msg = strdup("Invalid context, text, speaker embedding file, or streaming callback");
        return res;
    }

    std::vector<float> speaker_embedding;
    if (!qwen3_tts::load_speaker_embedding_file(speaker_embedding_file, speaker_embedding)) {
        qwen3_tts_result_t res = {0};
        res.success = 0;
        res.error_msg = strdup("Failed to load speaker embedding file");
        return res;
    }

    qwen3_tts::tts_audio_chunk_callback_t cb =
        [callback, user_data](const float* samples, int32_t n_samples, int32_t sample_rate) {
            return callback(samples, n_samples, sample_rate, user_data) != 0;
        };
    auto result = ctx->tts.synthesize_with_speaker_embedding_streaming(
        text, speaker_embedding, cb, convert_streaming_params(params));
    return convert_result(result);
}

int32_t qwen3_tts_extract_speaker_embedding(
    qwen3_tts_context_t* ctx,
    const char* reference_audio,
    const char* output_path
) {
    if (!ctx || !reference_audio || !output_path) return 0;

    std::vector<float> speaker_embedding;
    if (!ctx->tts.extract_speaker_embedding(reference_audio, speaker_embedding, nullptr)) {
        return 0;
    }

    return qwen3_tts::save_speaker_embedding_file(output_path, speaker_embedding) ? 1 : 0;
}

qwen3_tts_model_capabilities_t qwen3_tts_get_model_capabilities(qwen3_tts_context_t* ctx) {
    qwen3_tts_model_capabilities_t out = {0};
    out.model_kind = QWEN3_TTS_MODEL_KIND_UNKNOWN;
    if (!ctx) {
        return out;
    }

    const qwen3_tts::tts_model_capabilities caps = ctx->tts.get_model_capabilities();
    out.loaded = caps.loaded ? 1 : 0;
    out.supports_voice_clone = caps.supports_voice_clone ? 1 : 0;
    out.supports_named_speakers = caps.supports_named_speakers ? 1 : 0;
    out.supports_instruction = caps.supports_instruction ? 1 : 0;
    out.speaker_embedding_dim = caps.speaker_embedding_dim;
    out.speaker_count = caps.speaker_count;
    out.model_kind = to_model_kind(caps.model_type);
    return out;
}

char* qwen3_tts_get_available_speakers(qwen3_tts_context_t* ctx) {
    if (!ctx) {
        return strdup("");
    }

    const std::vector<std::string> speakers = ctx->tts.get_available_speakers();
    std::string joined;
    for (size_t i = 0; i < speakers.size(); ++i) {
        if (i != 0) {
            joined.push_back('\n');
        }
        joined += speakers[i];
    }

    return strdup(joined.c_str());
}

void qwen3_tts_free_string(char* value) {
    if (value) {
        free(value);
    }
}

void qwen3_tts_free_result(qwen3_tts_result_t result) {
    if (result.audio) free(result.audio);
    if (result.error_msg) free(result.error_msg);
}

void qwen3_tts_set_progress_callback(
    qwen3_tts_context_t* ctx, 
    qwen3_tts_progress_callback callback, 
    void* user_data
) {
    if (!ctx) return;
    ctx->progress_callback = callback;
    ctx->user_data = user_data;
    
    if (callback) {
        ctx->tts.set_progress_callback([ctx](int tokens, int max) {
            ctx->progress_callback(tokens, max, ctx->user_data);
        });
    } else {
        ctx->tts.set_progress_callback(nullptr);
    }
}

// ===== ASR / VAD =====

static qwen3_asr_result_t make_asr_error(const char* msg) {
    qwen3_asr_result_t r;
    std::memset(&r, 0, sizeof(r));
    r.success = 0;
    r.error_msg = strdup(msg);
    return r;
}

static qwen3_asr_result_t convert_asr_result(const qwen3_tts::asr_result& res) {
    qwen3_asr_result_t r;
    std::memset(&r, 0, sizeof(r));

    r.success = res.success ? 1 : 0;
    r.t_total_ms = res.t_total_ms;

    if (!res.text.empty()) {
        r.text = strdup(res.text.c_str());
    }

    if (!res.token_ids.empty()) {
        r.token_ids_len = static_cast<int32_t>(res.token_ids.size());
        r.token_ids = (int32_t*)malloc(r.token_ids_len * sizeof(int32_t));
        std::memcpy(r.token_ids, res.token_ids.data(), r.token_ids_len * sizeof(int32_t));
    }

    if (!res.segments.empty()) {
        r.segments_len = static_cast<int32_t>(res.segments.size());
        r.segments = (qwen3_vad_segment_t*)malloc(r.segments_len * sizeof(qwen3_vad_segment_t));
        for (int32_t i = 0; i < r.segments_len; i++) {
            r.segments[i].start_ms = res.segments[i].start_ms;
            r.segments[i].end_ms = res.segments[i].end_ms;
        }
    }

    if (!res.error_msg.empty()) {
        r.error_msg = strdup(res.error_msg.c_str());
    }

    return r;
}

qwen3_asr_result_t qwen3_asr_transcribe(
    qwen3_tts_context_t* ctx,
    const char* audio_path,
    const char* model_gguf,
    qwen3_asr_params_t params
) {
    if (!ctx) return make_asr_error("Invalid context");
    if (!audio_path) return make_asr_error("Audio path is required");

    // If model_gguf is NULL, use cached model
    if (!model_gguf && ctx->cached_asr_path.empty()) {
        return make_asr_error("ASR model path is required (or call qwen3_asr_load_model first)");
    }

    qwen3_tts::asr_params p;
    if (params.vad_model_path) p.vad_model_path = params.vad_model_path;
    p.vad_maxseg = params.vad_maxseg > 0 ? params.vad_maxseg : 30000;
    p.keep_tags = params.keep_tags != 0;
    p.output_ids = params.output_ids != 0;
    p.n_threads = params.n_threads > 0 ? params.n_threads : 8;

    // Load audio
    std::vector<float> pcm;
    int sample_rate = 0;
    if (!qwen3_tts::load_audio_file(audio_path, pcm, sample_rate)) {
        return make_asr_error("Failed to load audio file");
    }

    // Resample to 16kHz
    auto pcm_16k = qwen3_tts::resample_to_16k(pcm.data(), (int)pcm.size(), sample_rate);

    // Check if model is cached (either model_gguf matches cached path, or model_gguf is NULL)
    bool use_cached = model_gguf ? (ctx->cached_asr_path == model_gguf) : !ctx->cached_asr_path.empty();

    qwen3_tts::asr_result result;

    if (use_cached && ctx->cached_sensevoice) {
        // Use cached SenseVoice model
        result = qwen3_tts::transcribe_sensevoice_with_model(
            *ctx->cached_sensevoice,
            ctx->cached_sensevoice_qtok,
            ctx->cached_sensevoice_vocab,
            pcm_16k,
            p
        );
    } else if (use_cached && ctx->cached_paraformer) {
        // Use cached Paraformer model
        result = qwen3_tts::transcribe_paraformer_with_model(
            *ctx->cached_paraformer,
            ctx->cached_paraformer_vocab,
            pcm_16k,
            p
        );
    } else {
        // Load model from file (not cached)
        result = ctx->tts.transcribe(audio_path, model_gguf, p);
    }

    // Handle VAD segmentation if needed and using cached model
    if (use_cached && !p.vad_model_path.empty() && result.success) {
        // Re-run with VAD if the cached path doesn't include VAD handling
        // Actually, the _with_model functions don't handle VAD, so we need to do it here
        std::vector<qwen3_tts::vad_segment> segments;
        if (qwen3_tts::run_vad(p.vad_model_path, pcm_16k, segments, p.vad_maxseg)) {
            result.segments = segments;
            // Re-run transcription for each segment
            result.text.clear();
            result.token_ids.clear();

            if (ctx->cached_sensevoice) {
                auto mel_fb = sensevoice::mel_filterbank();
                float* emb = (float*)ctx->cached_sensevoice->g("embed.weight")->data;
                int nq = (int)ctx->cached_sensevoice_qtok.size();
                bool emit_ids = p.output_ids || ctx->cached_sensevoice_vocab.empty();

                for (auto& s : segments) {
                    int off = (int)((int64_t)s.start_ms * 16000 / 1000);
                    int end = (int)((int64_t)s.end_ms * 16000 / 1000);
                    if (end > (int)pcm_16k.size()) end = (int)pcm_16k.size();
                    if (end - off < sensevoice::WINLEN) continue;
                    std::vector<float> seg(pcm_16k.begin() + off, pcm_16k.begin() + end);
                    int t = 0;
                    auto fb = sensevoice::compute_fbank(seg, t, &mel_fb);
                    if (t > 0) {
                        auto ids = sensevoice::run_seg(*ctx->cached_sensevoice, ctx->cached_sensevoice_qtok.data(), nq, emb, fb, t, p.n_threads);
                        if (emit_ids) {
                            result.token_ids.insert(result.token_ids.end(), ids.begin(), ids.end());
                        } else {
                            std::string txt = sensevoice::detok_sv(ids, ctx->cached_sensevoice_vocab, p.keep_tags);
                            if (!result.text.empty() && !txt.empty()) result.text += " ";
                            result.text += txt;
                        }
                    }
                }
            } else if (ctx->cached_paraformer) {
                float* cmvn_shift = (float*)ctx->cached_paraformer->g("cmvn.shift")->data;
                float* cmvn_scale = (float*)ctx->cached_paraformer->g("cmvn.scale")->data;
                bool emit_ids = p.output_ids || ctx->cached_paraformer_vocab.empty();

                for (auto& s : segments) {
                    int off = (int)((int64_t)s.start_ms * 16000 / 1000);
                    int end = (int)((int64_t)s.end_ms * 16000 / 1000);
                    if (end > (int)pcm_16k.size()) end = (int)pcm_16k.size();
                    if (end - off < paraformer::WINLEN) continue;
                    std::vector<float> seg(pcm_16k.begin() + off, pcm_16k.begin() + end);
                    int T = 0;
                    auto fb = paraformer::compute_fbank_raw(seg, T);
                    if (T > 0) {
                        auto ids = paraformer::paraformer_run_seg(*ctx->cached_paraformer, fb, T, cmvn_shift, cmvn_scale, p.n_threads);
                        if (emit_ids) {
                            result.token_ids.insert(result.token_ids.end(), ids.begin(), ids.end());
                        } else {
                            std::string txt = paraformer::detok_pf(ids, ctx->cached_paraformer_vocab);
                            if (!result.text.empty() && !txt.empty()) result.text += " ";
                            result.text += txt;
                        }
                    }
                }
            }
        } else {
            result.error_msg = "VAD failed";
            result.success = false;
        }
    }

    return convert_asr_result(result);
}

int32_t qwen3_vad_detect(
    qwen3_tts_context_t* ctx,
    const char* audio_path,
    const char* vad_gguf,
    int32_t maxseg_ms,
    qwen3_vad_segment_t** out_segments,
    int32_t* out_segments_len
) {
    if (!ctx || !audio_path || !vad_gguf || !out_segments || !out_segments_len) return 0;

    *out_segments = nullptr;
    *out_segments_len = 0;

    std::vector<qwen3_tts::vad_segment> segments;
    if (!ctx->tts.detect_vad(audio_path, vad_gguf, segments, maxseg_ms > 0 ? maxseg_ms : 30000)) {
        return 0;
    }

    if (!segments.empty()) {
        *out_segments_len = static_cast<int32_t>(segments.size());
        *out_segments = (qwen3_vad_segment_t*)malloc(*out_segments_len * sizeof(qwen3_vad_segment_t));
        for (int32_t i = 0; i < *out_segments_len; i++) {
            (*out_segments)[i].start_ms = segments[i].start_ms;
            (*out_segments)[i].end_ms = segments[i].end_ms;
        }
    }

    return 1;
}

void qwen3_vad_free_segments(qwen3_vad_segment_t* segments) {
    if (segments) free(segments);
}

void qwen3_asr_free_result(qwen3_asr_result_t result) {
    if (result.text) free(result.text);
    if (result.token_ids) free(result.token_ids);
    if (result.segments) free(result.segments);
    if (result.error_msg) free(result.error_msg);
}

// ===== Model caching =====

static bool is_paraformer_path(const char* path) {
    if (!path) return false;
    std::string p(path);
    std::transform(p.begin(), p.end(), p.begin(), ::tolower);
    return p.find("paraformer") != std::string::npos;
}

int32_t qwen3_asr_load_model(qwen3_tts_context_t* ctx, const char* model_gguf) {
    if (!ctx || !model_gguf) return 0;

    // Already cached with same path?
    if (ctx->cached_asr_path == model_gguf) return 1;

    // Clear previous cache
    ctx->cached_sensevoice.reset();
    ctx->cached_paraformer.reset();
    ctx->cached_sensevoice_qtok.clear();
    ctx->cached_sensevoice_vocab.clear();
    ctx->cached_paraformer_vocab.clear();
    ctx->cached_asr_path.clear();

    if (is_paraformer_path(model_gguf)) {
        auto m = std::make_unique<paraformer::model>();
        if (!paraformer::paraformer_load(model_gguf, *m, ctx->cached_paraformer_vocab)) {
            return 0;
        }
        ctx->cached_paraformer = std::move(m);
    } else {
        auto m = std::make_unique<sensevoice::model>();
        if (!sensevoice::sensevoice_load(model_gguf, *m, ctx->cached_sensevoice_qtok, ctx->cached_sensevoice_vocab)) {
            return 0;
        }
        ctx->cached_sensevoice = std::move(m);
    }

    ctx->cached_asr_path = model_gguf;
    return 1;
}

void qwen3_asr_free_model(qwen3_tts_context_t* ctx) {
    if (!ctx) return;
    ctx->cached_sensevoice.reset();
    ctx->cached_paraformer.reset();
    ctx->cached_sensevoice_qtok.clear();
    ctx->cached_sensevoice_vocab.clear();
    ctx->cached_paraformer_vocab.clear();
    ctx->cached_asr_path.clear();
}

int32_t qwen3_vad_load_model(qwen3_tts_context_t* ctx, const char* vad_gguf) {
    if (!ctx || !vad_gguf) return 0;

    // Already cached with same path?
    if (ctx->cached_vad_path == vad_gguf) return 1;

    // Clear previous cache
    if (ctx->cached_vad && ctx->cached_vad->ctx) {
        ggml_free(ctx->cached_vad->ctx);
    }
    ctx->cached_vad.reset();
    ctx->cached_vad_path.clear();

    auto m = std::make_unique<funasr_vad_impl::vad>();
    gguf_init_params ip = {false, &m->ctx};
    gguf_context* gg = gguf_init_from_file(vad_gguf, ip);
    if (!gg) return 0;

    for (int i = 0; i < gguf_get_n_tensors(gg); i++) {
        const char* nm = gguf_get_tensor_name(gg, i);
        m->t[nm] = ggml_get_tensor(m->ctx, nm);
    }
    gguf_free(gg);

    ctx->cached_vad = std::move(m);
    ctx->cached_vad_path = vad_gguf;
    return 1;
}

void qwen3_vad_free_model(qwen3_tts_context_t* ctx) {
    if (!ctx) return;
    if (ctx->cached_vad && ctx->cached_vad->ctx) {
        ggml_free(ctx->cached_vad->ctx);
    }
    ctx->cached_vad.reset();
    ctx->cached_vad_path.clear();
}

// ===== Streaming VAD =====

struct qwen3_vad_stream {
    funasr_vad_impl::vad* vad_model = nullptr;  // borrowed from context, not owned
    vad_inc_state inc_state;  // incremental VAD state
    bool initialized = false;
};

qwen3_vad_stream_t* qwen3_vad_stream_new(qwen3_tts_context_t* ctx, int32_t max_seg_ms) {
    if (!ctx || !ctx->cached_vad) return nullptr;

    auto stream = new qwen3_vad_stream();
    stream->vad_model = ctx->cached_vad.get();
    if (!vad_inc_init(&stream->inc_state, *stream->vad_model, max_seg_ms > 0 ? max_seg_ms : 30000)) {
        delete stream;
        return nullptr;
    }
    stream->initialized = true;
    return stream;
}

int32_t qwen3_vad_stream_feed(qwen3_vad_stream_t* stream, const float* pcm, int32_t n_samples) {
    if (!stream || !stream->initialized || !pcm || n_samples <= 0) return 0;
    int frames = vad_inc_feed(&stream->inc_state, *stream->vad_model, pcm, n_samples);
    return frames >= 0 ? 1 : 0;
}

int32_t qwen3_vad_stream_get_segment_count(qwen3_vad_stream_t* stream) {
    if (!stream || !stream->initialized) return 0;
    return (int32_t)stream->inc_state.completed_segments.size();
}

int32_t qwen3_vad_stream_get_segment(
    qwen3_vad_stream_t* stream,
    int32_t index,
    int32_t* out_start_ms,
    int32_t* out_end_ms
) {
    if (!stream || !stream->initialized || index < 0 || index >= (int32_t)stream->inc_state.completed_segments.size()) return 0;
    *out_start_ms = stream->inc_state.completed_segments[index].first;
    *out_end_ms = stream->inc_state.completed_segments[index].second;
    return 1;
}

int32_t qwen3_vad_stream_get_open_segment(
    qwen3_vad_stream_t* stream,
    int32_t* out_start_ms,
    int32_t* out_end_ms
) {
    if (!stream || !stream->initialized) return 0;
    int start_ms, end_ms;
    if (!vad_inc_get_open_segment(&stream->inc_state, start_ms, end_ms)) return 0;
    *out_start_ms = start_ms;
    *out_end_ms = end_ms;
    return 1;
}

void qwen3_vad_stream_reset(qwen3_vad_stream_t* stream) {
    if (!stream || !stream->initialized) return;
    vad_inc_reset(&stream->inc_state);
}

void qwen3_vad_stream_free(qwen3_vad_stream_t* stream) {
    if (!stream) return;
    if (stream->initialized) {
        vad_inc_free(&stream->inc_state);
    }
    delete stream;
}

// ===== Streaming ASR =====

qwen3_asr_result_t qwen3_asr_transcribe_pcm(
    qwen3_tts_context_t* ctx,
    const float* pcm,
    int32_t n_samples,
    const char* model_gguf,
    qwen3_asr_params_t params
) {
    if (!ctx) return make_asr_error("Invalid context");
    if (!pcm || n_samples <= 0) return make_asr_error("Invalid PCM data");

    // If model_gguf is NULL, use cached model
    if (!model_gguf && ctx->cached_asr_path.empty()) {
        return make_asr_error("ASR model path is required (or call qwen3_asr_load_model first)");
    }

    qwen3_tts::asr_params p;
    if (params.vad_model_path) p.vad_model_path = params.vad_model_path;
    p.vad_maxseg = params.vad_maxseg > 0 ? params.vad_maxseg : 30000;
    p.keep_tags = params.keep_tags != 0;
    p.output_ids = params.output_ids != 0;
    p.n_threads = params.n_threads > 0 ? params.n_threads : 8;

    // Copy PCM to vector
    std::vector<float> pcm_16k(pcm, pcm + n_samples);

    // Check if model is cached
    bool use_cached = model_gguf ? (ctx->cached_asr_path == model_gguf) : !ctx->cached_asr_path.empty();

    qwen3_tts::asr_result result;

    if (use_cached && ctx->cached_sensevoice) {
        result = qwen3_tts::transcribe_sensevoice_with_model(
            *ctx->cached_sensevoice,
            ctx->cached_sensevoice_qtok,
            ctx->cached_sensevoice_vocab,
            pcm_16k,
            p
        );
    } else if (use_cached && ctx->cached_paraformer) {
        result = qwen3_tts::transcribe_paraformer_with_model(
            *ctx->cached_paraformer,
            ctx->cached_paraformer_vocab,
            pcm_16k,
            p
        );
    } else {
        // Cannot load from file when given raw PCM - need cached model
        return make_asr_error("Raw PCM transcription requires a cached ASR model");
    }

    return convert_asr_result(result);
}

// ===== Session API =====

qwen3_tts_session_t* qwen3_tts_session_create(qwen3_tts_context_t* ctx) {
    if (!ctx) return nullptr;
    auto s = new (std::nothrow) qwen3_tts_session();
    if (!s) return nullptr;
    s->session = ctx->tts.create_session();
    if (!s->session) {
        delete s;
        return nullptr;
    }
    return s;
}

void qwen3_tts_session_free(qwen3_tts_session_t* session) {
    delete session;
}

qwen3_tts_result_t qwen3_tts_session_synthesize(
    qwen3_tts_context_t* ctx,
    qwen3_tts_session_t* session,
    const char* text,
    qwen3_tts_params_t params
) {
    if (!ctx || !session || !session->session || !text) {
        qwen3_tts_result_t res;
        std::memset(&res, 0, sizeof(res));
        res.error_msg = strdup("Invalid context, session, or text");
        return res;
    }
    auto result = ctx->tts.synthesize(*session->session, text, convert_params(params));
    return convert_result(result);
}

qwen3_tts_result_t qwen3_tts_session_synthesize_with_voice(
    qwen3_tts_context_t* ctx,
    qwen3_tts_session_t* session,
    const char* text,
    const char* reference_audio,
    const char* reference_text,
    qwen3_tts_params_t params
) {
    if (!ctx || !session || !session->session || !text || !reference_audio) {
        qwen3_tts_result_t res;
        std::memset(&res, 0, sizeof(res));
        res.error_msg = strdup("Invalid context, session, text, or reference audio");
        return res;
    }
    std::vector<float> ref_samples;
    int ref_sample_rate;
    if (!qwen3_tts::load_audio_file(reference_audio, ref_samples, ref_sample_rate)) {
        qwen3_tts_result_t res;
        std::memset(&res, 0, sizeof(res));
        res.error_msg = strdup("Failed to load reference audio");
        return res;
    }
    const int target_rate = 24000;
    if (ref_sample_rate != target_rate) {
        std::vector<float> resampled;
        qwen3_tts::pipeline_internal::resample_linear(ref_samples.data(), (int)ref_samples.size(), ref_sample_rate, resampled, target_rate);
        ref_samples = std::move(resampled);
    }
    auto p = convert_params(params);
    if (reference_text) {
        p.reference_text = reference_text;
    }
    auto result = ctx->tts.synthesize_with_voice(*session->session, text, ref_samples.data(), (int32_t)ref_samples.size(), p);
    return convert_result(result);
}

qwen3_tts_result_t qwen3_tts_session_synthesize_streaming(
    qwen3_tts_context_t* ctx,
    qwen3_tts_session_t* session,
    const char* text,
    qwen3_tts_streaming_params_t params,
    qwen3_tts_audio_chunk_callback callback,
    void* user_data
) {
    if (!ctx || !session || !session->session || !text || !callback) {
        qwen3_tts_result_t res;
        std::memset(&res, 0, sizeof(res));
        res.error_msg = strdup("Invalid context, session, text, or streaming callback");
        return res;
    }
    qwen3_tts::tts_audio_chunk_callback_t cb =
        [callback, user_data](const float* samples, int32_t n_samples, int32_t sample_rate) {
            return callback(samples, n_samples, sample_rate, user_data) != 0;
        };
    auto result = ctx->tts.synthesize_streaming(*session->session, text, cb, convert_streaming_params(params));
    return convert_result(result);
}

qwen3_tts_result_t qwen3_tts_session_synthesize_with_voice_streaming(
    qwen3_tts_context_t* ctx,
    qwen3_tts_session_t* session,
    const char* text,
    const char* reference_audio,
    const char* reference_text,
    qwen3_tts_streaming_params_t params,
    qwen3_tts_audio_chunk_callback callback,
    void* user_data
) {
    if (!ctx || !session || !session->session || !text || !reference_audio || !callback) {
        qwen3_tts_result_t res;
        std::memset(&res, 0, sizeof(res));
        res.error_msg = strdup("Invalid context, session, text, reference audio, or callback");
        return res;
    }
    qwen3_tts::tts_audio_chunk_callback_t cb =
        [callback, user_data](const float* samples, int32_t n_samples, int32_t sample_rate) {
            return callback(samples, n_samples, sample_rate, user_data) != 0;
        };
    auto sp = convert_streaming_params(params);
    if (reference_text) {
        sp.generation.reference_text = reference_text;
    }
    auto result = ctx->tts.synthesize_with_voice_streaming(*session->session, text, reference_audio, cb, sp);
    return convert_result(result);
}

