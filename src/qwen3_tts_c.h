#ifndef QWEN3_TTS_C_H
#define QWEN3_TTS_C_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _WIN32
#  if defined(QWEN3_TTS_EXPORT) || defined(COMPILING_DLL)
#    define QWEN3_TTS_API __declspec(dllexport)
#  else
#    define QWEN3_TTS_API __declspec(dllimport)
#  endif
#else
#  define QWEN3_TTS_API __attribute__((visibility("default")))
#endif

typedef struct qwen3_tts_context qwen3_tts_context_t;

typedef struct {
    int32_t max_audio_tokens;
    float temperature;
    float top_p;
    int32_t top_k;
    int32_t n_threads;
    int32_t print_progress; // Use int32 instead of bool for ABI stability
    int32_t print_timing;   // Use int32
    float repetition_penalty;
    int32_t language_id;
    const char* instruction;
    const char* speaker;
} qwen3_tts_params_t;

typedef struct {
    float* audio;
    int32_t audio_len;
    int32_t sample_rate;
    int32_t success;        // Use int32
    char* error_msg;
    int64_t t_total_ms;
} qwen3_tts_result_t;

typedef struct {
    qwen3_tts_params_t generation;
    float chunk_sec;
    float left_context_sec;
    int32_t collect_audio;
} qwen3_tts_streaming_params_t;

typedef enum {
    QWEN3_TTS_MODEL_KIND_UNKNOWN = 0,
    QWEN3_TTS_MODEL_KIND_BASE = 1,
    QWEN3_TTS_MODEL_KIND_CUSTOM_VOICE = 2,
    QWEN3_TTS_MODEL_KIND_VOICE_DESIGN = 3,
} qwen3_tts_model_kind_t;

typedef struct {
    int32_t loaded;
    int32_t supports_voice_clone;
    int32_t supports_named_speakers;
    int32_t supports_instruction;
    int32_t speaker_embedding_dim;
    int32_t speaker_count;
    int32_t model_kind; // qwen3_tts_model_kind_t
} qwen3_tts_model_capabilities_t;

typedef void (*qwen3_tts_progress_callback)(int tokens_generated, int max_tokens, void* user_data);
typedef int32_t (*qwen3_tts_audio_chunk_callback)(
    const float* samples,
    int32_t n_samples,
    int32_t sample_rate,
    void* user_data
);

QWEN3_TTS_API qwen3_tts_context_t* qwen3_tts_init();
QWEN3_TTS_API void qwen3_tts_free(qwen3_tts_context_t* ctx);

QWEN3_TTS_API int32_t qwen3_tts_load_models(qwen3_tts_context_t* ctx, const char* model_dir);
QWEN3_TTS_API int32_t qwen3_tts_load_models_with_name(
    qwen3_tts_context_t* ctx,
    const char* model_dir,
    const char* model_name
);

QWEN3_TTS_API qwen3_tts_result_t qwen3_tts_synthesize(
    qwen3_tts_context_t* ctx, 
    const char* text, 
    qwen3_tts_params_t params
);

QWEN3_TTS_API qwen3_tts_result_t qwen3_tts_synthesize_with_voice(
    qwen3_tts_context_t* ctx,
    const char* text,
    const char* reference_audio,
    const char* reference_text,
    qwen3_tts_params_t params
);

QWEN3_TTS_API qwen3_tts_result_t qwen3_tts_synthesize_with_speaker_embedding(
    qwen3_tts_context_t* ctx,
    const char* text,
    const char* speaker_embedding_file,
    qwen3_tts_params_t params
);

QWEN3_TTS_API qwen3_tts_result_t qwen3_tts_synthesize_streaming(
    qwen3_tts_context_t* ctx,
    const char* text,
    qwen3_tts_streaming_params_t params,
    qwen3_tts_audio_chunk_callback callback,
    void* user_data
);

QWEN3_TTS_API qwen3_tts_result_t qwen3_tts_synthesize_with_voice_streaming(
    qwen3_tts_context_t* ctx,
    const char* text,
    const char* reference_audio,
    const char* reference_text,
    qwen3_tts_streaming_params_t params,
    qwen3_tts_audio_chunk_callback callback,
    void* user_data
);

QWEN3_TTS_API qwen3_tts_result_t qwen3_tts_synthesize_with_speaker_embedding_streaming(
    qwen3_tts_context_t* ctx,
    const char* text,
    const char* speaker_embedding_file,
    qwen3_tts_streaming_params_t params,
    qwen3_tts_audio_chunk_callback callback,
    void* user_data
);

// ===== Session API =====
// Opaque session handle for concurrent inference.
// Each session has its own scheduler, KV caches, and scratch buffers.
// Multiple sessions can share the same model weights safely from different threads.
typedef struct qwen3_tts_session qwen3_tts_session_t;

// Create a new inference session. The context must have models loaded.
// Returns NULL on failure.
QWEN3_TTS_API qwen3_tts_session_t* qwen3_tts_session_create(qwen3_tts_context_t* ctx);

// Free a session. Must not be used while synthesis is in progress.
QWEN3_TTS_API void qwen3_tts_session_free(qwen3_tts_session_t* session);

// Session-aware synthesis functions (parallel to existing non-session functions).
// Each session should be used from ONE thread at a time.
QWEN3_TTS_API qwen3_tts_result_t qwen3_tts_session_synthesize(
    qwen3_tts_context_t* ctx,
    qwen3_tts_session_t* session,
    const char* text,
    qwen3_tts_params_t params);

QWEN3_TTS_API qwen3_tts_result_t qwen3_tts_session_synthesize_with_voice(
    qwen3_tts_context_t* ctx,
    qwen3_tts_session_t* session,
    const char* text,
    const char* reference_audio,
    const char* reference_text,
    qwen3_tts_params_t params);

QWEN3_TTS_API qwen3_tts_result_t qwen3_tts_session_synthesize_streaming(
    qwen3_tts_context_t* ctx,
    qwen3_tts_session_t* session,
    const char* text,
    qwen3_tts_streaming_params_t params,
    qwen3_tts_audio_chunk_callback callback,
    void* user_data);

QWEN3_TTS_API qwen3_tts_result_t qwen3_tts_session_synthesize_with_voice_streaming(
    qwen3_tts_context_t* ctx,
    qwen3_tts_session_t* session,
    const char* text,
    const char* reference_audio,
    const char* reference_text,
    qwen3_tts_streaming_params_t params,
    qwen3_tts_audio_chunk_callback callback,
    void* user_data);

QWEN3_TTS_API int32_t qwen3_tts_extract_speaker_embedding(
    qwen3_tts_context_t* ctx,
    const char* reference_audio,
    const char* output_path
);

QWEN3_TTS_API qwen3_tts_model_capabilities_t qwen3_tts_get_model_capabilities(
    qwen3_tts_context_t* ctx
);

// Newline-separated speaker names (lowercase), or empty string if unavailable.
// Returned string is heap-allocated and must be released with qwen3_tts_free_string().
QWEN3_TTS_API char* qwen3_tts_get_available_speakers(qwen3_tts_context_t* ctx);
QWEN3_TTS_API void qwen3_tts_free_string(char* value);

QWEN3_TTS_API void qwen3_tts_free_result(qwen3_tts_result_t result);

QWEN3_TTS_API void qwen3_tts_set_progress_callback(
    qwen3_tts_context_t* ctx, 
    qwen3_tts_progress_callback callback, 
    void* user_data
);

// ===== ASR / VAD =====

typedef struct {
    int32_t start_ms;
    int32_t end_ms;
} qwen3_vad_segment_t;

// Silero VAD parameters. Mirrors whisper.cpp's whisper_vad_params (minus the
// samples_overlap field, which is not used by this project's time-direct
// segmentation). Values of 0 / <=0 mean "use the default" where noted.
typedef struct {
    float    threshold;                 // speech probability threshold (default 0.5)
    int32_t  min_speech_duration_ms;    // discard shorter segments (default 250)
    int32_t  min_silence_duration_ms;   // silence to end a segment (default 100)
    float    max_speech_duration_s;     // 0 = unlimited (default 30.0)
    int32_t  speech_pad_ms;             // pad each segment (default 30)
} qwen3_vad_params_t;

// Returns the default VAD parameters (matches whisper.cpp defaults, with
// max_speech_duration_s = 30.0 to preserve the project's legacy 30s cap).
QWEN3_TTS_API qwen3_vad_params_t qwen3_vad_default_params(void);

typedef struct {
    const char* vad_model_path;   // NULL = no VAD segmentation
    int32_t vad_maxseg;           // max segment duration in ms (default 30000)
    qwen3_vad_params_t vad_params; // full VAD params (used when vad_model_path is set)
    int32_t keep_tags;            // keep <|...|> meta tags in SenseVoice output
    int32_t output_ids;           // output token IDs instead of text
    int32_t n_threads;            // CPU threads (default 8)
} qwen3_asr_params_t;

typedef struct {
    char* text;                           // heap-allocated, free with qwen3_asr_free_result
    int32_t* token_ids;                   // heap-allocated array (NULL if output_ids=0)
    int32_t token_ids_len;
    qwen3_vad_segment_t* segments;        // heap-allocated array (NULL if no VAD)
    int32_t segments_len;
    int64_t t_total_ms;
    int32_t success;
    char* error_msg;                      // heap-allocated, free with qwen3_asr_free_result
} qwen3_asr_result_t;

// Transcribe an audio file (WAV, any sample rate).
// model_gguf: path to SenseVoice or Paraformer GGUF file.
// The model type is auto-detected from the file path (path contains "paraformer" -> Paraformer, else SenseVoice).
QWEN3_TTS_API qwen3_asr_result_t qwen3_asr_transcribe(
    qwen3_tts_context_t* ctx,
    const char* audio_path,
    const char* model_gguf,
    qwen3_asr_params_t params
);

// Detect speech segments using Silero VAD with default parameters.
// Returns segments via output params. Returns 1 on success, 0 on failure.
// Caller must free *out_segments with qwen3_vad_free_segments() when done.
QWEN3_TTS_API int32_t qwen3_vad_detect(
    qwen3_tts_context_t* ctx,
    const char* audio_path,
    const char* vad_gguf,
    int32_t maxseg_ms,
    qwen3_vad_segment_t** out_segments,
    int32_t* out_segments_len
);

// Like qwen3_vad_detect but with the full set of VAD parameters.
QWEN3_TTS_API int32_t qwen3_vad_detect_with_params(
    qwen3_tts_context_t* ctx,
    const char* audio_path,
    const char* vad_gguf,
    qwen3_vad_params_t params,
    qwen3_vad_segment_t** out_segments,
    int32_t* out_segments_len
);

QWEN3_TTS_API void qwen3_vad_free_segments(qwen3_vad_segment_t* segments);

QWEN3_TTS_API void qwen3_asr_free_result(qwen3_asr_result_t result);

// Pre-load and cache an ASR model (SenseVoice or Paraformer) to avoid reloading on every transcribe call.
// Model type is auto-detected from path (path contains "paraformer" -> Paraformer, else SenseVoice).
// Returns 1 on success, 0 on failure.
QWEN3_TTS_API int32_t qwen3_asr_load_model(qwen3_tts_context_t* ctx, const char* model_gguf);

// Free the cached ASR model.
QWEN3_TTS_API void qwen3_asr_free_model(qwen3_tts_context_t* ctx);

// Pre-load and cache a VAD model to avoid reloading on every detect call.
// Returns 1 on success, 0 on failure.
QWEN3_TTS_API int32_t qwen3_vad_load_model(qwen3_tts_context_t* ctx, const char* vad_gguf);

// Free the cached VAD model.
QWEN3_TTS_API void qwen3_vad_free_model(qwen3_tts_context_t* ctx);

// ===== Streaming VAD/ASR =====

// Opaque handle for streaming VAD state
typedef struct qwen3_vad_stream qwen3_vad_stream_t;

// Create a new streaming VAD context using the cached VAD model and default
// parameters. Returns NULL if no VAD model is cached.
QWEN3_TTS_API qwen3_vad_stream_t* qwen3_vad_stream_new(qwen3_tts_context_t* ctx, int32_t max_seg_ms);

// Like qwen3_vad_stream_new but with the full set of VAD parameters.
QWEN3_TTS_API qwen3_vad_stream_t* qwen3_vad_stream_new_with_params(
    qwen3_tts_context_t* ctx, qwen3_vad_params_t params);

// Feed PCM samples (f32, 16kHz mono) to the streaming VAD.
// Returns 1 on success, 0 on failure.
// New segments are available via qwen3_vad_stream_get_segments().
QWEN3_TTS_API int32_t qwen3_vad_stream_feed(
    qwen3_vad_stream_t* stream,
    const float* pcm,
    int32_t n_samples
);

// Get the number of completed segments.
QWEN3_TTS_API int32_t qwen3_vad_stream_get_segment_count(qwen3_vad_stream_t* stream);

// Get a completed segment by index. Returns 1 on success, 0 on failure.
QWEN3_TTS_API int32_t qwen3_vad_stream_get_segment(
    qwen3_vad_stream_t* stream,
    int32_t index,
    int32_t* out_start_ms,
    int32_t* out_end_ms
);

// Get the current open (in-progress) segment, if any.
// Returns 1 if there's an open segment, 0 if none.
QWEN3_TTS_API int32_t qwen3_vad_stream_get_open_segment(
    qwen3_vad_stream_t* stream,
    int32_t* out_start_ms,
    int32_t* out_end_ms
);

// Reset the streaming VAD state (clear all segments and accumulated state).
QWEN3_TTS_API void qwen3_vad_stream_reset(qwen3_vad_stream_t* stream);

// Free the streaming VAD context.
QWEN3_TTS_API void qwen3_vad_stream_free(qwen3_vad_stream_t* stream);

// Streaming ASR: transcribe PCM samples (f32, 16kHz mono) using cached ASR model.
// Returns transcription result. If model_gguf is NULL, uses cached model.
QWEN3_TTS_API qwen3_asr_result_t qwen3_asr_transcribe_pcm(
    qwen3_tts_context_t* ctx,
    const float* pcm,
    int32_t n_samples,
    const char* model_gguf,  // NULL = use cached model
    qwen3_asr_params_t params
);

#ifdef __cplusplus
}
#endif

#endif // QWEN3_TTS_C_H
