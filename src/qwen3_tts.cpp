#include "qwen3_tts.h"

namespace qwen3_tts {

Qwen3TTS::Qwen3TTS() = default;

Qwen3TTS::~Qwen3TTS() = default;

void Qwen3TTS::set_progress_callback(tts_progress_callback_t callback) {
    progress_callback_ = callback;
}

asr_result Qwen3TTS::transcribe(const std::string& audio_path,
                                const std::string& model_gguf,
                                const asr_params& params) {
    std::vector<float> samples;
    int sample_rate = 0;
    if (!load_audio_file(audio_path, samples, sample_rate)) {
        asr_result r;
        r.error_msg = "failed to load audio: " + audio_path;
        return r;
    }

    auto pcm_16k = resample_to_16k(samples.data(), (int)samples.size(), sample_rate);

    // Detect model type from GGUF path or content
    bool is_paraformer = model_gguf.find("paraformer") != std::string::npos;
    if (is_paraformer) {
        return transcribe_paraformer(model_gguf, pcm_16k, params);
    }
    return transcribe_sensevoice(model_gguf, pcm_16k, params);
}

bool Qwen3TTS::detect_vad(const std::string& audio_path,
                           const std::string& vad_gguf,
                           std::vector<vad_segment>& segments,
                           int maxseg_ms) {
    vad_params vp;
    if (maxseg_ms > 0) vp.max_speech_duration_s = maxseg_ms / 1000.0f;
    return detect_vad(audio_path, vad_gguf, vp, segments);
}

bool Qwen3TTS::detect_vad(const std::string& audio_path,
                           const std::string& vad_gguf,
                           const vad_params& vp,
                           std::vector<vad_segment>& segments) {
    std::vector<float> samples;
    int sample_rate = 0;
    if (!load_audio_file(audio_path, samples, sample_rate)) {
        return false;
    }

    auto pcm_16k = resample_to_16k(samples.data(), (int)samples.size(), sample_rate);
    return run_vad(vad_gguf, pcm_16k, vp, segments);
}

} // namespace qwen3_tts
