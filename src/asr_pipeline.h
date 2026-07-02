// asr_pipeline.h — Unified ASR/VAD API for qwen3-tts.cpp.
// Wraps FSMN-VAD, SenseVoice, and Paraformer from the funasr ggml runtime.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

// Forward declarations for model types (defined in single-header libraries)
namespace sensevoice { struct model; }
namespace paraformer { struct model; }

namespace qwen3_tts {

struct vad_segment {
    int start_ms = 0;
    int end_ms = 0;
};

struct asr_result {
    std::string text;
    std::vector<int> token_ids;
    std::vector<vad_segment> segments;
    int64_t t_total_ms = 0;
    bool success = false;
    std::string error_msg;
};

struct asr_params {
    std::string vad_model_path;
    int vad_maxseg = 30000;
    bool keep_tags = false;
    bool output_ids = false;
    int n_threads = 8;
};

// Resample audio from source_rate to 16kHz mono (linear interpolation).
std::vector<float> resample_to_16k(const float* samples, int n_samples, int source_rate);

// VAD-only: detect speech segments in 16kHz mono PCM.
bool run_vad(const std::string& vad_gguf,
             const std::vector<float>& pcm_16k,
             std::vector<vad_segment>& segments,
             int maxseg_ms = 30000);

// Transcribe with SenseVoice (16kHz mono PCM input).
asr_result transcribe_sensevoice(const std::string& model_gguf,
                                 const std::vector<float>& pcm_16k,
                                 const asr_params& params);

// Transcribe with SenseVoice using pre-loaded model (avoids reloading from disk).
asr_result transcribe_sensevoice_with_model(const sensevoice::model& model,
                                            const std::vector<int>& qtok,
                                            const std::vector<std::string>& vocab,
                                            const std::vector<float>& pcm_16k,
                                            const asr_params& params);

// Transcribe with Paraformer (16kHz mono PCM input).
asr_result transcribe_paraformer(const std::string& model_gguf,
                                 const std::vector<float>& pcm_16k,
                                 const asr_params& params);

// Transcribe with Paraformer using pre-loaded model (avoids reloading from disk).
asr_result transcribe_paraformer_with_model(const paraformer::model& model,
                                            const std::vector<std::string>& vocab,
                                            const std::vector<float>& pcm_16k,
                                            const asr_params& params);

} // namespace qwen3_tts
