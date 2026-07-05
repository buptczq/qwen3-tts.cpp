#include "transformer/transformer_state_internal.h"

namespace qwen3_tts {

TTSTransformerSession::TTSTransformerSession() = default;

TTSTransformerSession::~TTSTransformerSession() {
    if (state_.sched) {
        ggml_backend_sched_free(state_.sched);
        state_.sched = nullptr;
    }
    free_tts_kv_cache(state_.cache);
    free_tts_kv_cache(state_.code_pred_cache);
    if (own_backend_ && state_.backend) {
        ggml_backend_free(state_.backend);
        state_.backend = nullptr;
    }
}

} // namespace qwen3_tts
