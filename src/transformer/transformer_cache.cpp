#include "tts_transformer.h"
#include "transformer/transformer_state_internal.h"
#include "transformer/transformer_internal.h"

#include <algorithm>
#include <cstdio>
#include <string>

namespace qwen3_tts {
namespace {

void reset_scheduler_reserve_state(tts_transformer_state & state) {
    state.sched_reserved = false;
    state.sched_reserve_failed = false;
    state.sched_reserved_ctx = 0;
    state.sched_reserved_prefill_len = 0;
}

} // namespace

// ---------------------------------------------------------------------------
// Session-aware primary implementations
// ---------------------------------------------------------------------------

bool TTSTransformer::init_kv_cache(TTSTransformerSession & session, int32_t n_ctx) {
    const auto & cfg = impl_->model.config;

    free_tts_kv_cache(session.state_.cache);

    session.state_.cache.n_ctx = n_ctx;
    session.state_.cache.n_used = 0;
    session.state_.cache.head_dim = cfg.head_dim;
    session.state_.cache.n_kv_heads = cfg.n_key_value_heads;
    session.state_.cache.n_layers = cfg.n_layers;
    reset_scheduler_reserve_state(session.state_);

    const size_t n_tensors = cfg.n_layers * 2;
    const size_t ctx_size = n_tensors * ggml_tensor_overhead();

    struct ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };

    session.state_.cache.ctx = ggml_init(params);
    if (!session.state_.cache.ctx) {
        error_msg_ = "Failed to create KV cache context";
        return false;
    }

    session.state_.cache.k_cache.resize(cfg.n_layers);
    session.state_.cache.v_cache.resize(cfg.n_layers);

    for (int il = 0; il < cfg.n_layers; ++il) {
        session.state_.cache.k_cache[il] = ggml_new_tensor_3d(
            session.state_.cache.ctx, GGML_TYPE_F16,
            cfg.head_dim, cfg.n_key_value_heads, n_ctx);
        ggml_format_name(session.state_.cache.k_cache[il], "k_cache_%d", il);

        session.state_.cache.v_cache[il] = ggml_new_tensor_3d(
            session.state_.cache.ctx, GGML_TYPE_F16,
            cfg.head_dim, cfg.n_key_value_heads, n_ctx);
        ggml_format_name(session.state_.cache.v_cache[il], "v_cache_%d", il);
    }

    session.state_.cache.buffer = ggml_backend_alloc_ctx_tensors(session.state_.cache.ctx, session.state_.backend);
    if (!session.state_.cache.buffer) {
        error_msg_ = "Failed to allocate KV cache buffer";
        return false;
    }

    return true;
}

void TTSTransformer::clear_kv_cache(TTSTransformerSession & session) {
    session.state_.cache.n_used = 0;
}

bool TTSTransformer::init_code_pred_kv_cache(TTSTransformerSession & session, int32_t n_ctx) {
    const auto & cfg = impl_->model.config;

    free_tts_kv_cache(session.state_.code_pred_cache);

    session.state_.code_pred_cache.n_ctx = n_ctx;
    session.state_.code_pred_cache.n_used = 0;
    session.state_.code_pred_cache.head_dim = cfg.code_pred_head_dim;
    session.state_.code_pred_cache.n_kv_heads = cfg.code_pred_n_key_value_heads;
    session.state_.code_pred_cache.n_layers = cfg.code_pred_layers;
    reset_scheduler_reserve_state(session.state_);

    const size_t n_tensors = cfg.code_pred_layers * 2;
    const size_t ctx_size = n_tensors * ggml_tensor_overhead();

    struct ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };

    session.state_.code_pred_cache.ctx = ggml_init(params);
    if (!session.state_.code_pred_cache.ctx) {
        error_msg_ = "Failed to create code predictor KV cache context";
        return false;
    }

    session.state_.code_pred_cache.k_cache.resize(cfg.code_pred_layers);
    session.state_.code_pred_cache.v_cache.resize(cfg.code_pred_layers);

    for (int il = 0; il < cfg.code_pred_layers; ++il) {
        session.state_.code_pred_cache.k_cache[il] = ggml_new_tensor_3d(
            session.state_.code_pred_cache.ctx, GGML_TYPE_F16,
            cfg.code_pred_head_dim, n_ctx, cfg.code_pred_n_key_value_heads);
        ggml_format_name(session.state_.code_pred_cache.k_cache[il], "code_pred_k_cache_%d", il);

        session.state_.code_pred_cache.v_cache[il] = ggml_new_tensor_3d(
            session.state_.code_pred_cache.ctx, GGML_TYPE_F16,
            cfg.code_pred_head_dim, n_ctx, cfg.code_pred_n_key_value_heads);
        ggml_format_name(session.state_.code_pred_cache.v_cache[il], "code_pred_v_cache_%d", il);
    }

    session.state_.code_pred_cache.buffer = ggml_backend_alloc_ctx_tensors(session.state_.code_pred_cache.ctx, session.state_.backend);
    if (!session.state_.code_pred_cache.buffer) {
        error_msg_ = "Failed to allocate code predictor KV cache buffer";
        return false;
    }

    return true;
}

void TTSTransformer::clear_code_pred_kv_cache(TTSTransformerSession & session) {
    session.state_.code_pred_cache.n_used = 0;
}

void transformer_internal::ops::maybe_reserve_scheduler_graphs(TTSTransformer & self,
                                                                TTSTransformerSession & session,
                                                                int32_t prefill_len, int32_t required_ctx) {
    auto & impl = self.impl_;

    if (!session.state_.sched) {
        return;
    }
    if (session.state_.sched_reserve_failed) {
        return;
    }
    if (session.state_.code_pred_cache.n_ctx < 16) {
        return;
    }

    if (session.state_.sched_reserved &&
        session.state_.sched_reserved_ctx >= required_ctx &&
        session.state_.sched_reserved_prefill_len >= prefill_len) {
        return;
    }

    std::string first_failed_graph;
    auto reserve_graph = [&](struct ggml_cgraph * g, const char * name) -> bool {
        if (!g) {
            if (first_failed_graph.empty()) {
                first_failed_graph = name;
            }
            return false;
        }
        const bool ok = ggml_backend_sched_reserve(session.state_.sched, g);
        ggml_backend_sched_reset(session.state_.sched);
        if (!ok && first_failed_graph.empty()) {
            first_failed_graph = name;
        }
        return ok;
    };

    bool ok = true;
    ok &= reserve_graph(build_prefill_forward_graph(self, session, prefill_len, 0), "talker prefill");
    ok &= reserve_graph(build_step_graph(self, session, std::max<int32_t>(0, required_ctx - 1)), "talker step");
    ok &= reserve_graph(build_code_pred_prefill_graph(self, session), "code predictor prefill");

    for (int step = 1; step < 15; ++step) {
        char name[32];
        snprintf(name, sizeof(name), "code predictor step %d", step);
        ok &= reserve_graph(build_code_pred_step_graph(self, session, 15, step), name);
    }

    if (ok) {
        session.state_.sched_reserved = true;
        session.state_.sched_reserve_failed = false;
        session.state_.sched_reserved_ctx = required_ctx;
        session.state_.sched_reserved_prefill_len = prefill_len;
    } else {
        session.state_.sched_reserved = false;
        session.state_.sched_reserve_failed = true;
        const char * graph_name = first_failed_graph.empty() ? "unknown graph" : first_failed_graph.c_str();
        fprintf(stderr,
                "  Scheduler reserve failed at %s; disabling reserve warmup and using dynamic graph allocation\n",
                graph_name);
    }
}

// ---------------------------------------------------------------------------
// Backward-compat wrappers (delegate to default session)
// ---------------------------------------------------------------------------

bool TTSTransformer::init_kv_cache(int32_t n_ctx) {
    ensure_default_session();
    return init_kv_cache(*default_session_, n_ctx);
}

void TTSTransformer::clear_kv_cache() {
    ensure_default_session();
    clear_kv_cache(*default_session_);
}

bool TTSTransformer::init_code_pred_kv_cache(int32_t n_ctx) {
    ensure_default_session();
    return init_code_pred_kv_cache(*default_session_, n_ctx);
}

void TTSTransformer::clear_code_pred_kv_cache() {
    ensure_default_session();
    clear_code_pred_kv_cache(*default_session_);
}

} // namespace qwen3_tts
