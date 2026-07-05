#include "tts_transformer.h"
#include "transformer/transformer_state_internal.h"
#include "gguf_loader.h"
#include "transformer/transformer_internal.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <sys/stat.h>

namespace qwen3_tts {

namespace {

std::string filename_lower(const std::string & path) {
    size_t slash = path.find_last_of("/\\");
    std::string name = slash == std::string::npos ? path : path.substr(slash + 1);
    std::transform(name.begin(), name.end(), name.begin(),
                   [](unsigned char c) { return (char) std::tolower(c); });
    return name;
}

} // namespace

void TTSTransformer::unload_model() {
    default_session_.reset();

    free_transformer_model(impl_->model);

    impl_->coreml_code_predictor.unload();
    impl_->use_coreml_code_predictor = false;
    impl_->coreml_code_predictor_path.clear();
    impl_->skip_ggml_code_pred_layers = false;

    if (impl_->state.backend) {
        release_preferred_backend(impl_->state.backend);
        impl_->state.backend = nullptr;
    }
    if (impl_->state.backend_cpu) {
        ggml_backend_free(impl_->state.backend_cpu);
        impl_->state.backend_cpu = nullptr;
    }

    last_hidden_.clear();
}

bool TTSTransformer::load_model(const std::string & model_path) {
    unload_model();

    impl_->skip_ggml_code_pred_layers = false;
#if defined(__APPLE__)
    const char * use_coreml_env = std::getenv("QWEN3_TTS_USE_COREML");
    bool coreml_disabled = false;
    if (use_coreml_env && use_coreml_env[0] != '\0') {
        std::string use_coreml = use_coreml_env;
        std::transform(use_coreml.begin(), use_coreml.end(), use_coreml.begin(),
                       [](unsigned char c) { return (char) std::tolower(c); });
        coreml_disabled = use_coreml == "0" || use_coreml == "false" ||
                          use_coreml == "off" || use_coreml == "no";
    }

    if (!coreml_disabled) {
        std::string coreml_path;
        const char * override_env = std::getenv("QWEN3_TTS_COREML_MODEL");
        if (override_env && override_env[0] != '\0') {
            coreml_path = override_env;
        } else {
            size_t slash = model_path.find_last_of("/\\");
            const std::string model_dir = (slash == std::string::npos) ? "." : model_path.substr(0, slash);
            coreml_path = model_dir + "/coreml/code_predictor.mlpackage";
        }

        struct stat st = {};
        if (stat(coreml_path.c_str(), &st) == 0) {
            impl_->skip_ggml_code_pred_layers = true;
        } else if (use_coreml_env && use_coreml_env[0] != '\0') {
            impl_->skip_ggml_code_pred_layers = true;
        }
    }
#endif

    struct ggml_context * meta_ctx = nullptr;
    struct gguf_init_params params = {
        /*.no_alloc =*/ true,
        /*.ctx      =*/ &meta_ctx,
    };

    struct gguf_context * ctx = gguf_init_from_file(model_path.c_str(), params);
    if (!ctx) {
        error_msg_ = "Failed to open GGUF file: " + model_path;
        return false;
    }

    if (!transformer_internal::ops::parse_config(*this, ctx)) {
        gguf_free(ctx);
        if (meta_ctx) ggml_free(meta_ctx);
        return false;
    }
    {
        const std::string name = filename_lower(model_path);
        auto & cfg = impl_->model.config;
        if (name.find("customvoice") != std::string::npos && cfg.tts_model_type == "base") {
            cfg.tts_model_type = "custom_voice";
            fprintf(stderr, "  TTS model type inferred from filename: %s\n", cfg.tts_model_type.c_str());
        } else if (name.find("voicedesign") != std::string::npos && cfg.tts_model_type == "base") {
            cfg.tts_model_type = "voice_design";
            fprintf(stderr, "  TTS model type inferred from filename: %s\n", cfg.tts_model_type.c_str());
        }
    }

    if (!transformer_internal::ops::create_tensors(*this, ctx)) {
        gguf_free(ctx);
        if (meta_ctx) ggml_free(meta_ctx);
        return false;
    }

    if (!transformer_internal::ops::load_tensor_data(*this, model_path, ctx)) {
        free_transformer_model(impl_->model);
        gguf_free(ctx);
        if (meta_ctx) ggml_free(meta_ctx);
        return false;
    }

    if (!impl_->skip_ggml_code_pred_layers) {
        const auto & cfg = impl_->model.config;
        const bool projection_required = cfg.hidden_size > cfg.code_pred_hidden_size;
        const bool likely_legacy_1p7 = (cfg.hidden_size > 1024 &&
                                        impl_->model.code_pred_small_to_mtp_weight == nullptr);
        if ((projection_required || likely_legacy_1p7) &&
            impl_->model.code_pred_small_to_mtp_weight == nullptr) {
            error_msg_ =
                "Model is missing code_pred.mtp_proj/code_pred.small_to_mtp projection weights. "
                "Re-convert with the updated scripts/convert_tts_to_gguf.py.";
            free_transformer_model(impl_->model);
            gguf_free(ctx);
            if (meta_ctx) ggml_free(meta_ctx);
            return false;
        }
    }

    gguf_free(ctx);
    if (meta_ctx) ggml_free(meta_ctx);

    impl_->state.backend = init_preferred_backend("TTSTransformer", &error_msg_);
    if (!impl_->state.backend) {
        return false;
    }
    ggml_backend_dev_t device = ggml_backend_get_device(impl_->state.backend);
    const char * device_name = device ? ggml_backend_dev_name(device) : "Unknown";
    fprintf(stderr, "  TTSTransformer backend: %s\n", device_name);

    if (device && ggml_backend_dev_type(device) != GGML_BACKEND_DEVICE_TYPE_CPU) {
        impl_->state.backend_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        if (!impl_->state.backend_cpu) {
            error_msg_ = "Failed to initialize CPU fallback backend for TTSTransformer";
            return false;
        }
    }

    if (!transformer_internal::ops::try_init_coreml_code_predictor(*this, model_path)) {
        return false;
    }

    return true;
}

bool transformer_internal::ops::try_init_coreml_code_predictor(TTSTransformer & self, const std::string & model_path) {
    auto & impl = self.impl_;
    auto & error_msg = self.error_msg_;
    (void) model_path;
    impl->use_coreml_code_predictor = false;
    impl->coreml_code_predictor_path.clear();

    const char * use_coreml_env = std::getenv("QWEN3_TTS_USE_COREML");
    bool coreml_disabled = false;
    if (use_coreml_env && use_coreml_env[0] != '\0') {
        std::string use_coreml = use_coreml_env;
        std::transform(use_coreml.begin(), use_coreml.end(), use_coreml.begin(),
                       [](unsigned char c) { return (char) std::tolower(c); });
        coreml_disabled = use_coreml == "0" || use_coreml == "false" ||
                          use_coreml == "off" || use_coreml == "no";
    }

    if (coreml_disabled) {
        return true;
    }

#if !defined(__APPLE__)
    if (use_coreml_env && use_coreml_env[0] != '\0') {
        fprintf(stderr, "  CoreML code predictor requested but this build is not on Apple platform\n");
    }
    return true;
#else
    std::string coreml_path;
    const char * override_env = std::getenv("QWEN3_TTS_COREML_MODEL");
    if (override_env && override_env[0] != '\0') {
        coreml_path = override_env;
    } else {
        size_t slash = model_path.find_last_of("/\\");
        const std::string model_dir = (slash == std::string::npos) ? "." : model_path.substr(0, slash);
        coreml_path = model_dir + "/coreml/code_predictor.mlpackage";
    }

    if (!impl->coreml_code_predictor.load(coreml_path, impl->model.config.n_codebooks - 1)) {
        if (impl->skip_ggml_code_pred_layers) {
            error_msg = "CoreML code predictor load failed in strict mode: " + impl->coreml_code_predictor.get_error();
            return false;
        } else {
            fprintf(stderr, "  CoreML code predictor load failed: %s\n",
                    impl->coreml_code_predictor.get_error().c_str());
            fprintf(stderr, "  Falling back to GGML code predictor\n");
            return true;
        }
    }

    impl->use_coreml_code_predictor = true;
    impl->coreml_code_predictor_path = coreml_path;
    fprintf(stderr, "  CoreML code predictor enabled: %s\n", impl->coreml_code_predictor_path.c_str());
    return true;
#endif
}

} // namespace qwen3_tts
