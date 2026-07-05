#include "tts_transformer.h"
#include "transformer/transformer_state_internal.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>
#include <atomic>
#include <string>
#include <mutex>

static std::mutex g_log_mutex;

static void log_thread(int id, const char * msg) {
    std::lock_guard<std::mutex> lock(g_log_mutex);
    fprintf(stderr, "[thread %d] %s\n", id, msg);
}

int main(int argc, char ** argv) {
    std::string model_path;
    int n_threads = 2;
    int max_frames = 64;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            n_threads = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--max-frames") == 0 && i + 1 < argc) {
            max_frames = atoi(argv[++i]);
        } else {
            if (!model_path.empty()) {
                fprintf(stderr, "Unknown argument: %s\n", argv[i]);
            } else {
                model_path = argv[i];
            }
        }
    }

    if (model_path.empty()) {
        fprintf(stderr, "Usage: %s [--model] <model_dir_or_gguf> [--threads N] [--max-frames N]\n", argv[0]);
        return 1;
    }

    if (n_threads < 2) {
        fprintf(stderr, "Need at least 2 threads for multi-session test\n");
        return 1;
    }

    fprintf(stderr, "Loading model from: %s\n", model_path.c_str());

    qwen3_tts::TTSTransformer transformer;
    std::string gguf_path = model_path;
    if (gguf_path.find(".gguf") == std::string::npos) {
        gguf_path += "/tts_transformer.gguf";
        FILE * f = fopen(gguf_path.c_str(), "rb");
        if (!f) {
            gguf_path = model_path + "/transformer.gguf";
            f = fopen(gguf_path.c_str(), "rb");
        }
        if (f) fclose(f);
    }

    if (!transformer.load_model(gguf_path)) {
        fprintf(stderr, "Failed to load model: %s\n", transformer.get_error().c_str());
        return 1;
    }

    fprintf(stderr, "Model loaded. Creating %d sessions...\n", n_threads);

    std::vector<std::unique_ptr<qwen3_tts::TTSTransformerSession>> sessions;
    for (int i = 0; i < n_threads; i++) {
        auto s = transformer.create_session();
        if (!s) {
            fprintf(stderr, "Failed to create session %d\n", i);
            return 1;
        }
        sessions.push_back(std::move(s));
    }

    fprintf(stderr, "All sessions created. Starting concurrent generation...\n");

    std::atomic<int> success_count{0};
    std::atomic<int> fail_count{0};

    auto worker = [&](int id) {
        char buf[256];
        snprintf(buf, sizeof(buf), "starting generation (max %d frames)", max_frames);
        log_thread(id, buf);

        int32_t text_tokens[] = {151672, 882, 11, 8982, 311, 4344, 151673};
        int32_t n_tokens = 7;

        std::vector<int32_t> output;
        bool ok = transformer.generate(
            *sessions[id],
            text_tokens, n_tokens,
            nullptr,
            max_frames,
            output,
            2050,
            1.05f,
            0.0f,
            0,
            1.0f,
            42 + id,
            nullptr, 0,
            nullptr, 0,
            nullptr, 0, 0,
            nullptr
        );

        if (ok && !output.empty()) {
            int n_codebooks = transformer.get_config().n_codebooks;
            int n_frames = (int)output.size() / n_codebooks;
            snprintf(buf, sizeof(buf), "generated %d frames x %d codebooks", n_frames, n_codebooks);
            log_thread(id, buf);
            success_count++;
        } else {
            snprintf(buf, sizeof(buf), "generation failed (ok=%d, output_size=%zu)", (int)ok, output.size());
            log_thread(id, buf);
            fail_count++;
        }
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < n_threads; i++) {
        threads.emplace_back(worker, i);
    }

    for (auto & t : threads) {
        t.join();
    }

    fprintf(stderr, "\nResults: %d succeeded, %d failed out of %d threads\n",
            success_count.load(), fail_count.load(), n_threads);

    if (fail_count.load() > 0) {
        fprintf(stderr, "FAIL: some threads failed\n");
        return 1;
    }

    fprintf(stderr, "PASS: all %d sessions completed concurrently\n", n_threads);
    return 0;
}
