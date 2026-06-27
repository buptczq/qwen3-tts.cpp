# Framework Benchmarking Notes

This document tracks the local Qwen3 TTS implementations that can be compared for README-quality benchmark numbers. The benchmark harness lives in `scripts/benchmark_frameworks.ps1`.

## Harness

Preflight only, without synthesis:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\benchmark_frameworks.ps1 -ValidateOnly
```

Run a comparable voice-clone benchmark later:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\benchmark_frameworks.ps1 `
  -Scenario voice_clone `
  -Variant 1.7b-base `
  -Runs 3 `
  -MaxTokens 128
```

The harness records one CSV with raw runs, one CSV summary, logs, generated WAVs, git commit ids, dirty counts, audio duration, peak/RMS sanity checks, wall time, and RTF.

## Current Local Implementations

| Implementation | Local path | Current state | Update recommendation before benchmarks |
| --- | --- | --- | --- |
| qwen3-tts.cpp | `qwen3-tts.cpp` | Current work is committed on `perf/decoder-regression-timing`; local branch is ahead of origin. | Push before publishing benchmark numbers, so the recorded commit is reachable. |
| Serveurperso qwentts.cpp | `qwentts.cpp-serveurperso` | At origin/master, only an untracked build directory. | Already current. Use as-is. |
| audio.cpp | `audio.cpp` | Clean, behind origin by 3 commits after fetch. | Safe to fast-forward; changes look like release/docs updates. |
| Qwen3-TTS official Python | `Qwen3-TTS` | Behind origin by 1 commit; untracked `.venv`. | Safe to fast-forward; the remote change is a finetuning fix, likely not inference-critical. |
| faster-qwen3-tts | `faster-qwen3-tts` | Heavily divergent from origin and has untracked package metadata. | Do not pull in-place. Use a fresh clone or backup branch if we want latest upstream numbers. |
| ht-vllm-omni | `ht-vllm-omni` | Heavily divergent from origin. Remote contains many vLLM/Omni changes including Qwen3-TTS latency work. | Do not pull in-place. Use a fresh clone/worktree if we include it later. |
| predict-woo original qwen3-tts.cpp | `qwen3-tts-original.cpp` | Dirty local changes and behind upstream. | Do not update in-place. Use a clean clone if it is needed as a historical baseline. |
| Qwen3-TTS-Cpp | `Qwen3-TTS-Cpp` | Up to date with its origin; untracked build directory. | No update needed. Add to the harness only if we decide what scenario/model path is comparable. |

## Notes On Fairness

- The current shared scenario is `voice_clone`, because the Python base implementations are centered on reference-audio generation.
- `basic` is useful for C++ stacks, but is not yet a fair all-framework scenario.
- The first version measures command-line end-to-end wall time. That includes model load for each process. This is reproducible, but not the same as a resident server/session benchmark.
- Audio sanity checks are built in because we already saw a silent Serveurperso run; any WAV with invalid, empty, silent, or very low-level audio is marked separately from process success.
- Serveurperso currently has local Q8_0 1.7B GGUF files. qwen3-tts.cpp has f16, q8_0, and q4_k-family 1.7B files. For strict quantization comparisons, pass matching model files explicitly.
