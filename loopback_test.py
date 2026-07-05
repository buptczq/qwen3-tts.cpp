#!/usr/bin/env python3
"""
Loopback test: TTS -> VAD -> ASR -> TTS -> VAD -> ASR pipeline.

Measures performance, real-time factor, latency, and accuracy of the
speech processing pipeline.

Usage:
    python loopback_test.py \
        --tts-model-dir /path/to/tts/models \
        --asr-model /path/to/sensevoice.gguf \
        --vad-model /path/to/fsmn-vad.gguf \
        [--iterations 3]
"""

import argparse
import time
import sys
import threading
from pathlib import Path
from typing import List, Tuple
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent / "python"))
from qwen3_tts import Qwen3TTS


# Test sentences (Chinese)
TEST_SENTENCES = [
    "欢迎大家来体验达摩院推出的语音识别模型。",
    "今天天气真不错，我们一起去公园散步吧。",
    "人工智能技术正在快速发展，改变着我们的生活方式。",
    "请注意，前方路段正在施工，请减速慢行。",
    "这个产品的用户反馈非常积极，销量持续增长。",
]


def measure_tts_vad_streaming(tts: Qwen3TTS, text: str, max_seg_ms: int = 30000) -> Tuple[np.ndarray, int, List[dict], float, float, float]:
    """Run streaming TTS -> VAD pipeline.

    TTS generates audio chunks which are immediately fed to VAD.
    Returns (full_audio, sample_rate, segments, first_chunk_ms, tts_total_ms, vad_ms).
    """
    # Create VAD stream
    vad_stream = tts.create_vad_stream(max_seg_ms=max_seg_ms)

    # Collect all audio chunks
    audio_chunks = []
    sample_rate = [24000]  # default, will be updated by callback

    # Timing
    tts_start = time.perf_counter()
    first_chunk_time = [None]  # use list to allow mutation in nested function
    vad_time = 0.0

    def on_audio_chunk(chunk: np.ndarray, sr: int) -> bool:
        """Called for each TTS audio chunk. Feed to VAD immediately."""
        nonlocal vad_time
        sample_rate[0] = sr

        # Record first chunk latency
        if first_chunk_time[0] is None:
            first_chunk_time[0] = (time.perf_counter() - tts_start) * 1000

        audio_chunks.append(chunk)

        # Resample to 16kHz for VAD if needed
        if sr != 16000:
            ratio = 16000 / sr
            new_len = int(len(chunk) * ratio)
            chunk_16k = np.interp(
                np.linspace(0, len(chunk) - 1, new_len),
                np.arange(len(chunk)),
                chunk
            ).astype(np.float32)
        else:
            chunk_16k = chunk.astype(np.float32)

        # Feed to VAD immediately
        vad_start = time.perf_counter()
        vad_stream.feed(chunk_16k)
        vad_time += (time.perf_counter() - vad_start) * 1000

        return True  # continue

    # Run streaming TTS
    success, audio, sr, err, t_ms = tts.synthesize_streaming(
        text, on_audio_chunk, collect_audio=True
    )
    tts_total_ms = (time.perf_counter() - tts_start) * 1000

    if not success:
        vad_stream.close()
        raise RuntimeError(f"TTS failed: {err}")

    first_chunk_ms = first_chunk_time[0] if first_chunk_time[0] is not None else tts_total_ms

    # Get final segments (including open segment)
    segments = vad_stream.get_segments()
    open_seg = vad_stream.get_open_segment()
    if open_seg:
        segments.append(open_seg)
    vad_stream.close()

    # Concatenate all chunks
    full_audio = np.concatenate(audio_chunks) if audio_chunks else np.array([], dtype=np.float32)

    return full_audio, sample_rate[0], segments, first_chunk_ms, tts_total_ms, vad_time


def measure_asr(tts: Qwen3TTS, audio: np.ndarray, sr: int) -> Tuple[str, float]:
    """Run ASR on audio and return (text, elapsed_ms)."""
    # Resample to 16kHz if needed
    if sr != 16000:
        ratio = 16000 / sr
        new_len = int(len(audio) * ratio)
        audio_16k = np.interp(
            np.linspace(0, len(audio) - 1, new_len),
            np.arange(len(audio)),
            audio
        ).astype(np.float32)
    else:
        audio_16k = audio.astype(np.float32)

    start = time.perf_counter()
    result = tts.transcribe_pcm(audio_16k)
    elapsed = (time.perf_counter() - start) * 1000
    if not result["success"]:
        raise RuntimeError(f"ASR failed: {result['error_msg']}")
    return result["text"], elapsed


def compute_rtf(audio_duration_ms: float, processing_time_ms: float) -> float:
    """Compute real-time factor."""
    if audio_duration_ms <= 0:
        return 0.0
    return processing_time_ms / audio_duration_ms


def simple_cer(reference: str, hypothesis: str) -> float:
    """Compute character error rate (simplified)."""
    if not reference:
        return 0.0 if not hypothesis else 1.0

    # Remove punctuation for comparison
    ref = reference.replace("。", "").replace("，", "").replace("！", "").replace("？", "")
    hyp = hypothesis.replace("。", "").replace("，", "").replace("！", "").replace("？", "")

    if not ref:
        return 0.0 if not hyp else 1.0

    # Simple character-level comparison
    matches = sum(1 for r, h in zip(ref, hyp) if r == h)
    max_len = max(len(ref), len(hyp))
    return 1.0 - (matches / max_len) if max_len > 0 else 0.0


def run_concurrent_test(tts: Qwen3TTS, n_threads: int, rounds: int):
    """Run TTS->VAD->ASR pipeline concurrently from multiple threads.

    Each thread gets its own session and processes a subset of sentences.
    Verifies that concurrent inference does not crash or produce errors.
    """
    sessions = [tts.create_session() for _ in range(n_threads)]
    errors = [None] * n_threads
    results = [[] for _ in range(n_threads)]

    sentences_per_thread = (len(TEST_SENTENCES) + n_threads - 1) // n_threads

    def worker(thread_id: int):
        session = sessions[thread_id]
        start_idx = thread_id * sentences_per_thread
        end_idx = min(start_idx + sentences_per_thread, len(TEST_SENTENCES))
        my_sentences = TEST_SENTENCES[start_idx:end_idx]

        for iteration in range(rounds):
            for sent_idx, text in enumerate(my_sentences):
                label = f"[T{thread_id}] iter={iteration+1} sent={sent_idx+1}/{len(my_sentences)}"
                try:
                    audio, sr, segments, first_chunk_ms, tts_ms, vad_ms = \
                        measure_tts_vad_streaming_session(tts, session, text)
                    audio_duration_ms = len(audio) / sr * 1000

                    asr_texts = []
                    asr_total_ms = 0
                    for seg in segments:
                        s = int(seg["start_ms"] * sr / 1000)
                        e = int(seg["end_ms"] * sr / 1000)
                        seg_text, asr_ms = measure_asr(tts, audio[s:e], sr)
                        asr_texts.append(seg_text)
                        asr_total_ms += asr_ms

                    combined = "".join(asr_texts)
                    cer = simple_cer(text, combined)
                    total_ms = tts_ms + vad_ms + asr_total_ms
                    rtf = compute_rtf(audio_duration_ms, total_ms)

                    print(f"  {label}: \"{text}\" -> \"{combined}\" "
                          f"(CER={cer:.3f}, RTF={rtf:.3f}, {total_ms:.0f}ms)",
                          file=sys.stderr)
                    results[thread_id].append({
                        "text": text, "output": combined,
                        "cer": cer, "rtf": rtf, "total_ms": total_ms,
                    })
                except Exception as exc:
                    print(f"  {label}: FAILED — {exc}", file=sys.stderr)
                    errors[thread_id] = str(exc)

    print(f"\n[Concurrent] {n_threads} threads, {rounds} round(s), "
          f"{len(TEST_SENTENCES)} sentences", file=sys.stderr)
    print("-" * 70, file=sys.stderr)

    wall_start = time.perf_counter()
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall_ms = (time.perf_counter() - wall_start) * 1000

    for s in sessions:
        s.close()

    all_results = []
    for r in results:
        all_results.extend(r)

    any_error = any(e is not None for e in errors)
    if any_error:
        print(f"\n[Concurrent] FAILED — errors: {[e for e in errors if e]}", file=sys.stderr)
        return 1

    if not all_results:
        print("\n[Concurrent] FAILED — no results collected", file=sys.stderr)
        return 1

    avg_cer = sum(r["cer"] for r in all_results) / len(all_results)
    avg_rtf = sum(r["rtf"] for r in all_results) / len(all_results)
    total_sentences = len(all_results)

    print(f"\n[Concurrent] {total_sentences} sentences in {wall_ms:.0f}ms wall-clock", file=sys.stderr)
    print(f"  Avg CER: {avg_cer:.3f}, Avg RTF: {avg_rtf:.3f}", file=sys.stderr)
    print(f"  Result: {'PASS' if avg_cer < 0.3 else 'FAIL'}", file=sys.stderr)
    return 0 if avg_cer < 0.3 else 1


def measure_tts_vad_streaming_session(tts, session, text, max_seg_ms=30000):
    """Session-aware variant of measure_tts_vad_streaming."""
    vad_stream = tts.create_vad_stream(max_seg_ms=max_seg_ms)
    audio_chunks = []
    sample_rate = [24000]
    tts_start = time.perf_counter()
    first_chunk_time = [None]
    vad_time = 0.0

    def on_audio_chunk(chunk, sr):
        nonlocal vad_time
        sample_rate[0] = sr
        if first_chunk_time[0] is None:
            first_chunk_time[0] = (time.perf_counter() - tts_start) * 1000
        audio_chunks.append(chunk)
        if sr != 16000:
            ratio = 16000 / sr
            new_len = int(len(chunk) * ratio)
            chunk_16k = np.interp(
                np.linspace(0, len(chunk) - 1, new_len),
                np.arange(len(chunk)), chunk
            ).astype(np.float32)
        else:
            chunk_16k = chunk.astype(np.float32)
        vad_start = time.perf_counter()
        vad_stream.feed(chunk_16k)
        vad_time += (time.perf_counter() - vad_start) * 1000
        return True

    success, audio, sr, err, t_ms = tts.synthesize_streaming_session(
        session, text, on_audio_chunk, collect_audio=True
    )
    tts_total_ms = (time.perf_counter() - tts_start) * 1000

    if not success:
        vad_stream.close()
        raise RuntimeError(f"TTS failed: {err}")

    first_chunk_ms = first_chunk_time[0] if first_chunk_time[0] is not None else tts_total_ms
    segments = vad_stream.get_segments()
    open_seg = vad_stream.get_open_segment()
    if open_seg:
        segments.append(open_seg)
    vad_stream.close()

    full_audio = np.concatenate(audio_chunks) if audio_chunks else np.array([], dtype=np.float32)
    return full_audio, sample_rate[0], segments, first_chunk_ms, tts_total_ms, vad_time


def run_loopback_test(args):
    """Run the full loopback test with multiple rounds."""
    print("=" * 70)
    print("Loopback Test: TTS -> VAD -> ASR -> TTS -> VAD -> ASR ...")
    print("=" * 70)

    # Initialize
    print("\n[1/4] Initializing models...")
    init_start = time.perf_counter()

    tts = Qwen3TTS()

    # Load TTS models
    print(f"  Loading TTS models from {args.tts_model_dir}...")
    if not tts.load_models(args.tts_model_dir):
        raise RuntimeError("Failed to load TTS models")

    # Load and cache ASR model
    print(f"  Loading ASR model from {args.asr_model}...")
    if not tts.load_asr_model(args.asr_model):
        raise RuntimeError("Failed to load ASR model")

    # Load and cache VAD model
    print(f"  Loading VAD model from {args.vad_model}...")
    if not tts.load_vad_model(args.vad_model):
        raise RuntimeError("Failed to load VAD model")

    init_elapsed = (time.perf_counter() - init_start) * 1000
    print(f"  Model initialization: {init_elapsed:.1f} ms")

    # Run test iterations
    print(f"\n[2/4] Running {args.iterations} iterations, {args.rounds} rounds each...")
    print("-" * 70)

    all_results = []
    for iteration in range(args.iterations):
        print(f"\nIteration {iteration + 1}/{args.iterations}:")

        for idx, original_text in enumerate(TEST_SENTENCES):
            print(f"\n  [{idx + 1}/{len(TEST_SENTENCES)}] Original: {original_text}")

            current_text = original_text
            round_results = []

            for round_num in range(args.rounds):
                print(f"\n    Round {round_num + 1}/{args.rounds}: \"{current_text}\"")

                # Step 1+2: Streaming TTS -> VAD (overlapped)
                audio, sr, segments, first_chunk_ms, tts_total_ms, vad_ms = measure_tts_vad_streaming(tts, current_text)
                audio_duration_ms = len(audio) / sr * 1000
                print(f"      TTS: first_chunk={first_chunk_ms:.1f}ms, total={tts_total_ms:.1f}ms (audio: {audio_duration_ms:.0f}ms, RTF: {compute_rtf(audio_duration_ms, tts_total_ms):.3f})")
                print(f"      VAD: {vad_ms:.1f}ms, {len(segments)} segments")

                # Step 3: ASR on each segment
                asr_texts = []
                asr_total_ms = 0
                for seg_idx, seg in enumerate(segments):
                    # Extract segment audio
                    start_sample = int(seg["start_ms"] * sr / 1000)
                    end_sample = int(seg["end_ms"] * sr / 1000)
                    seg_audio = audio[start_sample:end_sample]

                    seg_text, asr_ms = measure_asr(tts, seg_audio, sr)
                    asr_total_ms += asr_ms
                    asr_texts.append(seg_text)
                    print(f"        Segment {seg_idx + 1} ({seg['start_ms']}-{seg['end_ms']} ms): "
                          f"ASR: {asr_ms:.1f} ms -> \"{seg_text}\"")

                combined_text = "".join(asr_texts)

                # Compute metrics for this round
                total_processing_ms = tts_total_ms + vad_ms + asr_total_ms
                overall_rtf = compute_rtf(audio_duration_ms, total_processing_ms)
                cer_from_original = simple_cer(original_text, combined_text)

                print(f"      Output: \"{combined_text}\"")
                print(f"      Total: {total_processing_ms:.1f} ms, RTF: {overall_rtf:.3f}, CER(orig): {cer_from_original:.3f}")

                round_results.append({
                    "round": round_num + 1,
                    "input_text": current_text,
                    "output_text": combined_text,
                    "first_chunk_ms": first_chunk_ms,
                    "tts_ms": tts_total_ms,
                    "vad_ms": vad_ms,
                    "asr_ms": asr_total_ms,
                    "total_ms": total_processing_ms,
                    "audio_duration_ms": audio_duration_ms,
                    "rtf": overall_rtf,
                    "cer_from_original": cer_from_original,
                })

                # Feed output back as input for next round
                current_text = combined_text

            all_results.append({
                "original": original_text,
                "rounds": round_results,
            })

    # Summary
    print("\n" + "=" * 70)
    print("[3/4] Summary")
    print("=" * 70)

    # Flatten all round results for aggregation
    flat_results = []
    for sentence_result in all_results:
        for rr in sentence_result["rounds"]:
            flat_results.append(rr)

    total_sentences = len(all_results)
    total_rounds = len(flat_results)

    # Per-round averages
    round_stats = {}
    for rr in flat_results:
        r = rr["round"]
        if r not in round_stats:
            round_stats[r] = {"first_chunk_ms": [], "tts_ms": [], "vad_ms": [], "asr_ms": [],
                              "total_ms": [], "audio_ms": [], "rtf": [], "cer": []}
        round_stats[r]["first_chunk_ms"].append(rr["first_chunk_ms"])
        round_stats[r]["tts_ms"].append(rr["tts_ms"])
        round_stats[r]["vad_ms"].append(rr["vad_ms"])
        round_stats[r]["asr_ms"].append(rr["asr_ms"])
        round_stats[r]["total_ms"].append(rr["total_ms"])
        round_stats[r]["audio_ms"].append(rr["audio_duration_ms"])
        round_stats[r]["rtf"].append(rr["rtf"])
        round_stats[r]["cer"].append(rr["cer_from_original"])

    def avg(lst):
        return sum(lst) / len(lst) if lst else 0

    print(f"\nProcessed {total_sentences} sentences x {args.rounds} rounds = {total_rounds} total passes")

    print(f"\nPer-round averages:")
    print(f"  {'Round':<6} {'1st chunk':>10} {'TTS(ms)':>10} {'VAD(ms)':>10} {'ASR(ms)':>10} {'Total(ms)':>10} {'Audio(ms)':>10} {'RTF':>8} {'CER':>8}")
    print(f"  {'-'*6} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*8} {'-'*8}")
    for r in sorted(round_stats.keys()):
        s = round_stats[r]
        print(f"  {r:<6} {avg(s['first_chunk_ms']):>10.1f} {avg(s['tts_ms']):>10.1f} {avg(s['vad_ms']):>10.1f} "
              f"{avg(s['asr_ms']):>10.1f} {avg(s['total_ms']):>10.1f} {avg(s['audio_ms']):>10.1f} {avg(s['rtf']):>8.3f} {avg(s['cer']):>8.3f}")

    # Overall averages
    print(f"\nOverall averages:")
    print(f"  First chunk: {avg([r['first_chunk_ms'] for r in flat_results]):8.1f} ms")
    print(f"  TTS total:   {avg([r['tts_ms'] for r in flat_results]):8.1f} ms")
    print(f"  VAD:         {avg([r['vad_ms'] for r in flat_results]):8.1f} ms")
    print(f"  ASR:         {avg([r['asr_ms'] for r in flat_results]):8.1f} ms")
    print(f"  Total:       {avg([r['total_ms'] for r in flat_results]):8.1f} ms")
    print(f"  Audio:       {avg([r['audio_duration_ms'] for r in flat_results]):8.1f} ms")
    print(f"  Avg RTF:     {avg([r['rtf'] for r in flat_results]):.3f}")
    print(f"  Avg CER:     {avg([r['cer_from_original'] for r in flat_results]):.3f}")

    # Per-sentence round-by-round
    print(f"\n[4/4] Per-sentence round-by-round results:")
    print("-" * 70)
    for i, sentence_result in enumerate(all_results):
        print(f"\n{i + 1}. Original: {sentence_result['original']}")
        for rr in sentence_result["rounds"]:
            r = rr["round"]
            print(f"   Round {r}: \"{rr['output_text']}\"  (CER: {rr['cer_from_original']:.3f}, RTF: {rr['rtf']:.3f})")

    final_cer = avg([r["cer_from_original"] for r in flat_results])
    print(f"\nDone! Final CER: {final_cer:.3f}")
    seq_ok = final_cer < 0.3

    # Concurrent test
    concurrent_ok = True
    if args.concurrent > 1:
        print("\n" + "=" * 70)
        print(f"Concurrent Test: {args.concurrent} threads")
        print("=" * 70)
        concurrent_ok = run_concurrent_test(tts, args.concurrent, args.rounds) == 0

    print("\nCleaning up...")
    tts.free_asr_model()
    tts.free_vad_model()
    tts.close()

    return 0 if (seq_ok and concurrent_ok) else 1


def main():
    parser = argparse.ArgumentParser(description="Loopback test for TTS -> VAD -> ASR pipeline")
    parser.add_argument("--tts-model-dir", required=True, help="Path to TTS model directory")
    parser.add_argument("--asr-model", required=True, help="Path to ASR model GGUF (SenseVoice or Paraformer)")
    parser.add_argument("--vad-model", required=True, help="Path to VAD model GGUF (FSMN-VAD)")
    parser.add_argument("--iterations", type=int, default=1, help="Number of test iterations")
    parser.add_argument("--rounds", type=int, default=3, help="Number of TTS->VAD->ASR rounds per sentence (default: 3)")
    parser.add_argument("--concurrent", type=int, default=0, metavar="N",
                        help="Run concurrent test with N threads after the sequential test (default: 0 = skip)")
    args = parser.parse_args()

    # Validate paths
    if not Path(args.tts_model_dir).is_dir():
        print(f"Error: TTS model directory not found: {args.tts_model_dir}")
        return 1
    if not Path(args.asr_model).is_file():
        print(f"Error: ASR model not found: {args.asr_model}")
        return 1
    if not Path(args.vad_model).is_file():
        print(f"Error: VAD model not found: {args.vad_model}")
        return 1

    return run_loopback_test(args)


if __name__ == "__main__":
    sys.exit(main())
