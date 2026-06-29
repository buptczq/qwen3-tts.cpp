#!/usr/bin/env python3
"""Demo: streaming TTS via libqwen3_tts ctypes bindings."""

import sys
import time
from pathlib import Path

# Add the python package to path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from qwen3_tts import Qwen3TTS


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <model_dir> [text]")
        sys.exit(1)

    model_dir = sys.argv[1]
    text = sys.argv[2] if len(sys.argv) > 2 else "Hello, this is a test of the streaming text to speech system."

    tts = Qwen3TTS()
    print(f"Loading models from {model_dir}...")
    t0 = time.time()
    ok = tts.load_models(model_dir)
    if not ok:
        print(f"Failed to load models")
        sys.exit(1)
    print(f"  loaded in {time.time()-t0:.2f}s")
    print(f"  capabilities: {tts.capabilities}")
    if tts.available_speakers:
        print(f"  speakers: {tts.available_speakers}")

    # ── batch synthesis ──────────────────────────────────────────────
    print(f"\nBatch synthesis: {text!r}")
    t0 = time.time()
    success, audio, sr, err, t_ms = tts.synthesize(
        text,
        temperature=0.7,
        n_threads=4,
        print_timing=True,
    )
    print(f"  success={success}, sr={sr}, len={len(audio)}, time={t_ms}ms ({time.time()-t0:.2f}s)")
    if err:
        print(f"  error: {err}")

    # Save batch output
    import soundfile as sf
    sf.write("output_batch.wav", audio, sr)
    print(f"  saved to output_batch.wav")

    # ── streaming synthesis ──────────────────────────────────────────
    print(f"\nStreaming synthesis: {text!r}")

    chunk_count = [0]
    def on_chunk(samples, sr):
        chunk_count[0] += 1
        print(f"  chunk {chunk_count[0]}: {len(samples)} samples @ {sr}Hz")
        return True  # continue

    t0 = time.time()
    success, audio, sr, err, t_ms = tts.synthesize_streaming(
        text,
        on_audio_chunk=on_chunk,
        temperature=0.7,
        n_threads=4,
        chunk_sec=1.0,
        collect_audio=True,
    )
    print(f"  success={success}, sr={sr}, len={len(audio)}, chunks={chunk_count[0]}, time={t_ms}ms ({time.time()-t0:.2f}s)")
    if err:
        print(f"  error: {err}")

    if len(audio) > 0:
        sf.write("output_streaming.wav", audio, sr)
        print(f"  saved to output_streaming.wav")

    tts.close()


if __name__ == "__main__":
    main()
