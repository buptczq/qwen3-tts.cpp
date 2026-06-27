#!/usr/bin/env python3
"""Small worker used by benchmark_frameworks.ps1 for Python Qwen3 TTS stacks."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np


def _import_torch():
    import torch

    return torch


def _write_wav(path: Path, wav, sample_rate: int) -> None:
    import soundfile as sf

    arr = wav[0] if isinstance(wav, (list, tuple)) else wav
    if hasattr(arr, "detach"):
        arr = arr.detach().cpu().numpy()
    arr = np.asarray(arr, dtype=np.float32).reshape(-1)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), arr, sample_rate)


def _audio_duration(path: Path) -> float:
    import soundfile as sf

    info = sf.info(str(path))
    if info.samplerate <= 0:
        return 0.0
    return float(info.frames) / float(info.samplerate)


def _set_seed(seed: int) -> None:
    if seed < 0:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch = _import_torch()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _sync_cuda() -> None:
    torch = _import_torch()
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _torch_dtype(name: str):
    torch = _import_torch()
    mapping = {
        "auto": None,
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return mapping[name]


def _load_official(args: argparse.Namespace):
    if args.repo:
        sys.path.insert(0, str(Path(args.repo).resolve()))
    from qwen_tts import Qwen3TTSModel

    kwargs = {}
    if args.device_map:
        kwargs["device_map"] = args.device_map
    dtype = _torch_dtype(args.dtype)
    if dtype is not None:
        kwargs["dtype"] = dtype
    return Qwen3TTSModel.from_pretrained(args.model, **kwargs)


def _load_faster(args: argparse.Namespace):
    if args.repo:
        sys.path.insert(0, str(Path(args.repo).resolve()))
    from faster_qwen3_tts import FasterQwen3TTS

    return FasterQwen3TTS.from_pretrained(args.model)


def _generate_voice_clone(model, args: argparse.Namespace):
    kwargs = {
        "text": args.text,
        "language": args.language,
        "ref_audio": args.reference_audio,
        "ref_text": args.reference_text,
        "max_new_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "do_sample": not args.greedy,
        "repetition_penalty": args.repetition_penalty,
    }

    if args.backend == "faster":
        kwargs["xvec_only"] = args.xvec_only
        kwargs["non_streaming_mode"] = args.non_streaming_mode
    else:
        kwargs["x_vector_only_mode"] = args.xvec_only
        kwargs["non_streaming_mode"] = args.non_streaming_mode

    return model.generate_voice_clone(**kwargs)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["official", "faster"], required=True)
    parser.add_argument("--repo", default="")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--language", default="English")
    parser.add_argument("--reference-audio", required=True)
    parser.add_argument("--reference-text", required=True)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--xvec-only", action="store_true")
    parser.add_argument("--non-streaming-mode", action="store_true")
    parser.add_argument("--device-map", default="cuda")
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="bfloat16")
    args = parser.parse_args()

    _set_seed(args.seed)
    t_load0 = time.perf_counter()
    if args.backend == "official":
        model = _load_official(args)
    else:
        model = _load_faster(args)
    _sync_cuda()
    load_s = time.perf_counter() - t_load0

    t0 = time.perf_counter()
    wavs, sample_rate = _generate_voice_clone(model, args)
    _sync_cuda()
    synth_s = time.perf_counter() - t0

    out = Path(args.output)
    _write_wav(out, wavs, int(sample_rate))
    duration_s = _audio_duration(out)

    result = {
        "backend": args.backend,
        "output": str(out),
        "load_seconds": load_s,
        "synth_seconds": synth_s,
        "wall_seconds": load_s + synth_s,
        "audio_seconds": duration_s,
        "rtf_audio_per_wall": duration_s / (load_s + synth_s) if load_s + synth_s > 0 else 0.0,
        "rtf_audio_per_synth": duration_s / synth_s if synth_s > 0 else 0.0,
        "sample_rate": int(sample_rate),
    }
    print("BENCHMARK_JSON " + json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
