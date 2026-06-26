#!/usr/bin/env python3
"""Dump Qwen3-TTS 12Hz speech-tokenizer codes for C++ ICL smoke tests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


def _default_python_repo() -> Path:
    return Path(__file__).resolve().parents[2] / "Qwen3-TTS"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", required=True, help="Path to the speech_tokenizer directory")
    parser.add_argument("--audio", required=True, help="Reference WAV path")
    parser.add_argument("--out", required=True, help="Output JSON array path")
    parser.add_argument("--text-model", help="Path to the parent Qwen3-TTS model snapshot for text token dumping")
    parser.add_argument("--reference-text-file", help="Reference transcript path for text token dumping")
    parser.add_argument("--reference-tokens-out", help="Output text-token ID list path")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--python-repo", default=str(_default_python_repo()))
    args = parser.parse_args()

    python_repo = Path(args.python_repo)
    if python_repo.exists():
        sys.path.insert(0, str(python_repo))

    from qwen_tts import Qwen3TTSTokenizer

    dtype = torch.bfloat16 if str(args.device).startswith("cuda") else torch.float32
    tokenizer = Qwen3TTSTokenizer.from_pretrained(
        args.tokenizer,
        device_map=args.device,
        dtype=dtype,
        attn_implementation="eager",
    )

    encoded = tokenizer.encode(args.audio)
    codes = encoded.audio_codes[0].detach().cpu().to(torch.int32)
    if codes.ndim != 2:
        raise RuntimeError(f"expected 2-D 12Hz codes, got shape={tuple(codes.shape)}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(codes.tolist(), f, separators=(",", ":"))
        f.write("\n")

    print(f"wrote {out_path} frames={codes.shape[0]} codebooks={codes.shape[1]}")

    if args.reference_tokens_out:
        if not args.text_model or not args.reference_text_file:
            raise RuntimeError("--reference-tokens-out requires --text-model and --reference-text-file")

        from transformers import AutoConfig, AutoProcessor
        from qwen_tts.core.models import Qwen3TTSConfig, Qwen3TTSProcessor

        AutoConfig.register("qwen3_tts", Qwen3TTSConfig)
        AutoProcessor.register(Qwen3TTSConfig, Qwen3TTSProcessor)
        processor = AutoProcessor.from_pretrained(args.text_model, fix_mistral_regex=True)
        ref_text = Path(args.reference_text_file).read_text(encoding="utf-8").strip()
        prompt = f"<|im_start|>assistant\n{ref_text}<|im_end|>\n"
        token_ids = processor(text=prompt, return_tensors="pt", padding=True)["input_ids"][0].tolist()
        token_out = Path(args.reference_tokens_out)
        token_out.parent.mkdir(parents=True, exist_ok=True)
        token_out.write_text(" ".join(str(int(x)) for x in token_ids) + "\n", encoding="utf-8")
        print(f"wrote {token_out} tokens={len(token_ids)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
