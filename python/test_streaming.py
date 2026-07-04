#!/usr/bin/env python3
"""
Integration test for the WebSocket streaming server with sentence splitting.

Starts the server as a subprocess, connects a client, sends a multi-sentence
text, and verifies that:
1. Audio chunks are received
2. sentence_end markers are received
3. The done message contains a valid WAV
"""

import base64
import json
import os
import shutil
import signal
import struct
import subprocess
import sys
import time
from pathlib import Path

import aiohttp
import asyncio
import numpy as np

from web_streaming import split_sentences, split_sub_segments

SERVER_PORT = 18765  # Use non-default port to avoid conflicts
SAMPLE_RATE = 24000
REFS_SOURCE = Path(__file__).resolve().parent.parent / "refs" / "测试.json"


def decode_pcm16_b64(b64data: str) -> np.ndarray:
    pcm = base64.b64decode(b64data)
    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    return samples


async def test_sentence_splitting():
    """Test that multi-sentence text produces sentence_end markers."""
    ws_url = f"ws://127.0.0.1:{SERVER_PORT}/ws"
    text = "Hello, how are you? I am fine, thank you. That is great!"

    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(ws_url, timeout=30) as ws:
            await ws.send_json({"text": text, "params": {"temperature": 0}})

            audio_count = 0
            sentence_end_count = 0
            sentence_texts = []
            got_done = False
            done_wav = None

            async for msg in ws:
                if msg.type != aiohttp.WSMsgType.TEXT:
                    continue

                try:
                    data = msg.json()
                except json.JSONDecodeError:
                    continue

                msg_type = data.get("type", "")
                print(f"  [{msg_type}]", file=sys.stderr)

                if msg_type == "audio":
                    audio_count += 1
                elif msg_type == "sentence_end":
                    sentence_end_count += 1
                    sentence_texts.append(data.get("text", ""))
                elif msg_type == "done":
                    got_done = True
                    break
                elif msg_type == "error":
                    print(f"  ERROR: {data.get('message')}", file=sys.stderr)
                    return False

            print(f"\nResults:", file=sys.stderr)
            print(f"  Audio chunks: {audio_count}", file=sys.stderr)
            print(f"  Sentence ends: {sentence_end_count}", file=sys.stderr)
            print(f"  Sentence texts: {sentence_texts}", file=sys.stderr)
            print(f"  Got done: {got_done}", file=sys.stderr)

            # Verify
            if sentence_end_count < 2:
                print(f"  FAIL: Expected >= 2 sentence_end markers, got {sentence_end_count}", file=sys.stderr)
                return False

            if not got_done:
                print("  FAIL: No done message received", file=sys.stderr)
                return False

            if audio_count < 3:
                print(f"  FAIL: Too few audio chunks ({audio_count})", file=sys.stderr)
                return False

            print("  PASS: All checks passed!", file=sys.stderr)
            return True


async def test_single_sentence():
    """Test that single-sentence text still works correctly."""
    ws_url = f"ws://127.0.0.1:{SERVER_PORT}/ws"
    text = "Hello world."

    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(ws_url, timeout=30) as ws:
            await ws.send_json({"text": text, "params": {"temperature": 0}})

            sentence_end_count = 0
            got_done = False

            async for msg in ws:
                if msg.type != aiohttp.WSMsgType.TEXT:
                    continue

                try:
                    data = msg.json()
                except json.JSONDecodeError:
                    continue

                msg_type = data.get("type", "")
                print(f"  [{msg_type}]", file=sys.stderr)

                if msg_type == "sentence_end":
                    sentence_end_count += 1
                elif msg_type == "done":
                    got_done = True
                    break
                elif msg_type == "error":
                    print(f"  ERROR: {data.get('message')}", file=sys.stderr)
                    return False

            print(f"\nSingle sentence results:", file=sys.stderr)
            print(f"  Sentence ends: {sentence_end_count}", file=sys.stderr)
            print(f"  Got done: {got_done}", file=sys.stderr)

            if sentence_end_count != 1:
                print(f"  FAIL: Expected 1 sentence_end, got {sentence_end_count}", file=sys.stderr)
                return False

            print("  PASS!", file=sys.stderr)
            return True


async def test_no_punctuation():
    """Test text with no sentence-ending punctuation."""
    ws_url = f"ws://127.0.0.1:{SERVER_PORT}/ws"
    text = "this is a long text without any punctuation at all it should be treated as one sentence"

    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(ws_url, timeout=30) as ws:
            await ws.send_json({"text": text, "params": {"temperature": 0}})

            sentence_end_count = 0
            got_done = False

            async for msg in ws:
                if msg.type != aiohttp.WSMsgType.TEXT:
                    continue

                try:
                    data = msg.json()
                except json.JSONDecodeError:
                    continue

                msg_type = data.get("type", "")
                print(f"  [{msg_type}]", file=sys.stderr)

                if msg_type == "sentence_end":
                    sentence_end_count += 1
                elif msg_type == "done":
                    got_done = True
                    break
                elif msg_type == "error":
                    print(f"  ERROR: {data.get('message')}", file=sys.stderr)
                    return False

            print(f"\nNo-punctuation results:", file=sys.stderr)
            print(f"  Sentence ends: {sentence_end_count}", file=sys.stderr)

            if sentence_end_count != 1:
                print(f"  FAIL: Expected 1 sentence_end, got {sentence_end_count}", file=sys.stderr)
                return False

            print("  PASS!", file=sys.stderr)
            return True


def test_split_sentences_unit():
    """Unit tests for split_sentences: commas must NOT cause splits."""
    tests_passed = 0
    tests_failed = 0

    def check(desc, text, expected_count):
        nonlocal tests_passed, tests_failed
        result = split_sentences(text)
        if len(result) == expected_count:
            tests_passed += 1
            print(f"  PASS: {desc} -> {len(result)} segment(s)", file=sys.stderr)
        else:
            tests_failed += 1
            print(f"  FAIL: {desc} -> expected {expected_count}, got {len(result)}: {result}",
                  file=sys.stderr)

    check("Chinese commas only",
          "佛得角驻华大使阿林多\u00b7多罗萨里奥表示，很高兴看到自己国家的球队\u201c佛得角蓝鲨\u201d所带来的精彩表现，自己对此感到非常自豪，自己国家的队伍虽小但内心强大。",
          1)

    check("Chinese period split",
          "你好世界。再见世界。",
          2)

    check("English commas preserved",
          "Hello, world, how are you?",
          1)

    check("Mixed punctuation",
          "你好，世界。再见，世界！",
          2)

    check("No punctuation",
          "hello world",
          1)

    check("Semicolon and colon don't split",
          "第一点；第二点：详情如下",
          1)

    check("Multiple sentence endings",
          "第一句。第二句！第三句？结束。",
          4)

    check("Newline splits",
          "line one\nline two\nline three",
          3)

    print(f"\n  Unit tests: {tests_passed} passed, {tests_failed} failed", file=sys.stderr)
    return tests_failed == 0


def test_split_sub_segments_unit():
    """Unit tests for split_sub_segments: pause punctuation splits, sentence punctuation doesn't."""
    tests_passed = 0
    tests_failed = 0

    def check(desc, text, expected_count):
        nonlocal tests_passed, tests_failed
        result = split_sub_segments(text)
        if len(result) == expected_count:
            tests_passed += 1
            print(f"  PASS: {desc} -> {len(result)} sub-segment(s)", file=sys.stderr)
        else:
            tests_failed += 1
            print(f"  FAIL: {desc} -> expected {expected_count}, got {len(result)}: {result}",
                  file=sys.stderr)

    check("Chinese commas split",
          "佛得角驻华大使表示，很高兴，自己感到自豪。",
          3)

    check("No pause punctuation",
          "你好世界。再见世界。",
          1)

    check("Mixed Chinese punctuation",
          "第一，第二；第三：结束。",
          4)

    check("English commas split",
          "Hello, world, how are you?",
          3)

    check("No punctuation at all",
          "hello world",
          1)

    check("Concatenated text matches original",
          "A，B，C。",
          3)

    joined = "".join(split_sub_segments("A，B，C。"))
    if joined == "A，B，C。":
        tests_passed += 1
        print(f"  PASS: join(split) == original", file=sys.stderr)
    else:
        tests_failed += 1
        print(f"  FAIL: join(split) = {joined!r} != 'A，B，C。'", file=sys.stderr)

    print(f"\n  Sub-segment unit tests: {tests_passed} passed, {tests_failed} failed",
          file=sys.stderr)
    return tests_failed == 0


async def test_voice_clone_chinese():
    """Test voice cloning with Chinese text containing commas.

    Uses refs/测试.json as reference audio. Verifies that:
    1. Chinese commas do NOT cause sentence splits
    2. Audio output is non-silent (no 没声音)
    3. Audio output has no excessive clipping (no 破音)
    """
    ws_url = f"ws://127.0.0.1:{SERVER_PORT}/ws"
    text = ("佛得角驻华大使阿林多\u00b7多罗萨里奥表示，"
            "很高兴看到自己国家的球队\u201c佛得角蓝鲨\u201d"
            "所带来的精彩表现，自己对此感到非常自豪，"
            "自己国家的队伍虽小但内心强大。"
            "此外，他再次感谢中国：\u201c是中国帮我们圆梦\u201d")

    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(ws_url, timeout=aiohttp.ClientTimeout(total=120)) as ws:
            await ws.send_json({
                "text": text,
                "params": {"ref_name": "测试", "temperature": 0},
            })

            audio_chunks = []
            sentence_end_count = 0
            sentence_texts = []
            got_done = False

            async for msg in ws:
                if msg.type != aiohttp.WSMsgType.TEXT:
                    continue

                try:
                    data = msg.json()
                except json.JSONDecodeError:
                    continue

                msg_type = data.get("type", "")
                print(f"  [{msg_type}]", file=sys.stderr)

                if msg_type == "audio":
                    audio_chunks.append(data["data"])
                elif msg_type == "sentence_end":
                    sentence_end_count += 1
                    sentence_texts.append(data.get("text", ""))
                elif msg_type == "done":
                    got_done = True
                    break
                elif msg_type == "error":
                    print(f"  ERROR: {data.get('message')}", file=sys.stderr)
                    return False

            print(f"\nVoice clone results:", file=sys.stderr)
            print(f"  Audio chunks: {len(audio_chunks)}", file=sys.stderr)
            print(f"  Sentence ends: {sentence_end_count}", file=sys.stderr)
            print(f"  Sentence texts: {sentence_texts}", file=sys.stderr)
            print(f"  Got done: {got_done}", file=sys.stderr)

            ok = True

            # The text has 2 sentences (split by 。).
            # Sentence 1 has 3 commas → 4 sub-segments.
            # Sentence 2 has 1 comma + 1 colon → 2 sub-segments (colon splits "此外，他再次感谢中国：" from rest).
            # Total: 6 sentence_end markers at sub-sentence boundaries.
            if sentence_end_count < 2:
                print(f"  FAIL: Expected >= 2 sentence_ends, got {sentence_end_count}",
                      file=sys.stderr)
                ok = False
            else:
                print(f"  PASS: {sentence_end_count} sentence_end markers (sub-sentence granularity)",
                      file=sys.stderr)

            if not got_done:
                print("  FAIL: No done message received", file=sys.stderr)
                return False

            if len(audio_chunks) < 1:
                print(f"  FAIL: No audio chunks received", file=sys.stderr)
                return False

            all_samples = np.concatenate([decode_pcm16_b64(c) for c in audio_chunks])
            duration = len(all_samples) / SAMPLE_RATE
            rms = np.sqrt(np.mean(all_samples ** 2))
            peak = np.max(np.abs(all_samples))
            silent_ratio = np.mean(np.abs(all_samples) < 0.001)

            print(f"  Audio duration: {duration:.2f}s", file=sys.stderr)
            print(f"  RMS: {rms:.4f}, Peak: {peak:.4f}", file=sys.stderr)
            print(f"  Near-zero sample ratio: {silent_ratio:.2%}", file=sys.stderr)

            if duration < 1.0:
                print(f"  FAIL: Audio too short ({duration:.2f}s)", file=sys.stderr)
                ok = False
            else:
                print(f"  PASS: Audio duration OK ({duration:.2f}s)", file=sys.stderr)

            if rms < 0.005:
                print(f"  FAIL: Audio nearly silent (RMS={rms:.4f})", file=sys.stderr)
                ok = False
            else:
                print(f"  PASS: Audio non-silent (RMS={rms:.4f})", file=sys.stderr)

            if peak > 0.99:
                clip_ratio = np.mean(np.abs(all_samples) > 0.99)
                if clip_ratio > 0.05:
                    print(f"  FAIL: Excessive clipping ({clip_ratio:.2%} samples)",
                          file=sys.stderr)
                    ok = False
                else:
                    print(f"  PASS: Clipping within bounds ({clip_ratio:.2%})",
                          file=sys.stderr)
            else:
                print(f"  PASS: No clipping (peak={peak:.4f})", file=sys.stderr)

            if silent_ratio > 0.5:
                print(f"  FAIL: Too much silence ({silent_ratio:.2%})", file=sys.stderr)
                ok = False
            else:
                print(f"  PASS: Content density OK ({1 - silent_ratio:.2%} active)",
                      file=sys.stderr)

            if ok:
                print("  PASS: All voice clone checks passed!", file=sys.stderr)
            return ok


async def main():
    print("=" * 50, file=sys.stderr)
    print("Unit tests: split_sentences", file=sys.stderr)
    print("=" * 50, file=sys.stderr)
    ok_unit = test_split_sentences_unit()

    print("\n" + "=" * 50, file=sys.stderr)
    print("Unit tests: split_sub_segments", file=sys.stderr)
    print("=" * 50, file=sys.stderr)
    ok_sub = test_split_sub_segments_unit()

    if not ok_unit or not ok_sub:
        print("\nUnit tests FAILED — aborting.", file=sys.stderr)
        sys.exit(1)

    print("\nStarting server...", file=sys.stderr)
    server_dir = Path(__file__).resolve().parent.parent
    refs_dir = server_dir / "/tmp/test_refs_streaming"
    refs_dir.mkdir(parents=True, exist_ok=True)

    if REFS_SOURCE.exists():
        shutil.copy2(REFS_SOURCE, refs_dir / "测试.json")
        print(f"Copied reference audio to {refs_dir / '测试.json'}", file=sys.stderr)
    else:
        print(f"WARNING: {REFS_SOURCE} not found, voice clone test will be skipped",
              file=sys.stderr)

    proc = await asyncio.create_subprocess_exec(
        "uv", "run", "python/web_streaming.py",
        str(server_dir / "models"),
        "--port", str(SERVER_PORT),
        "--refs-dir", str(refs_dir),
        "--log-level", "WARNING",
        cwd=server_dir,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )

    # Wait for server to start
    await asyncio.sleep(8)

    # Check if server is still running
    if proc.returncode is not None:
        print(f"Server failed to start (exit code {proc.returncode})", file=sys.stderr)
        sys.exit(1)

    try:
        print("\n=== Test 1: Multi-sentence text ===", file=sys.stderr)
        ok1 = await test_sentence_splitting()

        print("\n=== Test 2: Single sentence ===", file=sys.stderr)
        ok2 = await test_single_sentence()

        print("\n=== Test 3: No punctuation ===", file=sys.stderr)
        ok3 = await test_no_punctuation()

        print("\n=== Test 4: Voice clone + Chinese commas ===", file=sys.stderr)
        if REFS_SOURCE.exists():
            ok4 = await test_voice_clone_chinese()
        else:
            print("  SKIP: reference audio not available", file=sys.stderr)
            ok4 = True

        all_ok = ok1 and ok2 and ok3 and ok4
        print("\n" + "=" * 40, file=sys.stderr)
        if all_ok:
            print("ALL TESTS PASSED!", file=sys.stderr)
        else:
            print("SOME TESTS FAILED!", file=sys.stderr)
            sys.exit(1)
    finally:
        proc.send_signal(signal.SIGINT)
        try:
            await asyncio.wait_for(proc.wait(), timeout=5)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()


if __name__ == "__main__":
    asyncio.run(main())
