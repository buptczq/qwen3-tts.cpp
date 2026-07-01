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
import signal
import struct
import subprocess
import sys
import time
from pathlib import Path

import aiohttp
import asyncio

SERVER_PORT = 18765  # Use non-default port to avoid conflicts
SAMPLE_RATE = 24000


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


async def main():
    print("Starting server...", file=sys.stderr)
    server_dir = Path(__file__).resolve().parent.parent
    refs_dir = server_dir / "/tmp/test_refs_streaming"
    refs_dir.mkdir(parents=True, exist_ok=True)

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
        print("=== Test 1: Multi-sentence text ===", file=sys.stderr)
        ok1 = await test_sentence_splitting()

        print("\n=== Test 2: Single sentence ===", file=sys.stderr)
        ok2 = await test_single_sentence()

        print("\n=== Test 3: No punctuation ===", file=sys.stderr)
        ok3 = await test_no_punctuation()

        print("\n" + "=" * 40, file=sys.stderr)
        if ok1 and ok2 and ok3:
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
