#!/usr/bin/env python3
"""
WebSocket streaming TTS client for qwen3-tts.cpp.

Sends text to the WebSocket streaming server and saves the received audio.

Usage:
  # Basic synthesis (no voice cloning)
  uv run python/web_streaming_client.py --text "Hello, world!" -o hello.wav

  # With a saved reference (via HTTP API)
  uv run python/web_streaming_client.py --text "Hello!" --ref-name my_voice -o out.wav

  # List saved references
  uv run python/web_streaming_client.py --list-refs

  # Upload a reference
  uv run python/web_streaming_client.py --upload-ref ref.wav --ref-name my_voice

  # Custom server address
  uv run python/web_streaming_client.py --host 192.168.1.100 --port 8765 \\
      --text "Hello" -o out.wav
"""

import argparse
import asyncio
import base64
import json
import struct
import sys
import wave
from pathlib import Path
from typing import Optional

import aiohttp

SAMPLE_RATE = 24000
CHANNELS = 1
SAMPLE_WIDTH = 2  # PCM16


def pcm16_to_wav(pcm16_bytes: bytes, path: Path, sample_rate: int = SAMPLE_RATE):
    """Write raw PCM16 bytes to a WAV file."""
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16_bytes)


async def list_refs(session: aiohttp.ClientSession, url: str):
    """List all saved reference audio files."""
    async with session.get(f"{url}/api/refs") as resp:
        if resp.status != 200:
            print(f"Error: HTTP {resp.status}", file=sys.stderr)
            return
        refs = await resp.json()
        if not refs:
            print("No saved references.")
            return
        print(f"Saved references ({len(refs)}):")
        for r in refs:
            print(f"  {r['name']:20s}  created: {r.get('created_at', '?')}")


async def upload_ref(session: aiohttp.ClientSession, url: str,
                     wav_path: str, ref_name: str, ref_text: str = ""):
    """Upload a reference audio file to the server."""
    wav_path = Path(wav_path)
    if not wav_path.exists():
        print(f"Error: file not found: {wav_path}", file=sys.stderr)
        return False

    data = aiohttp.FormData()
    data.add_field("file", wav_path.read_bytes(),
                   filename=wav_path.name,
                   content_type="audio/wav")
    data.add_field("name", ref_name)
    if ref_text:
        data.add_field("reference_text", ref_text)

    async with session.post(f"{url}/api/refs", data=data) as resp:
        if resp.status == 200:
            result = await resp.json()
            print(f"Uploaded reference '{ref_name}' successfully.")
            return True
        else:
            body = await resp.text()
            print(f"Error uploading reference (HTTP {resp.status}): {body}", file=sys.stderr)
            return False


async def synthesize_text(ws_url: str, text: str, params: dict,
                          output: Optional[Path] = None,
                          timeout: float = 120.0):
    """
    Connect to the WebSocket server, send a synthesis request,
    collect audio chunks, and optionally save to a WAV file.
    """
    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(ws_url, timeout=timeout) as ws:
            # Send synthesis request
            await ws.send_json({"text": text, "params": params})
            print(f"Sent: text={text!r}  params={params}", file=sys.stderr)

            audio_chunks: list[bytes] = []
            request_id: Optional[str] = None
            done_wav_b64: Optional[str] = None
            done_sample_rate: int = SAMPLE_RATE

            async for msg in ws:
                if msg.type != aiohttp.WSMsgType.TEXT:
                    continue

                try:
                    data = msg.json()
                except json.JSONDecodeError:
                    continue

                msg_type = data.get("type", "")

                if msg_type == "queued":
                    print(f"  [queued] position={data.get('position')}", file=sys.stderr)

                elif msg_type == "start":
                    request_id = data.get("request_id", "?")
                    print(f"  [start]  request_id={request_id}", file=sys.stderr)

                elif msg_type == "audio":
                    chunk_b64 = data.get("data", "")
                    sr = data.get("sample_rate", SAMPLE_RATE)
                    chunk = base64.b64decode(chunk_b64)
                    audio_chunks.append(chunk)
                    duration = len(chunk) / (SAMPLE_WIDTH * sr)
                    print(f"  [audio]  {len(chunk)} bytes ({duration:.2f}s)", file=sys.stderr)

                elif msg_type == "done":
                    request_id = data.get("request_id", request_id)
                    done_wav_b64 = data.get("wav_base64")
                    done_sample_rate = data.get("sample_rate", SAMPLE_RATE)
                    print(f"  [done]   request_id={request_id}", file=sys.stderr)
                    # If the server also sends the full WAV in the done message,
                    # prefer that over the streamed chunks.
                    if done_wav_b64:
                        full_wav = base64.b64decode(done_wav_b64)
                        if output:
                            output.write_bytes(full_wav)
                            print(f"  Saved full audio ({len(full_wav)} bytes) to {output}", file=sys.stderr)
                        return True
                    break

                elif msg_type == "cancelled":
                    print(f"  [cancelled] request_id={data.get('request_id', '?')}", file=sys.stderr)
                    return False

                elif msg_type == "error":
                    print(f"  [error]   {data.get('message', 'unknown error')}", file=sys.stderr)
                    return False

            # If we got streamed chunks but no done wav_base64, assemble them
            if audio_chunks and output:
                pcm_data = b"".join(audio_chunks)
                pcm16_to_wav(pcm_data, output, done_sample_rate)
                print(f"  Saved streamed audio ({len(pcm_data)} bytes) to {output}", file=sys.stderr)

            return True


async def main():
    parser = argparse.ArgumentParser(
        description="Test client for qwen3-tts.cpp WebSocket streaming server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Server connection
    parser.add_argument("--host", default="127.0.0.1", help="Server host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765, help="Server port (default: 8765)")

    # Actions
    parser.add_argument("--text", "-t", help="Text to synthesize")
    parser.add_argument("--output", "-o", type=Path, default=None, help="Output WAV file path")
    parser.add_argument("--ref-name", help="Name of a saved reference for voice cloning")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature (default: 0.7)")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max audio tokens (default: 512)")

    # HTTP API actions
    parser.add_argument("--list-refs", action="store_true", help="List saved references via HTTP API")
    parser.add_argument("--upload-ref", type=str, default=None,
                        help="Upload a reference WAV file via HTTP API")
    parser.add_argument("--ref-text", type=str, default="",
                        help="Reference transcript text (for --upload-ref)")

    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"
    ws_url = f"ws://{args.host}:{args.port}/ws"

    async with aiohttp.ClientSession() as session:
        # HTTP API actions
        if args.list_refs:
            await list_refs(session, base_url)
            return

        if args.upload_ref:
            await upload_ref(session, base_url, args.upload_ref, args.ref_name or "default", args.ref_text)
            return

        # WebSocket synthesis
        if not args.text:
            parser.print_help()
            sys.exit(1)

        params = {
            "temperature": args.temperature,
            "max_audio_tokens": args.max_tokens,
        }
        if args.ref_name:
            params["ref_name"] = args.ref_name

        if not args.output:
            args.output = Path("output.wav")

        success = await synthesize_text(ws_url, args.text, params, args.output)
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())
