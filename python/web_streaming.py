#!/usr/bin/env python3
"""
WebSocket streaming TTS server with persistent reference audio.

Reference audio is stored as JSON files in <refs_dir>/<name>.json:
  {"name": "...", "reference_text": "...", "wav_base64": "...", "created_at": "..."}

HTTP API
--------
POST   /api/refs              Upload reference audio (multipart: file + name + reference_text)
GET    /api/refs              List all saved references (without wav data)
GET    /api/refs/<name>       Get reference details (includes wav_base64)
DELETE /api/refs/<name>       Delete a reference

WebSocket API
-------------
Client -> Server (JSON):
  {"text": "...", "params": {"ref_name": "..."}}  — enqueue synthesis
  {"type": "cancel"}                                — cancel current synthesis

Server -> Client (JSON lines):
  {"type": "queued",    "position": N}
  {"type": "start",     "request_id": "...", "text": "..."}
  {"type": "audio",     "data": "<base64 PCM16>", "sample_rate": 24000}
  {"type": "done",      "request_id": "..."}
  {"type": "cancelled", "request_id": "..."}
  {"type": "error",     "message": "..."}
"""

import argparse
import asyncio
import atexit
import base64
import json
import logging
import sys
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

import aiohttp
from aiohttp import web

from qwen3_tts import Qwen3TTS

logger = logging.getLogger("web_streaming")

SAMPLE_RATE = 24000

# ── reference audio persistence ─────────────────────────────────────────────


def _load_refs(refs_dir: Path) -> list[dict]:
    """Load all reference metadata (without wav data) from refs_dir."""
    refs = []
    if not refs_dir.exists():
        return refs
    for f in sorted(refs_dir.iterdir()):
        if f.suffix == ".json":
            try:
                data = json.loads(f.read_text("utf-8"))
                refs.append({
                    "name": data["name"],
                    "reference_text": data.get("reference_text", ""),
                    "created_at": data.get("created_at", ""),
                    "size": len(data.get("wav_base64", "")),
                })
            except Exception as e:
                logger.warning("Failed to load ref %s: %s", f.name, e)
    return refs


def _save_ref(refs_dir: Path, name: str, wav_base64: str, reference_text: str):
    """Save a reference audio as a JSON file."""
    refs_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "name": name,
        "reference_text": reference_text,
        "wav_base64": wav_base64,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    path = refs_dir / f"{name}.json"
    path.write_text(json.dumps(data, ensure_ascii=False), "utf-8")


def _delete_ref(refs_dir: Path, name: str) -> bool:
    path = refs_dir / f"{name}.json"
    if path.exists():
        path.unlink()
        return True
    return False


def _get_ref_full(refs_dir: Path, name: str) -> Optional[dict]:
    path = refs_dir / f"{name}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text("utf-8"))
    except Exception:
        return None


def pcm16_encode(samples: np.ndarray) -> bytes:
    """Convert float32 [-1, 1] samples to PCM16 bytes."""
    clipped = np.clip(samples, -1.0, 1.0)
    return (clipped * 32767).astype(np.int16).tobytes()


def wav_to_pcm(wav_bytes: bytes) -> np.ndarray:
    """Decode WAV bytes to float32 PCM samples normalized to [-1, 1].

    Supports mono/stereo 16-bit WAV. Multi-channel audio is averaged to mono.
    """
    import struct

    if wav_bytes[:4] != b"RIFF" or wav_bytes[8:12] != b"WAVE":
        raise ValueError("Not a valid WAV file")

    # Parse chunks
    pos = 12
    sample_rate = 24000
    bits_per_sample = 16
    channels = 1
    data = None

    while pos + 8 <= len(wav_bytes):
        chunk_id = wav_bytes[pos:pos+4]
        chunk_size = struct.unpack_from("<I", wav_bytes, pos + 4)[0]
        pos += 8
        if chunk_id == b"fmt ":
            fmt_data = wav_bytes[pos:pos+chunk_size]
            audio_format = struct.unpack_from("<H", fmt_data, 0)[0]
            channels = struct.unpack_from("<H", fmt_data, 2)[0]
            sample_rate = struct.unpack_from("<I", fmt_data, 4)[0]
            bits_per_sample = struct.unpack_from("<H", fmt_data, 14)[0]
            if audio_format != 1:  # PCM
                raise ValueError(f"Unsupported WAV format: {audio_format}")
        elif chunk_id == b"data":
            data = wav_bytes[pos:pos+chunk_size]
            break
        pos += chunk_size

    if data is None:
        raise ValueError("No data chunk found in WAV")

    # Convert to float32
    if bits_per_sample == 16:
        raw = np.frombuffer(data, dtype=np.int16).astype(np.float32)
    elif bits_per_sample == 32:
        raw = np.frombuffer(data, dtype=np.int32).astype(np.float32)
    elif bits_per_sample == 8:
        raw = np.frombuffer(data, dtype=np.uint8).astype(np.float32) - 128.0
    else:
        raise ValueError(f"Unsupported bit depth: {bits_per_sample}")

    # Normalize to [-1, 1]
    if bits_per_sample == 32:
        raw /= 2147483648.0
    else:
        raw /= 32768.0

    # Average channels to mono
    if channels > 1:
        raw = raw.reshape(-1, channels).mean(axis=1)

    return raw


# ── per-client session ──────────────────────────────────────────────────────

@dataclass
class Request:
    """A single TTS synthesis request queued by the client."""
    id: str
    text: str
    params: dict
    cancelled: bool = False


_CHUNK_END = None


class ClientSession:
    """Manages a queue of synthesis requests for one WebSocket connection."""

    def __init__(self, tts: Qwen3TTS, ws, refs_dir: Path):
        self._tts = tts
        self._ws = ws
        self._refs_dir = refs_dir
        self._queue: deque[Request] = deque()
        self._current: Optional[Request] = None
        self._cancel_event = asyncio.Event()
        self._worker_task: Optional[asyncio.Task] = None

    async def start(self):
        self._worker_task = asyncio.create_task(self._worker_loop())

    async def stop(self):
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass

    async def enqueue(self, text: str, params: dict) -> str:
        req = Request(id=uuid.uuid4().hex[:12], text=text, params=params)
        self._queue.append(req)
        await self._send({"type": "queued", "position": len(self._queue)})
        return req.id

    def cancel_current(self):
        self._cancel_event.set()

    async def _worker_loop(self):
        while True:
            while not self._queue:
                await asyncio.sleep(0.05)
            req = self._queue.popleft()
            self._current = req
            self._cancel_event.clear()

            try:
                await self._send({
                    "type": "start",
                    "request_id": req.id,
                    "text": req.text,
                })
                wav_b64 = await self._synthesize(req)
                if req.cancelled:
                    await self._send({"type": "cancelled", "request_id": req.id})
                else:
                    await self._send({
                        "type": "done",
                        "request_id": req.id,
                        "wav_base64": wav_b64,
                        "sample_rate": SAMPLE_RATE,
                    })
            except Exception as e:
                logger.exception("synthesis failed")
                try:
                    await self._send({
                        "type": "error", "request_id": req.id,
                        "message": str(e),
                    })
                except Exception:
                    pass
            finally:
                self._current = None

    async def _synthesize(self, req: Request):
        """Run streaming synthesis in a thread, sending PCM16 chunks via WebSocket."""
        params = {
            "temperature": 0.7,
            "n_threads": 4,
            "chunk_sec": 1.0,
            "left_context_sec": 2.0,
            "collect_audio": False,
            "print_progress": False,
            "print_timing": False,
        }
        params.update(req.params)

        # Resolve reference audio from persisted refs
        ref_name = params.pop("ref_name", None)
        ref_samples: Optional[np.ndarray] = None
        ref_text: Optional[str] = None
        if ref_name:
            ref_data = _get_ref_full(self._refs_dir, ref_name)
            if ref_data is None:
                logger.warning("ref_name '%s' not found, falling back to no voice clone", ref_name)
            else:
                wav_bytes = base64.b64decode(ref_data["wav_base64"])
                ref_samples = wav_to_pcm(wav_bytes)
                ref_text = ref_data.get("reference_text") or None

        chunk_queue: asyncio.Queue = asyncio.Queue()

        def _run():
            def _on_chunk(samples: np.ndarray, sr: int) -> bool:
                if self._cancel_event.is_set():
                    return False
                fut = asyncio.run_coroutine_threadsafe(
                    chunk_queue.put(samples), loop
                )
                fut.result(timeout=10)
                return True

            try:
                if ref_samples is not None:
                    self._tts.synthesize_with_voice_streaming_from_pcm(
                        req.text, ref_samples,
                        on_audio_chunk=_on_chunk,
                        reference_text=ref_text,
                        **params,
                    )
                else:
                    self._tts.synthesize_streaming(
                        req.text, on_audio_chunk=_on_chunk, **params,
                    )
            except Exception as e:
                asyncio.run_coroutine_threadsafe(
                    chunk_queue.put(e), loop
                ).result()
            finally:
                asyncio.run_coroutine_threadsafe(
                    chunk_queue.put(_CHUNK_END), loop
                ).result()

        loop = asyncio.get_running_loop()
        thread_task = loop.run_in_executor(None, _run)

        while True:
            item = await chunk_queue.get()
            if item is _CHUNK_END:
                break
            if isinstance(item, Exception):
                raise item
            if req.cancelled:
                break
            pcm = pcm16_encode(item)
            b64 = base64.b64encode(pcm).decode("ascii")
            await self._send({
                "type": "audio",
                "data": b64,
                "sample_rate": SAMPLE_RATE,
            })

        await thread_task

    async def _send(self, msg: dict):
        try:
            await self._ws.send_json(msg)
        except Exception:
            pass


# ── WebSocket handler ──────────────────────────────────────────────────────

async def ws_handler(ws, tts: Qwen3TTS, refs_dir: Path):
    session = ClientSession(tts, ws, refs_dir)
    await session.start()
    try:
        async for msg in ws:
            if msg.type != aiohttp.WSMsgType.TEXT:
                continue
            raw = msg.data
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                await ws.send_json({"type": "error", "message": "invalid JSON"})
                continue

            msg_type = data.get("type", "synthesize")

            if msg_type == "cancel":
                session.cancel_current()
            elif msg_type == "synthesize" or "text" in data:
                text = data.get("text", "").strip()
                if not text:
                    await ws.send_json({"type": "error", "message": "empty text"})
                    continue
                params = data.get("params", {})
                await session.enqueue(text, params)
            else:
                await ws.send_json({"type": "error", "message": f"unknown type: {msg_type}"})
    finally:
        await session.stop()


# ── HTTP API handlers ──────────────────────────────────────────────────────

async def handle_list_refs(request):
    refs_dir: Path = request.app["refs_dir"]
    return web.json_response(_load_refs(refs_dir))


async def handle_get_ref(request):
    refs_dir: Path = request.app["refs_dir"]
    name = request.match_info["name"]
    data = _get_ref_full(refs_dir, name)
    if data is None:
        return web.json_response({"error": "not found"}, status=404)
    return web.json_response(data)


async def handle_delete_ref(request):
    refs_dir: Path = request.app["refs_dir"]
    name = request.match_info["name"]
    if _delete_ref(refs_dir, name):
        return web.json_response({"status": "deleted"})
    return web.json_response({"error": "not found"}, status=404)


async def handle_upload_ref(request):
    refs_dir: Path = request.app["refs_dir"]
    reader = await request.multipart()

    name = None
    reference_text = ""
    wav_bytes = None

    async for part in reader:
        if part.name == "name":
            name = (await part.read()).decode("utf-8").strip()
        elif part.name == "reference_text":
            reference_text = (await part.read()).decode("utf-8").strip()
        elif part.name == "file":
            wav_bytes = await part.read()

    if not name:
        return web.json_response({"error": "name is required"}, status=400)
    if not wav_bytes:
        return web.json_response({"error": "file is required"}, status=400)
    if not wav_bytes.startswith(b"RIFF"):
        return web.json_response({"error": "file is not a valid WAV"}, status=400)

    wav_base64 = base64.b64encode(wav_bytes).decode("ascii")
    _save_ref(refs_dir, name, wav_base64, reference_text)

    logger.info("Saved reference audio: %s", name)
    return web.json_response({
        "name": name,
        "reference_text": reference_text,
        "size": len(wav_bytes),
    })


# ── HTTP index page ────────────────────────────────────────────────────────

INDEX_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Qwen3-TTS Streaming</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         background: #f5f5f5; height: 100vh; display: flex; flex-direction: column; }
  #header { background: #1a73e8; color: #fff; padding: 14px 20px; font-size: 18px;
            font-weight: 600; flex-shrink: 0; display: flex; align-items: center; gap: 12px; }
  #header .sub { font-size: 12px; font-weight: 400; opacity: .8; }
  #chat { flex: 1; overflow-y: auto; padding: 16px 20px; display: flex;
          flex-direction: column; gap: 10px; }
  .msg { max-width: 80%; padding: 10px 14px; border-radius: 12px; line-height: 1.5;
         word-break: break-word; }
  .msg.user { background: #e3f2fd; align-self: flex-end; }
  .msg.system { background: #fff; border: 1px solid #e0e0e0; align-self: flex-start; }
  .msg.system .meta { font-size: 12px; color: #666; margin-top: 4px; }
  .msg.system .error { color: #d32f2f; }
  #controls { flex-shrink: 0; padding: 12px 20px; background: #fff;
              border-top: 1px solid #e0e0e0; display: flex; gap: 10px;
              align-items: flex-start; flex-wrap: wrap; }
  #input { flex: 1; min-width: 200px; padding: 10px 14px; border: 1px solid #dadce0;
           border-radius: 8px; font-size: 14px; resize: none; outline: none; }
  #input:focus { border-color: #1a73e8; }
  .btn { padding: 10px 20px; border: none; border-radius: 8px; font-size: 14px;
         cursor: pointer; font-weight: 500; white-space: nowrap; }
  .btn-primary { background: #1a73e8; color: #fff; }
  .btn-primary:hover { background: #1557b0; }
  .btn-danger { background: #d32f2f; color: #fff; }
  .btn-danger:hover { background: #b71c1c; }
  .btn-secondary { background: #e0e0e0; color: #333; }
  .btn-secondary:hover { background: #bdbdbd; }
  .btn-success { background: #2e7d32; color: #fff; }
  .btn-success:hover { background: #1b5e20; }
  #params-panel { flex-shrink: 0; background: #fff; border-top: 1px solid #e0e0e0;
                  padding: 12px 20px; display: none; flex-wrap: wrap; gap: 12px;
                  align-items: center; }
  #params-panel.open { display: flex; }
  .param { display: flex; align-items: center; gap: 6px; }
  .param label { font-size: 13px; color: #555; }
  .param input { width: 70px; padding: 4px 8px; border: 1px solid #dadce0;
                 border-radius: 4px; font-size: 13px; }
  .param input[type="range"] { width: 100px; }
  .param .val { font-size: 12px; color: #1a73e8; min-width: 24px; text-align: right; }
  #ref-bar { flex-shrink: 0; background: #fff; border-top: 1px solid #e0e0e0;
             padding: 8px 20px; display: flex; gap: 8px; align-items: center;
             font-size: 13px; flex-wrap: wrap; }
  #ref-bar .label { color: #555; }
  #ref-bar select { padding: 4px 8px; border: 1px solid #dadce0; border-radius: 4px;
                    font-size: 13px; min-width: 150px; }
  #ref-bar input[type="file"] { display: none; }
  #upload-dialog { display: none; position: fixed; top: 0; left: 0; right: 0; bottom: 0;
                   background: rgba(0,0,0,0.4); align-items: center; justify-content: center;
                   z-index: 100; }
  #upload-dialog.open { display: flex; }
  #upload-dialog .box { background: #fff; border-radius: 12px; padding: 24px; width: 420px;
                        max-width: 90vw; box-shadow: 0 8px 32px rgba(0,0,0,0.2); }
  #upload-dialog h3 { margin-bottom: 16px; }
  #upload-dialog label { display: block; font-size: 13px; color: #555; margin: 8px 0 4px; }
  #upload-dialog input, #upload-dialog textarea { width: 100%; padding: 8px 10px;
    border: 1px solid #dadce0; border-radius: 6px; font-size: 14px; }
  #upload-dialog textarea { resize: vertical; min-height: 50px; }
  #upload-dialog .btns { display: flex; gap: 8px; justify-content: flex-end; margin-top: 16px; }
  #connect-bar { flex-shrink: 0; padding: 8px 20px; background: #fff3e0;
                 border-bottom: 1px solid #ffe0b2; font-size: 13px; display: none; }
  #connect-bar.error { background: #ffebee; border-color: #ffcdd2; display: block; }
  #connect-bar.connected { background: #e8f5e9; border-color: #c8e6c9; display: block; }
</style>
</head>
<body>
<div id="header">Qwen3-TTS 流式合成 <span class="sub">voice clone</span></div>
<div id="connect-bar"></div>
<div id="chat"></div>
<div id="ref-bar">
  <span class="label">参考音频:</span>
  <select id="ref-select">
    <option value="">不使用</option>
  </select>
  <button class="btn btn-secondary" id="btn-refresh" style="padding:4px 8px;font-size:12px">刷新</button>
  <button class="btn btn-success" id="btn-add-ref" style="padding:4px 12px;font-size:12px">+ 新建</button>
  <button class="btn btn-danger" id="btn-del-ref" style="padding:4px 12px;font-size:12px;display:none">删除</button>
</div>
<div id="params-panel">
  <div class="param"><label>温度</label>
    <input type="range" id="p-temp" min="0" max="2" step="0.1" value="0.7">
    <span class="val" id="p-temp-v">0.7</span></div>
  <div class="param"><label>top-p</label>
    <input type="range" id="p-topp" min="0" max="1" step="0.05" value="0.9">
    <span class="val" id="p-topp-v">0.9</span></div>
  <div class="param"><label>top-k</label>
    <input type="number" id="p-topk" value="50" min="0" max="200"></div>
  <div class="param"><label>重复惩罚</label>
    <input type="range" id="p-rep" min="1" max="2" step="0.05" value="1.05">
    <span class="val" id="p-rep-v">1.05</span></div>
  <div class="param"><label>chunk(秒)</label>
    <input type="number" id="p-chunk" value="1.0" min="0.1" max="10" step="0.1"></div>
  <div class="param"><label>上文(秒)</label>
    <input type="number" id="p-leftctx" value="2.0" min="0" max="10" step="0.5"></div>
  <div class="param"><label>语言</label>
    <select id="p-lang">
      <option value="2050">English (en)</option>
      <option value="2055">中文 (zh)</option>
      <option value="2058">日本語 (ja)</option>
      <option value="2064">한국어 (ko)</option>
      <option value="2069">Русский (ru)</option>
      <option value="2053">Deutsch (de)</option>
      <option value="2061">Français (fr)</option>
      <option value="2054">Español (es)</option>
    </select></div>
  <div class="param" style="flex:1;min-width:120px"><label>风格指令</label>
    <input type="text" id="p-instr" value="" placeholder="e.g. 用平静的语气说" style="width:auto;flex:1"></div>
  <div class="param"><label>说话人</label>
    <input type="text" id="p-speaker" value="" placeholder="speaker name" style="width:100px"></div>
</div>
<div id="controls">
  <textarea id="input" rows="2" placeholder="输入要合成的文本，按 Enter 发送 (Shift+Enter 换行)"></textarea>
  <button class="btn btn-primary" id="btn-send">发送</button>
  <button class="btn btn-danger" id="btn-cancel">取消</button>
  <button class="btn btn-secondary" id="btn-params">参数</button>
</div>

<!-- Upload dialog -->
<div id="upload-dialog">
  <div class="box">
    <h3>新建参考音频</h3>
    <label>名称</label>
    <input type="text" id="up-name" placeholder="e.g. my_voice">
    <label>参考文本 (音频内容)</label>
    <textarea id="up-text" placeholder="音频里说的内容"></textarea>
    <label>上传 WAV 文件</label>
    <input type="file" id="up-file" accept=".wav,audio/wav">
    <label>或录音</label>
    <button class="btn btn-secondary" id="up-record" style="width:100%">开始录音</button>
    <div class="btns">
      <button class="btn btn-secondary" id="up-cancel">取消</button>
      <button class="btn btn-primary" id="up-save">保存</button>
    </div>
  </div>
</div>

<script>
const wsProto = location.protocol === "https:" ? "wss:" : "ws:";
const wsUrl = wsProto + "//" + location.host;
let ws = null;
let currentReqId = null;
let audioCtx = null;
let gainNode = null;
let mediaRecorder = null;
let recordingChunks = [];
let selectedRef = "";
// PCM cache per request: reqId -> { b64chunks: string[], sampleRate, text }
let pcmCache = {};
let pcmCacheOrder = [];

function connect() {
  ws = new WebSocket(wsUrl);
  const bar = document.getElementById("connect-bar");

  ws.onopen = () => {
    bar.className = "connected";
    bar.textContent = "已连接";
    addSystemMsg("WebSocket 已连接");
  };
  ws.onclose = () => {
    bar.className = "error";
    bar.textContent = "连接断开，5秒后重连...";
    addSystemMsg("连接断开", true);
    setTimeout(connect, 5000);
  };
  ws.onerror = () => {
    bar.className = "error";
    bar.textContent = "连接错误";
  };
  ws.onmessage = (e) => {
    const msg = JSON.parse(e.data);
    handleMsg(msg);
  };
}

function handleMsg(msg) {
  switch (msg.type) {
    case "queued":
      addSystemMsg("已排队，位置 #" + msg.position);
      break;
    case "start":
      currentReqId = msg.request_id;
      pcmCache[currentReqId] = { b64chunks: [], sampleRate: 0, text: msg.text || "" };
      pcmCacheOrder.push(currentReqId);
      addSystemMsg("开始合成: " + (msg.text || ""));
      break;
    case "audio": {
      if (msg.request_id && msg.request_id !== currentReqId) break;
      const sr = msg.sample_rate || 24000;
      const float32 = decodePcm16B64(msg.data);
      playFloat32(float32, sr);
      const rid = msg.request_id || currentReqId;
      if (rid && pcmCache[rid]) {
        pcmCache[rid].b64chunks.push(msg.data);
        pcmCache[rid].sampleRate = sr;
      }
      break;
    }
    case "done":
      addDoneWithDownload(msg.request_id || currentReqId);
      if (msg.request_id === currentReqId || !msg.request_id) currentReqId = null;
      break;
    case "cancelled":
      addSystemMsg("已取消", true);
      if (currentReqId && pcmCache[currentReqId]) delete pcmCache[currentReqId];
      currentReqId = null;
      break;
    case "error":
      addSystemMsg("错误: " + msg.message, true);
      if (currentReqId && pcmCache[currentReqId]) delete pcmCache[currentReqId];
      currentReqId = null;
      break;
  }
}

function decodePcm16B64(b64data) {
  const pcm = base64ToBytes(b64data);
  const n = Math.floor(pcm.length / 2);
  const view = new DataView(pcm.buffer, pcm.byteOffset, n * 2);
  const float32 = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    float32[i] = view.getInt16(i * 2, true) / 32768;
  }
  return float32;
}

function playFloat32(float32, sampleRate) {
  if (!audioCtx) {
    audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate });
    gainNode = audioCtx.createGain();
    gainNode.gain.value = 1.0;
    gainNode.connect(audioCtx.destination);
  }
  const buf = audioCtx.createBuffer(1, float32.length, sampleRate);
  buf.getChannelData(0).set(float32);
  const src = audioCtx.createBufferSource();
  src.buffer = buf;
  src.connect(gainNode);
  src.start();
}

function addDoneWithDownload(reqId) {
  const chat = document.getElementById("chat");
  const div = document.createElement("div");
  div.className = "msg system";
  const time = new Date().toLocaleTimeString();
  const c = reqId ? pcmCache[reqId] : null;
  if (c && c.b64chunks.length > 0) {
    const chunks = c.b64chunks.map(b => decodePcm16B64(b));
    const total = chunks.reduce((s, a) => s + a.length, 0);
    const dur = (total / c.sampleRate).toFixed(2);
    const wav = encodeWav(chunks, c.sampleRate);
    const blob = new Blob([wav], { type: "audio/wav" });
    const url = URL.createObjectURL(blob);
    const safeName = (c.text || "tts").substring(0, 30).replace(/[^\w\u4e00-\u9fff]/g, "_") || "tts";
    div.innerHTML = `<div>合成完成 (${dur}s) — <a href="${url}" download="${safeName}.wav">下载 WAV</a></div><div class="meta">${time}</div>`;
    prunePcmCache(10);
  } else {
    div.innerHTML = `<div>合成完成</div><div class="meta">${time}</div>`;
    if (reqId && pcmCache[reqId]) delete pcmCache[reqId];
  }
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
}

function prunePcmCache(maxKeep) {
  while (pcmCacheOrder.length > maxKeep) {
    const old = pcmCacheOrder.shift();
    delete pcmCache[old];
  }
}

function encodeWav(float32Chunks, sampleRate) {
  const total = float32Chunks.reduce((s, a) => s + a.length, 0);
  const buf = new ArrayBuffer(44 + total * 2);
  const view = new DataView(buf);
  // RIFF header
  writeStr(view, 0, "RIFF");
  view.setUint32(4, 36 + total * 2, true);
  writeStr(view, 8, "WAVE");
  // fmt chunk
  writeStr(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);        // PCM
  view.setUint16(22, 1, true);        // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true); // byte rate
  view.setUint16(32, 2, true);        // block align
  view.setUint16(34, 16, true);       // bits per sample
  // data chunk
  writeStr(view, 36, "data");
  view.setUint32(40, total * 2, true);
  let off = 44;
  for (const chunk of float32Chunks) {
    for (let i = 0; i < chunk.length; i++, off += 2) {
      let s = Math.max(-1, Math.min(1, chunk[i]));
      view.setInt16(off, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    }
  }
  return buf;
}

function writeStr(view, off, str) {
  for (let i = 0; i < str.length; i++) view.setUint8(off + i, str.charCodeAt(i));
}

function base64ToBytes(b64) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return bytes;
}

function bytesToBase64(bytes) {
  let bin = "";
  for (let i = 0; i < bytes.length; i++) bin += String.fromCharCode(bytes[i]);
  return btoa(bin);
}

function addSystemMsg(text, isError) {
  const chat = document.getElementById("chat");
  const div = document.createElement("div");
  div.className = "msg system";
  const time = new Date().toLocaleTimeString();
  div.innerHTML = `<div>${escapeHtml(text)}</div><div class="meta">${time}</div>`;
  if (isError) div.querySelector("div").className = "error";
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
}

function addUserMsg(text) {
  const chat = document.getElementById("chat");
  const div = document.createElement("div");
  div.className = "msg user";
  div.textContent = text;
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
}

function escapeHtml(s) {
  return s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
}

function getParams() {
  const instr = document.getElementById("p-instr").value.trim();
  const speaker = document.getElementById("p-speaker").value.trim();
  const p = {
    temperature: parseFloat(document.getElementById("p-temp").value),
    top_p: parseFloat(document.getElementById("p-topp").value),
    top_k: parseInt(document.getElementById("p-topk").value),
    repetition_penalty: parseFloat(document.getElementById("p-rep").value),
    chunk_sec: parseFloat(document.getElementById("p-chunk").value),
    left_context_sec: parseFloat(document.getElementById("p-leftctx").value),
    language_id: parseInt(document.getElementById("p-lang").value),
  };
  if (instr) p.instruction = instr;
  if (speaker) p.speaker = speaker;
  return p;
}

function sendText() {
  const input = document.getElementById("input");
  const text = input.value.trim();
  if (!text || !ws || ws.readyState !== WebSocket.OPEN) return;
  addUserMsg(text);
  const params = getParams();
  if (selectedRef) {
    params.ref_name = selectedRef;
  }
  ws.send(JSON.stringify({ text, params }));
  input.value = "";
}

// ── reference audio management ─────────────────────────────────────────

async function loadRefs() {
  try {
    const resp = await fetch("/api/refs");
    const refs = await resp.json();
    const sel = document.getElementById("ref-select");
    const current = sel.value;
    sel.innerHTML = '<option value="">不使用</option>';
    for (const r of refs) {
      const opt = document.createElement("option");
      opt.value = r.name;
      opt.textContent = r.name + (r.reference_text ? " (" + r.reference_text.substring(0, 30) + ")" : "");
      sel.appendChild(opt);
    }
    if ([...sel.options].some(o => o.value === current)) {
      sel.value = current;
    }
    onRefChange();
  } catch (e) {
    addSystemMsg("加载参考音频列表失败: " + e.message, true);
  }
}

function onRefChange() {
  selectedRef = document.getElementById("ref-select").value;
  document.getElementById("btn-del-ref").style.display = selectedRef ? "" : "none";
}

document.getElementById("ref-select").onchange = onRefChange;
document.getElementById("btn-refresh").onclick = loadRefs;

document.getElementById("btn-del-ref").onclick = async () => {
  if (!selectedRef) return;
  if (!confirm("删除参考音频 '" + selectedRef + "'?")) return;
  try {
    const resp = await fetch("/api/refs/" + encodeURIComponent(selectedRef), { method: "DELETE" });
    if (resp.ok) {
      addSystemMsg("已删除: " + selectedRef);
      selectedRef = "";
      await loadRefs();
    }
  } catch (e) {
    addSystemMsg("删除失败: " + e.message, true);
  }
};

// ── upload dialog ──────────────────────────────────────────────────────

document.getElementById("btn-add-ref").onclick = () => {
  document.getElementById("upload-dialog").classList.add("open");
};

document.getElementById("up-cancel").onclick = () => {
  document.getElementById("upload-dialog").classList.remove("open");
};

document.getElementById("up-save").onclick = async () => {
  const name = document.getElementById("up-name").value.trim();
  const refText = document.getElementById("up-text").value.trim();
  const fileInput = document.getElementById("up-file");
  const file = fileInput.files[0];

  if (!name) { alert("请输入名称"); return; }
  if (!file) { alert("请选择 WAV 文件或录音"); return; }

  const form = new FormData();
  form.append("name", name);
  form.append("reference_text", refText);
  form.append("file", file);

  try {
    const resp = await fetch("/api/refs", { method: "POST", body: form });
    if (resp.ok) {
      addSystemMsg("参考音频已保存: " + name);
      document.getElementById("upload-dialog").classList.remove("open");
      document.getElementById("up-name").value = "";
      document.getElementById("up-text").value = "";
      fileInput.value = "";
      await loadRefs();
      document.getElementById("ref-select").value = name;
      onRefChange();
    } else {
      const err = await resp.json();
      alert("上传失败: " + (err.error || resp.status));
    }
  } catch (e) {
    alert("上传失败: " + e.message);
  }
};

// ── recording in upload dialog ─────────────────────────────────────────

document.getElementById("up-record").onclick = async () => {
  const btn = document.getElementById("up-record");
  if (mediaRecorder && mediaRecorder.state === "recording") {
    mediaRecorder.stop();
    btn.textContent = "开始录音";
    btn.className = "btn btn-secondary";
    return;
  }

  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream, { mimeType: "audio/webm;codecs=opus" });
    recordingChunks = [];

    mediaRecorder.ondataavailable = (e) => {
      if (e.data.size > 0) recordingChunks.push(e.data);
    };

    mediaRecorder.onstop = async () => {
      stream.getTracks().forEach(t => t.stop());
      const blob = new Blob(recordingChunks, { type: "audio/webm;codecs=opus" });
      try {
        const ctx = new (window.AudioContext || window.webkitAudioContext)();
        const ab = await blob.arrayBuffer();
        const decoded = await ctx.decodeAudioData(ab);
        const numChannels = decoded.numberOfChannels;
        const length = decoded.length;
        const sr = decoded.sampleRate;
        const mono = new Float32Array(length);
        for (let i = 0; i < numChannels; i++) {
          const ch = decoded.getChannelData(i);
          for (let s = 0; s < length; s++) mono[s] += ch[s] / numChannels;
        }
        const wav = encodeWavMono(mono, sr);
        const wavFile = new File([wav], "recording.wav", { type: "audio/wav" });
        const dt = new DataTransfer();
        dt.items.add(wavFile);
        document.getElementById("up-file").files = dt.files;
        addSystemMsg("录音完成: " + (wav.byteLength / 1024).toFixed(0) + "KB");
        ctx.close();
      } catch (err) {
        addSystemMsg("录音转码失败: " + err.message, true);
      }
    };

    mediaRecorder.start();
    btn.textContent = "停止录音";
    btn.className = "btn btn-danger";
  } catch (err) {
    addSystemMsg("录音启动失败: " + err.message, true);
  }
};

function encodeWavMono(samples, sampleRate) {
  const len = samples.length;
  const dataLen = len * 2;
  const buf = new ArrayBuffer(44 + dataLen);
  const view = new DataView(buf);
  const writeStr = (off, str) => {
    for (let i = 0; i < str.length; i++) view.setUint8(off + i, str.charCodeAt(i));
  };
  writeStr(0, "RIFF");
  view.setUint32(4, 36 + dataLen, true);
  writeStr(8, "WAVE");
  writeStr(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeStr(36, "data");
  view.setUint32(40, dataLen, true);
  let off = 44;
  for (let i = 0; i < len; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(off, s * 32767, true);
    off += 2;
  }
  return new Uint8Array(buf);
}

// ── event bindings ─────────────────────────────────────────────────────

document.getElementById("btn-send").onclick = sendText;
document.getElementById("btn-cancel").onclick = () => {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: "cancel" }));
  }
};
document.getElementById("btn-params").onclick = () => {
  document.getElementById("params-panel").classList.toggle("open");
};
document.getElementById("input").onkeydown = (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendText();
  }
};

["p-temp","p-topp","p-rep"].forEach(id => {
  const el = document.getElementById(id);
  el.oninput = () => { document.getElementById(id+"-v").textContent = parseFloat(el.value).toFixed(2); };
});

connect();
loadRefs();
</script>
</body>
</html>"""


# ── server ──────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Qwen3-TTS WebSocket streaming server")
    parser.add_argument("model_dir", help="Path to directory containing model GGUF files")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address")
    parser.add_argument("--port", type=int, default=8765, help="Port")
    parser.add_argument("--refs-dir", default="refs", help="Directory for persistent reference audio")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    logger.info("Loading TTS models from %s ...", args.model_dir)
    t0 = time.time()
    tts = Qwen3TTS()
    if not tts.load_models(args.model_dir):
        logger.error("Failed to load models from %s", args.model_dir)
        sys.exit(1)
    logger.info("Models loaded in %.2fs", time.time() - t0)
    logger.info("Capabilities: %s", tts.capabilities)

    atexit.register(tts.close)

    refs_dir = Path(args.refs_dir).resolve()
    refs_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Reference audio directory: %s", refs_dir)

    app = web.Application()
    app["refs_dir"] = refs_dir

    # HTTP API routes
    app.router.add_get("/api/refs", handle_list_refs)
    app.router.add_get("/api/refs/{name}", handle_get_ref)
    app.router.add_post("/api/refs", handle_upload_ref)
    app.router.add_delete("/api/refs/{name}", handle_delete_ref)

    # WebSocket + index page
    async def ws_or_index(request):
        if request.headers.get("upgrade", "").lower() == "websocket":
            ws = web.WebSocketResponse()
            await ws.prepare(request)
            await ws_handler(ws, tts, refs_dir)
            return ws
        return web.Response(
            content_type="text/html",
            charset="utf-8",
            text=INDEX_HTML,
        )

    app.router.add_get("/", ws_or_index)
    app.router.add_get("/ws", ws_or_index)

    logger.info("Starting server on %s:%d", args.host, args.port)
    web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
