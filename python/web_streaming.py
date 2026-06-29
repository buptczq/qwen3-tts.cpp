#!/usr/bin/env python3
"""
WebSocket streaming TTS server.

Protocol
--------
Client -> Server (JSON):
  {"text": "...", "params": {...}}   — enqueue a synthesis request
  {"type": "cancel"}                 — cancel the currently playing request

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
from typing import Optional

import numpy as np

import aiohttp
from aiohttp import web

from qwen3_tts import Qwen3TTS

logger = logging.getLogger("web_streaming")

SAMPLE_RATE = 24000


def pcm16_encode(samples: np.ndarray) -> bytes:
    """Convert float32 [-1, 1] samples to PCM16 bytes."""
    clipped = np.clip(samples, -1.0, 1.0)
    return (clipped * 32767).astype(np.int16).tobytes()


# ── per-client session ──────────────────────────────────────────────────────

@dataclass
class Request:
    """A single TTS synthesis request queued by the client."""
    id: str
    text: str
    params: dict
    cancelled: bool = False


_CHUNK_END = None  # sentinel to signal end of chunk stream


class ClientSession:
    """Manages a queue of synthesis requests for one WebSocket connection."""

    def __init__(self, tts: Qwen3TTS, ws):
        self._tts = tts
        self._ws = ws
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

    # ── worker ──────────────────────────────────────────────────────────

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
                await self._synthesize(req)
                if req.cancelled:
                    await self._send({"type": "cancelled", "request_id": req.id})
                else:
                    await self._send({"type": "done", "request_id": req.id})
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

        # Queue: thread puts np.ndarray chunks, async task gets them
        chunk_queue: asyncio.Queue = asyncio.Queue()

        def _run():
            """Blocking function run in executor thread."""
            def _on_chunk(samples: np.ndarray, sr: int) -> bool:
                if self._cancel_event.is_set():
                    return False
                # Put chunk into the async queue
                fut = asyncio.run_coroutine_threadsafe(
                    chunk_queue.put(samples), loop
                )
                fut.result(timeout=10)
                return True

            try:
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

        # Read chunks from the queue and send them
        while True:
            item = await chunk_queue.get()
            if item is _CHUNK_END:
                break
            if isinstance(item, Exception):
                raise item
            # item is np.ndarray
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

async def handler(ws, tts: Qwen3TTS):
    session = ClientSession(tts, ws)
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
            font-weight: 600; flex-shrink: 0; }
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
  #connect-bar { flex-shrink: 0; padding: 8px 20px; background: #fff3e0;
                 border-bottom: 1px solid #ffe0b2; font-size: 13px; display: none; }
  #connect-bar.error { background: #ffebee; border-color: #ffcdd2; display: block; }
  #connect-bar.connected { background: #e8f5e9; border-color: #c8e6c9; display: block; }
</style>
</head>
<body>
<div id="header">Qwen3-TTS 流式合成</div>
<div id="connect-bar"></div>
<div id="chat"></div>
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

<script>
const wsProto = location.protocol === "https:" ? "wss:" : "ws:";
const wsUrl = wsProto + "//" + location.host;
let ws = null;
let currentReqId = null;
let audioCtx = null;
let gainNode = null;

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
      addSystemMsg("开始合成: " + (msg.text || ""));
      break;
    case "audio":
      if (msg.request_id && msg.request_id !== currentReqId) break;
      playAudio(msg.data, msg.sample_rate || 24000);
      break;
    case "done":
      addSystemMsg("合成完成");
      currentReqId = null;
      break;
    case "cancelled":
      addSystemMsg("已取消", true);
      currentReqId = null;
      break;
    case "error":
      addSystemMsg("错误: " + msg.message, true);
      currentReqId = null;
      break;
  }
}

function playAudio(b64data, sampleRate) {
  const pcm = base64ToBytes(b64data);
  const int16 = new Int16Array(pcm.buffer);
  const float32 = new Float32Array(int16.length);
  for (let i = 0; i < int16.length; i++) {
    float32[i] = int16[i] / 32768;
  }

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

function base64ToBytes(b64) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return bytes;
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
  ws.send(JSON.stringify({ text, params }));
  input.value = "";
}

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
</script>
</body>
</html>"""


# ── server ──────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Qwen3-TTS WebSocket streaming server")
    parser.add_argument("model_dir", help="Path to directory containing model GGUF files")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address")
    parser.add_argument("--port", type=int, default=8765, help="Port")
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

    # Clean up ggml resources before Python's exit() triggers static destructors
    atexit.register(tts.close)

    app = web.Application()

    async def ws_or_index(request):
        if request.headers.get("upgrade", "").lower() == "websocket":
            ws = web.WebSocketResponse()
            await ws.prepare(request)
            await handler(ws, tts)
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
