#!/usr/bin/env python3
"""Quick ASR + LLM Answer: Microphone -> VAD -> ASR -> LLM.

Captures microphone audio via browser, runs streaming VAD to detect speech
segments, performs ASR on each segment, then calls an OpenAI-compatible
chat completion API to answer the recognized text. Results are displayed
on the web page, newest first.

Usage:
    uv run python/web_asr_answer.py \\
        --asr-model /path/to/sensevoice.gguf \\
        --vad-model /path/to/silero-vad.gguf \\
        --port 8767
"""

import argparse
import asyncio
import atexit
import base64
import json
import logging
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np

import aiohttp
from aiohttp import web

from qwen3_tts import Qwen3TTS, VadParams

logger = logging.getLogger("web_asr_answer")

SAMPLE_RATE_MIC = 16000


def pcm16_decode(b64: str) -> np.ndarray:
    raw = base64.b64decode(b64)
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def resample(samples: np.ndarray, from_sr: int, to_sr: int) -> np.ndarray:
    if from_sr == to_sr or len(samples) == 0:
        return samples
    ratio = to_sr / from_sr
    new_len = int(len(samples) * ratio)
    return np.interp(
        np.linspace(0, len(samples) - 1, new_len),
        np.arange(len(samples)),
        samples,
    ).astype(np.float32)


_SHUTDOWN_SENTINEL = object()
_RESET_SENTINEL = object()


def _vad_params_from_dict(d: dict) -> VadParams:
    return VadParams(
        threshold=d.get("threshold") if d.get("threshold") is not None else None,
        min_speech_duration_ms=d.get("min_speech_duration_ms") if d.get("min_speech_duration_ms") is not None else None,
        min_silence_duration_ms=d.get("min_silence_duration_ms") if d.get("min_silence_duration_ms") is not None else None,
        max_speech_duration_s=d.get("max_speech_duration_s") if d.get("max_speech_duration_s") is not None else None,
        speech_pad_ms=d.get("speech_pad_ms") if d.get("speech_pad_ms") is not None else None,
    )


class AsrPipeline:
    """Mic audio -> streaming VAD -> ASR -> LLM answer."""

    def __init__(self, tts: Qwen3TTS, out_queue, loop: asyncio.AbstractEventLoop,
                 vad_params=None, llm_url: str = "", llm_model: str = "",
                 system_prompt: str = "", http_session: aiohttp.ClientSession = None):
        self._tts = tts
        self._out = out_queue
        self._loop = loop
        self._vad_params = vad_params
        self._vad = tts.create_vad_stream(vad_params=vad_params)
        self._audio_chunks: list[np.ndarray] = []
        self._total_samples = 0
        self._seen_segments = 0
        self._seg_counter = 0
        self._q: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()
        self._trimmed_offset = 0
        self._lock = threading.Lock()
        self._processing = False
        self._llm_url = llm_url
        self._llm_model = llm_model
        self._system_prompt = system_prompt
        self._http_session = http_session
        self._context_enabled = False
        self._history: list[dict] = []

    def set_params(self, params: dict) -> None:
        with self._lock:
            vp = params.get("vad_params")
            if vp is not None:
                if isinstance(vp, dict):
                    vp = _vad_params_from_dict(vp)
                self._vad_params = vp
                old = self._vad
                self._vad = self._tts.create_vad_stream(vad_params=vp)
                try:
                    old.close()
                except Exception:
                    pass
                self._audio_chunks.clear()
                self._total_samples = 0
                self._seen_segments = 0
                self._trimmed_offset = 0
            if "system_prompt" in params:
                self._system_prompt = params["system_prompt"]
            if "context_enabled" in params:
                enabled = bool(params["context_enabled"])
                self._context_enabled = enabled
                if not enabled:
                    self._history.clear()

    def _put_out(self, msg: dict) -> None:
        try:
            self._loop.call_soon_threadsafe(self._out.put_nowait, msg)
        except (RuntimeError, asyncio.QueueFull):
            pass

    def feed_audio(self, pcm_16k: np.ndarray) -> None:
        with self._lock:
            if self._processing:
                return

            t_feed_start = time.perf_counter()
            self._audio_chunks.append(pcm_16k)
            self._total_samples += len(pcm_16k)

            t_vad_start = time.perf_counter()
            self._vad.feed(pcm_16k)
            t_vad_ms = (time.perf_counter() - t_vad_start) * 1000

            segments = self._vad.get_segments()
            if len(segments) > self._seen_segments:
                for seg in segments[self._seen_segments:]:
                    self._seg_counter += 1
                    seg_id = self._seg_counter
                    self._q.put({
                        "seg": seg,
                        "seg_id": seg_id,
                        "vad_ms": t_vad_ms,
                    })
                    self._put_out({
                        "type": "vad_segment",
                        "segment_id": seg_id,
                        "start_ms": seg["start_ms"],
                        "end_ms": seg["end_ms"],
                    })
                self._seen_segments = len(segments)

            open_seg = self._vad.get_open_segment()
            self._put_out({
                "type": "vad_open",
                "open": open_seg is not None,
                "start_ms": open_seg["start_ms"] if open_seg else 0,
                "total_audio_ms": self._total_samples / SAMPLE_RATE_MIC * 1000,
            })

    def _run(self):
        while not self._stop.is_set():
            try:
                item = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            if item is _SHUTDOWN_SENTINEL:
                break
            if item is _RESET_SENTINEL:
                self._reset_pipeline()
                continue
            try:
                self._process(item)
            except Exception as exc:
                logger.error("worker error: %s", exc, exc_info=True)

    def _trim_front(self, keep_from_abs: int) -> None:
        if keep_from_abs <= self._trimmed_offset or not self._audio_chunks:
            return
        to_drop = keep_from_abs - self._trimmed_offset
        dropped = 0
        while self._audio_chunks and dropped + len(self._audio_chunks[0]) <= to_drop:
            dropped += len(self._audio_chunks[0])
            self._audio_chunks.pop(0)
        if self._audio_chunks and dropped < to_drop:
            chunk = self._audio_chunks[0]
            n = to_drop - dropped
            self._audio_chunks[0] = chunk[n:]
            dropped += n
        self._trimmed_offset += dropped
        self._total_samples -= dropped
        if self._total_samples < 0:
            self._total_samples = 0

    def _process(self, item):
        seg = item["seg"]
        seg_id = item["seg_id"]
        s_abs = int(seg["start_ms"] * SAMPLE_RATE_MIC / 1000)
        e_abs = int(seg["end_ms"] * SAMPLE_RATE_MIC / 1000)

        with self._lock:
            self._processing = True
            s = s_abs - self._trimmed_offset
            e = e_abs - self._trimmed_offset
            full = np.concatenate(self._audio_chunks) if self._audio_chunks else np.array([], dtype=np.float32)
            if e > len(full):
                e = len(full)
            if s >= e:
                return
            audio = full[s:e]
            seg_dur_ms = len(audio) / SAMPLE_RATE_MIC * 1000

            TRIM_MARGIN = int(10.0 * SAMPLE_RATE_MIC)
            keep_from = max(0, s_abs - TRIM_MARGIN)
            self._trim_front(keep_from)

        try:
            t_asr = time.perf_counter()
            res = self._tts.transcribe_pcm(audio)
            asr_ms = (time.perf_counter() - t_asr) * 1000

            text = res.get("text", "") if res["success"] else ""
            logger.info("[seg#%d] ASR: %.1fms text=%r", seg_id, asr_ms, text)

            self._put_out({
                "type": "asr_result",
                "segment_id": seg_id,
                "text": text,
                "success": res["success"],
                "latency_ms": asr_ms,
                "error": res.get("error_msg", ""),
            })

            if not res["success"] or not text.strip():
                return

            self._put_out({
                "type": "llm_start",
                "segment_id": seg_id,
            })

            asyncio.run_coroutine_threadsafe(
                self._call_llm(seg_id, text), self._loop
            )
        finally:
            self._reset_pipeline()

    async def _call_llm(self, seg_id: int, question: str) -> None:
        if not self._llm_url:
            self._put_out({
                "type": "llm_answer",
                "segment_id": seg_id,
                "text": "(LLM URL not configured)",
                "error": True,
            })
            return

        messages = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})
        if self._context_enabled:
            messages.extend(self._history)
        messages.append({"role": "user", "content": question})

        payload = {
            "model": self._llm_model or "default",
            "messages": messages,
            "stream": True,
        }

        t_llm = time.perf_counter()
        try:
            async with self._http_session.post(
                self._llm_url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    self._put_out({
                        "type": "llm_answer",
                        "segment_id": seg_id,
                        "text": f"HTTP {resp.status}: {body[:200]}",
                        "error": True,
                    })
                    return

                accumulated = ""
                async for line in resp.content:
                    line = line.decode("utf-8").strip()
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        obj = json.loads(data)
                        delta = obj.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            accumulated += content
                            self._put_out({
                                "type": "llm_chunk",
                                "segment_id": seg_id,
                                "text": content,
                            })
                    except json.JSONDecodeError:
                        continue

            llm_ms = (time.perf_counter() - t_llm) * 1000
            self._put_out({
                "type": "llm_done",
                "segment_id": seg_id,
                "latency_ms": llm_ms,
            })
            if self._context_enabled and accumulated.strip():
                self._history.append({"role": "user", "content": question})
                self._history.append({"role": "assistant", "content": accumulated})
        except Exception as exc:
            logger.error("LLM call failed: %s", exc, exc_info=True)
            if accumulated:
                llm_ms = (time.perf_counter() - t_llm) * 1000
                self._put_out({
                    "type": "llm_done",
                    "segment_id": seg_id,
                    "latency_ms": llm_ms,
                })
                if self._context_enabled and accumulated.strip():
                    self._history.append({"role": "user", "content": question})
                    self._history.append({"role": "assistant", "content": accumulated})
            else:
                self._put_out({
                    "type": "llm_answer",
                    "segment_id": seg_id,
                    "text": f"Error: {exc}",
                    "error": True,
                })

    def _reset_pipeline(self) -> None:
        with self._lock:
            self._processing = False
            if self._vad:
                self._vad.reset()
            self._audio_chunks.clear()
            self._total_samples = 0
            self._seen_segments = 0
            self._trimmed_offset = 0
        while not self._q.empty():
            try:
                item = self._q.get_nowait()
                if item is _SHUTDOWN_SENTINEL:
                    self._q.put(item)
            except queue.Empty:
                break

    def request_reset(self) -> None:
        with self._lock:
            self._history.clear()
        self._q.put(_RESET_SENTINEL)

    def close(self) -> None:
        self._stop.set()
        self._q.put(_SHUTDOWN_SENTINEL)
        self._worker.join()
        with self._lock:
            if self._vad:
                self._vad.close()
                self._vad = None


async def ws_handler(request):
    ws = web.WebSocketResponse(max_msg_size=16 * 1024 * 1024)
    await ws.prepare(request)

    tts: Qwen3TTS = request.app["tts"]
    loop = asyncio.get_running_loop()
    out_q: asyncio.Queue = asyncio.Queue(maxsize=256)
    http_session: aiohttp.ClientSession = request.app["http_session"]

    pipeline = AsrPipeline(
        tts, out_q, loop,
        vad_params=request.app.get("vad_params"),
        llm_url=request.app["llm_url"],
        llm_model=request.app["llm_model"],
        system_prompt=request.app["system_prompt"],
        http_session=http_session,
    )

    async def drain():
        while True:
            msg = await out_q.get()
            if msg is None:
                return
            try:
                await ws.send_json(msg)
            except Exception:
                return

    drain_task = asyncio.create_task(drain())

    try:
        async for msg in ws:
            if msg.type != aiohttp.WSMsgType.TEXT:
                continue
            try:
                data = json.loads(msg.data)
            except json.JSONDecodeError:
                await ws.send_json({"type": "error", "message": "invalid JSON"})
                continue

            t = data.get("type")
            if t == "audio_chunk":
                pcm = pcm16_decode(data["data"])
                client_sr = data.get("sample_rate", SAMPLE_RATE_MIC)
                if client_sr != SAMPLE_RATE_MIC:
                    pcm = resample(pcm, client_sr, SAMPLE_RATE_MIC)
                pipeline.feed_audio(pcm)
            elif t == "config":
                pipeline.set_params(data.get("params", {}))
            elif t == "reset":
                pipeline.request_reset()
                await ws.send_json({"type": "reset_ack"})
            else:
                await ws.send_json({"type": "error", "message": f"unknown type: {t}"})
    finally:
        pipeline.close()
        out_q.put_nowait(None)
        drain_task.cancel()
        try:
            await drain_task
        except asyncio.CancelledError:
            pass

    return ws


async def index_handler(request):
    return web.Response(content_type="text/html", charset="utf-8", text=INDEX_HTML)


INDEX_HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ASR + AI Answer</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: #f5f5f5; min-height: 100vh; display: flex; flex-direction: column;
  }
  #header {
    background: #1e40af; color: #fff; padding: 14px 20px;
    font-size: 18px; font-weight: 600;
    display: flex; align-items: center; justify-content: space-between;
  }
  #header .sub { font-size: 12px; font-weight: 400; opacity: .85; }
  #controls {
    padding: 12px 20px; background: #fff; border-bottom: 1px solid #e5e7eb;
    display: flex; gap: 10px; align-items: center; flex-wrap: wrap;
  }
  .btn {
    padding: 8px 18px; border: none; border-radius: 8px;
    font-size: 14px; cursor: pointer; font-weight: 500;
  }
  .btn-primary { background: #1e40af; color: #fff; }
  .btn-primary:hover { background: #1e3a8a; }
  .btn-danger { background: #b91c1c; color: #fff; }
  .btn-danger:hover { background: #991b1b; }
  .btn-secondary { background: #e5e7eb; color: #374151; }
  .btn:disabled { opacity: 0.5; cursor: not-allowed; }
  .badge {
    display: inline-block; padding: 3px 10px; border-radius: 12px;
    font-size: 12px; font-weight: 500;
  }
  .badge-live { background: #fee2e2; color: #991b1b; animation: pulse 1.2s infinite; }
  .badge-idle { background: #e5e7eb; color: #374151; }
  @keyframes pulse { 0%,100% {opacity:1} 50% {opacity:.5} }
  #mic-meter {
    width: 120px; height: 8px; background: #e5e7eb;
    border-radius: 4px; overflow: hidden;
  }
  #mic-meter .fill { height: 100%; background: #10b981; width: 0%; transition: width 60ms; }
  #qa-list {
    flex: 1; padding: 16px 20px; overflow-y: auto;
    display: flex; flex-direction: column; gap: 16px;
  }
  .qa-item {
    background: #fff; border-radius: 12px; padding: 18px 22px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.07);
  }
  .qa-item .question {
    font-size: 20px; font-weight: 600; color: #111827;
    line-height: 1.5; margin-bottom: 10px;
    border-left: 4px solid #1e40af; padding-left: 12px;
  }
  .qa-item .answer {
    font-size: 22px; color: #1e40af; font-weight: 500;
    line-height: 1.5;
  }
  .qa-item .answer.error { color: #b91c1c; font-size: 16px; }
  .qa-item .answer.pending { color: #9ca3af; font-style: italic; font-size: 18px; }
  .qa-item .meta {
    margin-top: 8px; font-size: 12px; color: #9ca3af;
    display: flex; gap: 10px;
  }
  .qa-item .meta span { background: #eef2ff; color: #4338ca; padding: 2px 8px; border-radius: 4px; }
  #prompt-bar {
    padding: 8px 20px; background: #fff; border-top: 1px solid #e5e7eb;
    display: flex; gap: 8px; align-items: center;
  }
  #prompt-bar label { font-size: 13px; color: #6b7280; white-space: nowrap; }
  #prompt-bar input {
    flex: 1; padding: 6px 10px; border: 1px solid #d1d5db;
    border-radius: 6px; font-size: 13px;
  }
</style>
</head>
<body>

<div id="header">
  <span>ASR + AI Answer</span>
  <span class="sub">Mic &rarr; VAD &rarr; ASR &rarr; LLM</span>
</div>

<div id="controls">
  <button class="btn btn-primary" id="btn-start">Start Mic</button>
  <button class="btn btn-danger" id="btn-stop" disabled>Stop</button>
  <button class="btn btn-secondary" id="btn-reset">Reset</button>
  <label style="display:flex;align-items:center;gap:4px;font-size:13px;color:#374151;cursor:pointer;">
    <input type="checkbox" id="ctx-toggle"> Context
  </label>
  <span class="badge badge-idle" id="live-badge">IDLE</span>
  <div id="mic-meter"><div class="fill"></div></div>
</div>

<div id="prompt-bar">
  <label>System Prompt:</label>
  <input type="text" id="sys-prompt" value="直接回答问题，不要有多余的废话。10个字以内。">
</div>

<div id="qa-list"></div>

<script>
const wsProto = location.protocol === "https:" ? "wss:" : "ws:";
const wsUrl = wsProto + "//" + location.host + "/ws";

let ws = null;
let micStream = null;
let captureCtx = null;
let captureSrc = null;
let captureProc = null;
let captureSr = 16000;

const segState = {};

function setBadge(cls, text) {
  const b = document.getElementById("live-badge");
  b.className = "badge " + cls;
  b.textContent = text;
}

function ensureQaItem(segId) {
  let el = document.getElementById(`qa-${segId}`);
  if (!el) {
    el = document.createElement("div");
    el.className = "qa-item";
    el.id = `qa-${segId}`;
    el.innerHTML = `
      <div class="question" id="qa-${segId}-q"></div>
      <div class="answer pending" id="qa-${segId}-a">thinking...</div>
      <div class="meta" id="qa-${segId}-meta"></div>`;
    const list = document.getElementById("qa-list");
    list.insertBefore(el, list.firstChild);
  }
  return el;
}

function handleMsg(m) {
  switch (m.type) {
    case "vad_segment": {
      segState[m.segment_id] = { startWall: Date.now() };
      setBadge("badge-live", "LISTENING...");
      break;
    }
    case "vad_open": {
      if (m.open) {
        setBadge("badge-live", "SPEAKING");
      } else {
        setBadge("badge-idle", "LISTENING");
      }
      break;
    }
    case "asr_result": {
      const el = ensureQaItem(m.segment_id);
      const qEl = el.querySelector(`#qa-${m.segment_id}-q`);
      if (m.success && m.text) {
        qEl.textContent = m.text;
        segState[m.segment_id].asrText = m.text;
      } else {
        qEl.textContent = `[ASR error: ${m.error || "empty"}]`;
        qEl.style.color = "#b91c1c";
        const aEl = el.querySelector(`#qa-${m.segment_id}-a`);
        aEl.style.display = "none";
      }
      const meta = el.querySelector(`#qa-${m.segment_id}-meta`);
      meta.innerHTML = `<span>ASR ${Math.round(m.latency_ms)}ms</span>`;
      setBadge("badge-idle", "LISTENING");
      break;
    }
    case "llm_start": {
      const el = ensureQaItem(m.segment_id);
      const aEl = el.querySelector(`#qa-${m.segment_id}-a`);
      aEl.className = "answer pending";
      aEl.textContent = "thinking...";
      break;
    }
    case "llm_chunk": {
      const el = ensureQaItem(m.segment_id);
      const aEl = el.querySelector(`#qa-${m.segment_id}-a`);
      if (aEl.classList.contains("pending")) {
        aEl.className = "answer";
        aEl.textContent = "";
      }
      aEl.textContent += m.text;
      break;
    }
    case "llm_answer": {
      const el = ensureQaItem(m.segment_id);
      const aEl = el.querySelector(`#qa-${m.segment_id}-a`);
      aEl.className = "answer" + (m.error ? " error" : "");
      aEl.textContent = m.text;
      break;
    }
    case "llm_done": {
      const el = document.getElementById(`qa-${m.segment_id}`);
      if (el) {
        const meta = el.querySelector(`#qa-${m.segment_id}-meta`);
        meta.innerHTML += `<span>LLM ${Math.round(m.latency_ms)}ms</span>`;
      }
      break;
    }
    case "error":
      console.error("ws error:", m.message);
      break;
    case "reset_ack":
      break;
  }
}

function connectWs() {
  ws = new WebSocket(wsUrl);
  ws.onopen = () => { sendConfig(); };
  ws.onclose = () => { setTimeout(connectWs, 3000); };
  ws.onmessage = (e) => handleMsg(JSON.parse(e.data));
}

function sendConfig() {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({
      type: "config",
      params: {
        context_enabled: document.getElementById("ctx-toggle").checked,
        system_prompt: document.getElementById("sys-prompt").value,
        vad_params: {
          threshold: 0.5,
          min_speech_duration_ms: 250,
          min_silence_duration_ms: 100,
          max_speech_duration_s: 30,
          speech_pad_ms: 30,
        },
      },
    }));
  }
}

async function startMic() {
  try {
    micStream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true }
    });
  } catch (e) {
    alert("Microphone access denied");
    return;
  }
  captureCtx = new (window.AudioContext || window.webkitAudioContext)();
  captureSr = captureCtx.sampleRate;
  captureSrc = captureCtx.createMediaStreamSource(micStream);
  captureProc = captureCtx.createScriptProcessor(4096, 1, 1);

  captureProc.onaudioprocess = (ev) => {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    const input = ev.inputBuffer.getChannelData(0);
    let rms = 0;
    for (let i = 0; i < input.length; i++) rms += input[i] * input[i];
    rms = Math.sqrt(rms / input.length);
    const meter = document.getElementById("mic-meter");
    meter.querySelector(".fill").style.width = Math.min(100, rms * 500) + "%";

    let pcm16k;
    if (captureSr === 16000) {
      pcm16k = input;
    } else {
      const ratio = 16000 / captureSr;
      const newLen = Math.floor(input.length * ratio);
      pcm16k = new Float32Array(newLen);
      for (let i = 0; i < newLen; i++) {
        const srcIdx = i / ratio;
        const i0 = Math.floor(srcIdx);
        const i1 = Math.min(i0 + 1, input.length - 1);
        const t = srcIdx - i0;
        pcm16k[i] = input[i0] * (1 - t) + input[i1] * t;
      }
    }

    const pcm16 = new Int16Array(pcm16k.length);
    for (let i = 0; i < pcm16k.length; i++) {
      const s = Math.max(-1, Math.min(1, pcm16k[i]));
      pcm16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
    }
    const bytes = new Uint8Array(pcm16.buffer);
    let bin = "";
    for (let i = 0; i < bytes.length; i++) bin += String.fromCharCode(bytes[i]);
    ws.send(JSON.stringify({
      type: "audio_chunk",
      data: btoa(bin),
      sample_rate: 16000,
    }));
  };
  captureSrc.connect(captureProc);
  captureProc.connect(captureCtx.destination);
  setBadge("badge-idle", "LISTENING");
  sendConfig();
}

function stopMic() {
  if (captureProc) { captureProc.disconnect(); captureProc = null; }
  if (captureSrc) { captureSrc.disconnect(); captureSrc = null; }
  if (captureCtx) { captureCtx.close().catch(()=>{}); captureCtx = null; }
  if (micStream) { micStream.getTracks().forEach(t => t.stop()); micStream = null; }
  setBadge("badge-idle", "IDLE");
}

document.getElementById("btn-start").onclick = async () => {
  document.getElementById("btn-start").disabled = true;
  document.getElementById("btn-stop").disabled = false;
  document.getElementById("qa-list").innerHTML = "";
  for (const k of Object.keys(segState)) delete segState[k];
  await startMic();
};
document.getElementById("btn-stop").onclick = () => {
  stopMic();
  document.getElementById("btn-start").disabled = false;
  document.getElementById("btn-stop").disabled = true;
};
document.getElementById("btn-reset").onclick = () => {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: "reset" }));
  }
  document.getElementById("qa-list").innerHTML = "";
  for (const k of Object.keys(segState)) delete segState[k];
};
document.getElementById("sys-prompt").onchange = sendConfig;
document.getElementById("ctx-toggle").onchange = sendConfig;

connectWs();
</script>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(description="ASR + AI Answer")
    parser.add_argument("--asr-model", required=True,
                        help="Path to ASR GGUF (SenseVoice / Paraformer)")
    parser.add_argument("--vad-model", required=True,
                        help="Path to VAD model (Silero VAD ggml format)")
    parser.add_argument("--tts-model-dir", default=None,
                        help="Directory with TTS GGUF models (needed for Qwen3TTS init)")
    parser.add_argument("--vad-threshold", type=float, default=None)
    parser.add_argument("--vad-min-speech-duration-ms", type=int, default=None)
    parser.add_argument("--vad-min-silence-duration-ms", type=int, default=None)
    parser.add_argument("--vad-max-speech-duration-s", type=float, default=None)
    parser.add_argument("--vad-speech-pad-ms", type=int, default=None)
    parser.add_argument("--llm-url", default="http://127.0.0.1:1234/v1/chat/completions",
                        help="OpenAI-compatible chat completions URL")
    parser.add_argument("--llm-model", default="default",
                        help="Model name for the LLM API")
    parser.add_argument("--system-prompt", default="直接回答问题，不要有多余的废话。10个字以内。",
                        help="System prompt for the LLM")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    logger.info("Loading TTS/ASR/VAD models...")
    tts = Qwen3TTS()
    if args.tts_model_dir:
        if not tts.load_models(args.tts_model_dir):
            logger.error("Failed to load TTS models from %s", args.tts_model_dir)
            sys.exit(1)

    logger.info("Loading ASR model: %s", args.asr_model)
    if not tts.load_asr_model(args.asr_model):
        logger.error("Failed to load ASR model: %s", args.asr_model)
        sys.exit(1)

    logger.info("Loading VAD model: %s", args.vad_model)
    if not tts.load_vad_model(args.vad_model):
        logger.error("Failed to load VAD model: %s", args.vad_model)
        sys.exit(1)

    vad_params = VadParams(
        threshold=args.vad_threshold,
        min_speech_duration_ms=args.vad_min_speech_duration_ms,
        min_silence_duration_ms=args.vad_min_silence_duration_ms,
        max_speech_duration_s=args.vad_max_speech_duration_s,
        speech_pad_ms=args.vad_speech_pad_ms,
    )

    logger.info("Warming up ASR backend...")
    t_warmup = time.perf_counter()
    warmup = np.zeros(SAMPLE_RATE_MIC // 2, dtype=np.float32)
    try:
        tts.transcribe_pcm(warmup)
    except Exception as exc:
        logger.warning("ASR warmup failed: %s", exc)
    logger.info("ASR warmup done in %.1fms", (time.perf_counter() - t_warmup) * 1000)

    atexit.register(lambda: (tts.free_asr_model(), tts.free_vad_model(), tts.close()))

    app = web.Application()
    app["tts"] = tts
    app["vad_params"] = vad_params
    app["llm_url"] = args.llm_url
    app["llm_model"] = args.llm_model
    app["system_prompt"] = args.system_prompt

    async def on_startup(app_):
        app_["http_session"] = aiohttp.ClientSession()

    async def on_cleanup(app_):
        await app_["http_session"].close()

    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)

    app.router.add_get("/", index_handler)
    app.router.add_get("/ws", ws_handler)

    logger.info("Serving on http://%s:%d", args.host, args.port)
    logger.info("LLM endpoint: %s", args.llm_url)
    web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
