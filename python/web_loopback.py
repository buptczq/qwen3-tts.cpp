#!/usr/bin/env python3
"""
Web Loopback Streaming Test: Microphone -> VAD -> ASR -> TTS -> Playback.

Captures microphone audio via browser, runs streaming VAD to detect speech
segments, performs ASR on each segment, and streams TTS synthesis of the
recognized text back to the browser. Displays real-time metrics for each
pipeline stage.

Usage:
    uv run python/web_loopback.py \\
        --tts-model-dir /path/to/tts/models \\
        --asr-model /path/to/sensevoice.gguf \\
        --vad-model /path/to/fsmn-vad.gguf \\
        --port 8766 --refs-dir refs
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
import uuid
from pathlib import Path
from typing import Optional

import numpy as np

import aiohttp
from aiohttp import web

from qwen3_tts import Qwen3TTS

logger = logging.getLogger("web_loopback")

SAMPLE_RATE_TTS = 24000
SAMPLE_RATE_MIC = 16000


# ── helpers ──────────────────────────────────────────────────────────────────


def pcm16_encode(samples: np.ndarray) -> bytes:
    clipped = np.clip(samples, -1.0, 1.0)
    return (clipped * 32767).astype(np.int16).tobytes()


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


# ── loopback pipeline ───────────────────────────────────────────────────────


# Sentinel put on _q to wake the worker for shutdown.
_SHUTDOWN_SENTINEL = object()
# Sentinel put on _q to request a pipeline reset from the event loop.
_RESET_SENTINEL = object()


class LoopbackPipeline:
    """Mic audio -> streaming VAD -> ASR -> streaming TTS.

    Audio chunks are fed from the WebSocket thread via feed_audio().
    VAD detects speech segments. Each completed segment is queued to a
    worker thread that runs ASR then streaming TTS sequentially. Results
    are posted to out_queue as dicts for the WebSocket handler to forward.
    """

    def __init__(self, tts: Qwen3TTS, session, out_queue, refs_dir: Path,
                 loop: asyncio.AbstractEventLoop):
        self._tts = tts
        self._session = session
        self._out = out_queue
        self._refs_dir = refs_dir
        self._loop = loop
        self._vad = tts.create_vad_stream()
        self._audio_chunks: list[np.ndarray] = []
        self._total_samples = 0
        self._seen_segments = 0
        self._seg_counter = 0
        self._q: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._params: dict = {}
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()
        self._trimmed_offset = 0
        # Lock protecting _audio_chunks, _total_samples, _trimmed_offset,
        # _seen_segments, _vad, and _params.
        self._lock = threading.Lock()
        # When True, feed_audio discards incoming audio (set inside _process).
        self._processing = False

    def set_params(self, params: dict) -> None:
        with self._lock:
            self._params = dict(params)

    def _put_out(self, msg: dict) -> None:
        """Thread-safe push to the asyncio out queue (non-blocking, drop on full)."""
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
                        "audio_ready_ms": (time.perf_counter() - t_feed_start) * 1000,
                    })
                    self._put_out({
                        "type": "vad_segment",
                        "segment_id": seg_id,
                        "start_ms": seg["start_ms"],
                        "end_ms": seg["end_ms"],
                        "vad_latency_ms": t_vad_ms,
                    })
                    logger.info(
                        "[seg#%d] VAD detected: start=%dms end=%dms dur=%.2fs "
                        "queue_depth=%d buf_chunks=%d buf=%.1fs vad_ms=%.1f",
                        seg_id, seg["start_ms"], seg["end_ms"],
                        (seg["end_ms"] - seg["start_ms"]) / 1000,
                        self._q.qsize(), len(self._audio_chunks),
                        self._total_samples / SAMPLE_RATE_MIC, t_vad_ms,
                    )
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
        """Trim _audio_chunks so all samples before absolute position keep_from_abs are removed.
        Updates _trimmed_offset to keep absolute-to-relative position conversion working.
        """
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

        # Snapshot params and extract audio under lock, then release lock for ASR+TTS.
        with self._lock:
            params_snapshot = dict(self._params)
            logger.info(
                "[seg#%d] _process start: queue_depth=%d buf_chunks=%d trimmed_offset=%d",
                seg_id, self._q.qsize(), len(self._audio_chunks), self._trimmed_offset,
            )
            self._processing = True

            s = s_abs - self._trimmed_offset
            e = e_abs - self._trimmed_offset
            buf_samples = sum(len(c) for c in self._audio_chunks)
            full = np.concatenate(self._audio_chunks) if self._audio_chunks else np.array([], dtype=np.float32)
            if e > len(full):
                e = len(full)
            if s >= e:
                logger.info("[seg#%d] skip empty slice s_abs=%d e_abs=%d buf=%d rel=[%d..%d]",
                            seg_id, s_abs, e_abs, buf_samples, s, e)
                return
            if e == len(full) and len(full) < (e_abs - self._trimmed_offset):
                logger.warning(
                    "[seg#%d] BUFFER UNDERRUN: expected %d samples, got %d "
                    "(buf=%d trimmed_offset=%d s_abs=%d e_abs=%d)",
                    seg_id, e_abs - s_abs, len(full[s:]),
                    buf_samples, self._trimmed_offset, s_abs, e_abs,
                )
            audio = full[s:e]
            seg_dur_ms = len(audio) / SAMPLE_RATE_MIC * 1000
            rms = float(np.sqrt(np.mean(audio ** 2))) if len(audio) else 0.0

            TRIM_MARGIN = int(10.0 * SAMPLE_RATE_MIC)
            keep_from = max(0, s_abs - TRIM_MARGIN)
            self._trim_front(keep_from)
            buf_samples = sum(len(c) for c in self._audio_chunks)

        logger.info(
            "[seg#%d] ASR slice: s=%d e=%d (%.2fs, %d samples) "
            "buf=%.1fs rms=%.4f vad_feed=%.2fms",
            seg_id, s_abs, e_abs, seg_dur_ms / 1000, len(audio),
            buf_samples / SAMPLE_RATE_MIC, rms, item.get("vad_ms", 0),
        )

        try:
            t_asr = time.perf_counter()
            res = self._tts.transcribe_pcm(audio)
            asr_ms = (time.perf_counter() - t_asr) * 1000

            text = res.get("text", "") if res["success"] else ""
            logger.info(
                "[seg#%d] ASR done: %.1fms (%.2fx realtime) text=%r success=%s err=%r",
                seg_id, asr_ms, asr_ms / seg_dur_ms if seg_dur_ms else 0,
                text, res["success"], res.get("error_msg", ""),
            )
            self._put_out({
                "type": "asr_result",
                "segment_id": seg_id,
                "text": text,
                "success": res["success"],
                "latency_ms": asr_ms,
                "seg_dur_ms": seg_dur_ms,
                "error": res.get("error_msg", ""),
            })

            if not res["success"] or not text.strip():
                return

            ref_name = params_snapshot.get("ref_name")
            ref_path: Optional[str] = None
            ref_text: Optional[str] = None
            if ref_name and self._refs_dir:
                ref_file = self._refs_dir / f"{ref_name}.json"
                if ref_file.exists():
                    try:
                        data = json.loads(ref_file.read_text("utf-8"))
                        wav_bytes = base64.b64decode(data["wav_base64"])
                        tmp = self._refs_dir / f"_{uuid.uuid4().hex}.wav"
                        tmp.write_bytes(wav_bytes)
                        ref_path = str(tmp)
                        ref_text = data.get("reference_text") or None
                    except Exception as exc:
                        logger.warning("ref '%s' load failed: %s", ref_name, exc)

            tts_params = {k: v for k, v in params_snapshot.items()
                          if k not in ("ref_name",)}
            tts_params.setdefault("max_audio_tokens", max(128, len(text) * 3))

            t_tts = time.perf_counter()
            first_chunk_ms = [None]
            total_samples = [0]
            chunk_count = [0]

            def on_chunk(samples: np.ndarray, sr: int) -> bool:
                if first_chunk_ms[0] is None:
                    first_chunk_ms[0] = (time.perf_counter() - t_tts) * 1000
                total_samples[0] += len(samples)
                chunk_count[0] += 1
                pcm = pcm16_encode(samples)
                self._put_out({
                    "type": "tts_chunk",
                    "segment_id": seg_id,
                    "data": base64.b64encode(pcm).decode("ascii"),
                    "sample_rate": sr,
                })
                return True

            logger.info(
                "[seg#%d] TTS start: text=%r ref=%s max_tokens=%d",
                seg_id, text, ref_name or "none", tts_params.get("max_audio_tokens", 0),
            )
            try:
                if ref_path:
                    self._tts.synthesize_with_voice_streaming_session(
                        self._session, text, ref_path,
                        on_audio_chunk=on_chunk,
                        reference_text=ref_text,
                        **tts_params,
                    )
                else:
                    self._tts.synthesize_streaming_session(
                        self._session, text,
                        on_audio_chunk=on_chunk,
                        **tts_params,
                    )
            except Exception as exc:
                logger.exception("TTS failed for segment %d", seg_id)
                self._put_out({
                    "type": "error",
                    "segment_id": seg_id,
                    "message": f"TTS failed: {exc}",
                })
                return
            finally:
                if ref_path:
                    try:
                        Path(ref_path).unlink(missing_ok=True)
                    except Exception:
                        pass

            tts_ms = (time.perf_counter() - t_tts) * 1000
            out_dur_ms = total_samples[0] / SAMPLE_RATE_TTS * 1000 if total_samples[0] else 0
            logger.info(
                "[seg#%d] TTS done: total=%.1fms first_chunk=%.1fms "
                "chunks=%d out=%.2fs (%.2fx realtime)",
                seg_id, tts_ms,
                first_chunk_ms[0] if first_chunk_ms[0] is not None else tts_ms,
                chunk_count[0], out_dur_ms / 1000,
                tts_ms / out_dur_ms if out_dur_ms else 0,
            )
            self._put_out({
                "type": "tts_done",
                "segment_id": seg_id,
                "latency_ms": tts_ms,
                "first_chunk_ms": first_chunk_ms[0] if first_chunk_ms[0] is not None else tts_ms,
                "out_dur_ms": out_dur_ms,
                "tokens": tts_params.get("max_audio_tokens", 0),
            })
        finally:
            self._reset_pipeline()

    def _reset_pipeline(self) -> None:
        """Reset pipeline state: clear audio buffer, reset VAD, drain queue.
        Called from the worker thread.
        """
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
        """Thread-safe reset request — marshalled to the worker via _q."""
        self._q.put(_RESET_SENTINEL)

    def close(self) -> None:
        """Shutdown the worker thread. Blocks until the worker has fully exited.
        After this returns, it is safe to close the TTS session.
        """
        self._stop.set()
        self._q.put(_SHUTDOWN_SENTINEL)
        self._worker.join()
        with self._lock:
            if self._vad:
                self._vad.close()
                self._vad = None


# ── WebSocket handler ───────────────────────────────────────────────────────


async def ws_handler(request):
    ws = web.WebSocketResponse(max_msg_size=16 * 1024 * 1024)
    await ws.prepare(request)

    tts: Qwen3TTS = request.app["tts"]
    refs_dir: Path = request.app["refs_dir"]

    session = tts.create_session()
    loop = asyncio.get_running_loop()
    out_q: asyncio.Queue = asyncio.Queue(maxsize=256)
    pipeline = LoopbackPipeline(tts, session, out_q, refs_dir, loop)

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
        session.close()

    return ws


async def index_handler(request):
    return web.Response(content_type="text/html", charset="utf-8", text=INDEX_HTML)


# ── HTML frontend ───────────────────────────────────────────────────────────


INDEX_HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Loopback Streaming Test</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: #f5f5f5; min-height: 100vh; display: flex; flex-direction: column;
  }
  #header {
    background: #0f766e; color: #fff; padding: 14px 20px;
    font-size: 18px; font-weight: 600;
    display: flex; align-items: center; justify-content: space-between;
  }
  #header .sub { font-size: 12px; font-weight: 400; opacity: .85; }
  #main {
    flex: 1; display: grid; grid-template-columns: 1fr 340px;
    gap: 12px; padding: 12px 16px; min-height: 0;
  }
  #left, #right { display: flex; flex-direction: column; gap: 12px; min-height: 0; }
  .card {
    background: #fff; border-radius: 10px; padding: 14px 16px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
  }
  .card h3 {
    font-size: 13px; font-weight: 600; color: #374151;
    text-transform: uppercase; letter-spacing: .04em; margin-bottom: 10px;
  }
  #controls-row { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
  .btn {
    padding: 8px 16px; border: none; border-radius: 8px;
    font-size: 14px; cursor: pointer; font-weight: 500; white-space: nowrap;
  }
  .btn-primary { background: #0f766e; color: #fff; }
  .btn-primary:hover { background: #115e59; }
  .btn-danger { background: #b91c1c; color: #fff; }
  .btn-danger:hover { background: #991b1b; }
  .btn-secondary { background: #e5e7eb; color: #374151; }
  .btn-secondary:hover { background: #d1d5db; }
  .btn:disabled { opacity: 0.5; cursor: not-allowed; }
  .badge {
    display: inline-block; padding: 3px 8px; border-radius: 12px;
    font-size: 11px; font-weight: 500; background: #e5e7eb; color: #374151;
  }
  .badge-live { background: #fee2e2; color: #991b1b; animation: pulse 1.2s infinite; }
  .badge-idle { background: #e5e7eb; color: #374151; }
  @keyframes pulse { 0%,100% {opacity:1} 50% {opacity:.5} }
  #mic-meter {
    flex: 1; min-width: 120px; height: 8px; background: #e5e7eb;
    border-radius: 4px; overflow: hidden;
  }
  #mic-meter .fill { height: 100%; background: #10b981; width: 0%; transition: width 60ms; }
  #mic-meter.hot .fill { background: #f59e0b; }
  #mic-meter.clip .fill { background: #ef4444; }
  #status-text { font-size: 13px; color: #6b7280; }
  #transcript {
    flex: 1; min-height: 240px; max-height: 520px; overflow-y: auto;
    font-family: -apple-system, sans-serif; font-size: 14px; line-height: 1.6;
    padding: 4px; display: flex; flex-direction: column; gap: 8px;
  }
  .turn {
    padding: 8px 12px; border-radius: 8px; background: #f9fafb;
    border-left: 3px solid #0f766e;
  }
  .turn .asr {
    color: #111827; font-size: 15px; word-break: break-word;
  }
  .turn .asr.pending { color: #9ca3af; font-style: italic; }
  .turn .tts-echo {
    color: #0f766e; font-size: 13px; margin-top: 4px;
    padding-top: 4px; border-top: 1px dashed #e5e7eb;
  }
  .turn .tts-echo.mismatch { color: #b45309; }
  .turn .meta {
    display: flex; gap: 6px; margin-top: 6px; font-size: 11px; color: #6b7280;
    flex-wrap: wrap;
  }
  .turn .meta span { background: #eef2ff; color: #4338ca; padding: 2px 6px; border-radius: 4px; }
  .turn .meta span.err { background: #fee2e2; color: #991b1b; }
  #params-body { display: none; margin-top: 10px; grid-template-columns: 1fr 1fr; gap: 8px 14px; }
  #params-body.open { display: grid; }
  .param { display: flex; align-items: center; gap: 6px; font-size: 13px; }
  .param label { color: #4b5563; min-width: 78px; }
  .param input, .param select {
    flex: 1; padding: 4px 8px; border: 1px solid #d1d5db;
    border-radius: 6px; font-size: 13px; min-width: 0;
  }
  .param input[type="range"] { padding: 0; }
  .param .val { color: #0f766e; font-weight: 500; min-width: 28px; text-align: right; font-variant-numeric: tabular-nums; }
  .param-full { grid-column: 1 / -1; }
  table.metrics { width: 100%; border-collapse: collapse; font-size: 13px; font-variant-numeric: tabular-nums; }
  table.metrics th, table.metrics td {
    padding: 5px 8px; text-align: right; border-bottom: 1px solid #f3f4f6;
  }
  table.metrics th { color: #6b7280; font-weight: 500; font-size: 12px; }
  table.metrics th:first-child, table.metrics td:first-child { text-align: left; }
  table.metrics td.good { color: #059669; }
  table.metrics td.warn { color: #d97706; }
  table.metrics td.bad { color: #dc2626; }
  #events {
    flex: 1; min-height: 180px; max-height: 260px; overflow-y: auto;
    font-family: "SF Mono", Monaco, Consolas, monospace; font-size: 11px;
    color: #4b5563; line-height: 1.5; padding: 4px;
  }
  #events .line { white-space: pre-wrap; word-break: break-all; }
  #events .line.err { color: #b91c1c; }
  #events .line.ok  { color: #047857; }
  #events .line.vad { color: #6d28d9; }
  #footer {
    padding: 8px 16px; font-size: 11px; color: #6b7280; text-align: center;
    border-top: 1px solid #e5e7eb;
  }
  @media (max-width: 860px) {
    #main { grid-template-columns: 1fr; }
  }
</style>
</head>
<body>

<div id="header">
  <span>Loopback Streaming Test</span>
  <span class="sub">Mic &rarr; VAD &rarr; ASR &rarr; TTS &rarr; Playback</span>
</div>

<div id="main">
  <div id="left">
    <div class="card">
      <div id="controls-row">
        <button class="btn btn-primary" id="btn-start">Start Mic</button>
        <button class="btn btn-danger" id="btn-stop" disabled>Stop</button>
        <button class="btn btn-secondary" id="btn-reset">Reset</button>
        <span class="badge badge-idle" id="live-badge">IDLE</span>
        <span id="status-text">click Start, then speak</span>
        <div id="mic-meter"><div class="fill"></div></div>
      </div>
    </div>

    <div class="card" style="flex:1; display:flex; flex-direction:column; min-height:0;">
      <h3>Transcript</h3>
      <div id="transcript"></div>
    </div>

    <div class="card">
      <div style="display:flex; justify-content: space-between; align-items: center;">
        <h3 style="margin:0">TTS Parameters</h3>
        <button class="btn btn-secondary" id="btn-toggle-params" style="padding:4px 10px; font-size:12px">Show</button>
      </div>
      <div id="params-body">
        <div class="param"><label>Temperature</label>
          <input type="range" id="p-temp" min="0" max="2" step="0.1" value="0.7">
          <span class="val" id="p-temp-v">0.7</span></div>
        <div class="param"><label>top-p</label>
          <input type="range" id="p-topp" min="0" max="1" step="0.05" value="0.9">
          <span class="val" id="p-topp-v">0.90</span></div>
        <div class="param"><label>top-k</label>
          <input type="number" id="p-topk" value="50" min="0" max="200"></div>
        <div class="param"><label>Rep penalty</label>
          <input type="range" id="p-rep" min="1" max="2" step="0.05" value="1.05">
          <span class="val" id="p-rep-v">1.05</span></div>
        <div class="param"><label>chunk (s)</label>
          <input type="number" id="p-chunk" value="1.0" min="0.1" max="10" step="0.1"></div>
        <div class="param"><label>left ctx (s)</label>
          <input type="number" id="p-leftctx" value="2.0" min="0" max="10" step="0.5"></div>
        <div class="param"><label>Language</label>
          <select id="p-lang">
            <option value="2050">English (en)</option>
            <option value="2055" selected>中文 (zh)</option>
            <option value="2058">日本語 (ja)</option>
            <option value="2064">한국어 (ko)</option>
            <option value="2069">Русский (ru)</option>
          </select></div>
        <div class="param"><label>Threads</label>
          <input type="number" id="p-threads" value="4" min="1" max="16"></div>
        <div class="param param-full"><label>Reference</label>
          <select id="p-ref"><option value="">none</option></select>
          <button class="btn btn-secondary" id="btn-refresh-refs" style="padding:4px 8px;font-size:12px">Refresh</button>
          </div>
        <div class="param param-full"><label>Instruction</label>
          <input type="text" id="p-instr" placeholder="e.g. 用平静的语气说"></div>
        <div class="param param-full"><label>Speaker</label>
          <input type="text" id="p-speaker" placeholder="speaker name"></div>
      </div>
    </div>
  </div>

  <div id="right">
    <div class="card">
      <h3>Metrics</h3>
      <table class="metrics" id="metrics-table">
        <thead>
          <tr><th>Stage</th><th>Last (ms)</th><th>Avg (ms)</th><th>N</th></tr>
        </thead>
        <tbody>
          <tr><td>VAD feed</td><td id="m-vad-last">&mdash;</td><td id="m-vad-avg">&mdash;</td><td id="m-vad-n">0</td></tr>
          <tr><td>ASR</td><td id="m-asr-last">&mdash;</td><td id="m-asr-avg">&mdash;</td><td id="m-asr-n">0</td></tr>
          <tr><td>TTS first chunk</td><td id="m-ttsfc-last">&mdash;</td><td id="m-ttsfc-avg">&mdash;</td><td id="m-ttsfc-n">0</td></tr>
          <tr><td>TTS total</td><td id="m-tts-last">&mdash;</td><td id="m-tts-avg">&mdash;</td><td id="m-tts-n">0</td></tr>
          <tr><td>E2E (speech&rarr;play)</td><td id="m-e2e-last">&mdash;</td><td id="m-e2e-avg">&mdash;</td><td id="m-e2e-n">0</td></tr>
          <tr><td>RTF (proc/dur)</td><td id="m-rtf-last">&mdash;</td><td id="m-rtf-avg">&mdash;</td><td id="m-rtf-n">0</td></tr>
        </tbody>
      </table>
    </div>

    <div class="card" style="flex:1; display:flex; flex-direction:column; min-height:0;">
      <h3>Event Log</h3>
      <div id="events"></div>
    </div>
  </div>
</div>

<div id="footer">Qwen3-TTS Loopback &middot; speak into mic, see ASR text + hear TTS echo</div>

<script>
const wsProto = location.protocol === "https:" ? "wss:" : "ws:";
const wsUrl = wsProto + "//" + location.host + "/ws";

let ws = null;
let micStream = null;
let captureCtx = null;
let captureSrc = null;
let captureProc = null;
let capturedSamples = 0;       // cumulative samples captured (at capture SR)
let captureStartWall = 0;      // Date.now() when capture started
let captureSr = 16000;         // actual capture sample rate (browser choice)

let playbackCtx = null;
let playbackGain = null;
let nextPlayTime = 0;

// Per-segment state: segId -> { asrText, ttsText, firstChunkWall, startWall, ... }
const segState = {};
let activeSegId = null;        // currently-open VAD segment (from vad_open)

// Metrics accumulators
const metrics = {
  vad:  { last: null, sum: 0, n: 0 },
  asr:  { last: null, sum: 0, n: 0 },
  ttsfc:{ last: null, sum: 0, n: 0 },
  tts:  { last: null, sum: 0, n: 0 },
  e2e:  { last: null, sum: 0, n: 0 },
  rtf:  { last: null, sum: 0, n: 0 },
};

function logEvent(text, cls) {
  const el = document.getElementById("events");
  const line = document.createElement("div");
  line.className = "line" + (cls ? " " + cls : "");
  const ts = new Date().toLocaleTimeString();
  line.textContent = `[${ts}] ${text}`;
  el.appendChild(line);
  el.scrollTop = el.scrollHeight;
}

function updateMetrics() {
  const set = (id, val, fmt) => {
    document.getElementById(id).textContent = val === null ? "—" : fmt(val);
  };
  const fmtInt = v => Math.round(v);
  const fmtF2 = v => v.toFixed(2);
  for (const [k, m] of Object.entries(metrics)) {
    const p = { vad: "vad", asr: "asr", ttsfc: "ttsfc", tts: "tts", e2e: "e2e", rtf: "rtf" }[k];
    set(`m-${p}-last`, m.last, k === "rtf" ? fmtF2 : fmtInt);
    set(`m-${p}-avg`, m.n ? m.sum / m.n : null, k === "rtf" ? fmtF2 : fmtInt);
    document.getElementById(`m-${p}-n`).textContent = m.n;
    // colorize
    const lastCell = document.getElementById(`m-${p}-last`);
    lastCell.classList.remove("good", "warn", "bad");
    if (m.last !== null && k !== "rtf") {
      if (m.last < 300) lastCell.classList.add("good");
      else if (m.last < 800) lastCell.classList.add("warn");
      else lastCell.classList.add("bad");
    }
  }
}

function addMetric(stage, val) {
  metrics[stage].last = val;
  metrics[stage].sum += val;
  metrics[stage].n += 1;
  updateMetrics();
}

function ensureTurn(segId) {
  let turn = document.getElementById(`turn-${segId}`);
  if (!turn) {
    turn = document.createElement("div");
    turn.className = "turn";
    turn.id = `turn-${segId}`;
    turn.innerHTML = `
      <div class="asr pending" id="turn-${segId}-asr">listening&hellip;</div>
      <div class="tts-echo" id="turn-${segId}-tts" style="display:none"></div>
      <div class="meta" id="turn-${segId}-meta"></div>`;
    document.getElementById("transcript").appendChild(turn);
  }
  return turn;
}

function setBadge(cls, text) {
  const b = document.getElementById("live-badge");
  b.className = "badge " + cls;
  b.textContent = text;
}

function setStatus(text) {
  document.getElementById("status-text").textContent = text;
}

// ── WebSocket ────────────────────────────────────────────────────────────

function connectWs() {
  ws = new WebSocket(wsUrl);
  ws.onopen = () => {
    logEvent("connected", "ok");
    setStatus("connected");
    sendConfig();
  };
  ws.onclose = () => {
    logEvent("disconnected — retry in 3s", "err");
    setStatus("disconnected");
    setTimeout(connectWs, 3000);
  };
  ws.onerror = () => logEvent("ws error", "err");
  ws.onmessage = (e) => handleMsg(JSON.parse(e.data));
}

function handleMsg(m) {
  switch (m.type) {
    case "vad_segment": {
      addMetric("vad", m.vad_latency_ms);
      const turn = ensureTurn(m.segment_id);
      segState[m.segment_id] = {
        asrText: "", ttsText: "",
        segStartMs: m.start_ms, segEndMs: m.end_ms,
        segDurMs: m.end_ms - m.start_ms,
        startWall: captureStartWall + m.start_ms,
      };
      turn.querySelector(`#turn-${m.segment_id}-asr`).textContent = `segment #${m.segment_id} (${m.start_ms}-${m.end_ms}ms) — recognizing...`;
      turn.querySelector(`#turn-${m.segment_id}-asr`).classList.add("pending");
      logEvent(`VAD seg#${m.segment_id} ${m.start_ms}-${m.end_ms}ms`, "vad");
      break;
    }
    case "vad_open": {
      if (m.open) {
        setBadge("badge-live", "SPEAKING");
      } else if (Object.keys(segState).length === 0 || !activeSegId) {
        setBadge("badge-idle", "LISTENING");
      }
      break;
    }
    case "asr_result": {
      addMetric("asr", m.latency_ms);
      const turn = ensureTurn(m.segment_id);
      const asrEl = turn.querySelector(`#turn-${m.segment_id}-asr`);
      asrEl.classList.remove("pending");
      if (m.success && m.text) {
        asrEl.textContent = m.text;
        segState[m.segment_id].asrText = m.text;
        logEvent(`ASR seg#${m.segment_id}: "${m.text}" (${Math.round(m.latency_ms)}ms)`, "ok");
      } else {
        asrEl.textContent = `[ASR failed: ${m.error || "empty"}]`;
        asrEl.style.color = "#b91c1c";
        logEvent(`ASR seg#${m.segment_id} FAILED: ${m.error}`, "err");
      }
      const meta = turn.querySelector(`#turn-${m.segment_id}-meta`);
      meta.innerHTML += `<span>ASR ${Math.round(m.latency_ms)}ms</span>` +
                        `<span>seg ${Math.round(m.seg_dur_ms)}ms</span>`;
      break;
    }
    case "tts_chunk": {
      const st = segState[m.segment_id];
      if (!st) break;
      if (st.firstChunkWall === undefined) {
        st.firstChunkWall = Date.now();
        const fc = st.firstChunkWall - (st.startWall || st.firstChunkWall);
        addMetric("ttsfc", fc);
        addMetric("e2e", fc);
      }
      playPcm16B64(m.data, m.sample_rate);
      break;
    }
    case "tts_done": {
      addMetric("tts", m.latency_ms);
      const st = segState[m.segment_id];
      if (st && m.out_dur_ms > 0) {
        const total = m.latency_ms + (metrics.asr.last || 0);
        addMetric("rtf", total / m.out_dur_ms);
      }
      const turn = document.getElementById(`turn-${m.segment_id}`);
      if (turn) {
        const meta = turn.querySelector(`#turn-${m.segment_id}-meta`);
        meta.innerHTML += `<span>TTS fc ${Math.round(m.first_chunk_ms)}ms</span>` +
                          `<span>TTS ${Math.round(m.latency_ms)}ms</span>` +
                          `<span>out ${Math.round(m.out_dur_ms)}ms</span>`;
        const ttsEl = turn.querySelector(`#turn-${m.segment_id}-tts`);
        if (st && st.asrText) {
          ttsEl.textContent = `echo: ${st.asrText}`;
          ttsEl.style.display = "block";
        }
      }
      setBadge("badge-idle", "LISTENING");
      logEvent(`TTS seg#${m.segment_id} done: ${Math.round(m.latency_ms)}ms (fc ${Math.round(m.first_chunk_ms)}ms, out ${Math.round(m.out_dur_ms)}ms)`, "ok");
      break;
    }
    case "error":
      logEvent(`error: ${m.message}`, "err");
      break;
    case "reset_ack":
      logEvent("pipeline reset", "ok");
      break;
  }
}

// ── Mic capture ──────────────────────────────────────────────────────────

async function startMic() {
  try {
    micStream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true }
    });
  } catch (e) {
    logEvent("mic denied: " + e.message, "err");
    return;
  }
  // Use default sample rate; downsample to 16kHz in the processor
  captureCtx = new (window.AudioContext || window.webkitAudioContext)();
  captureSr = captureCtx.sampleRate;
  captureSrc = captureCtx.createMediaStreamSource(micStream);
  // ScriptProcessor is deprecated but widely supported. bufferSize=4096 ~ 85ms @48kHz.
  captureProc = captureCtx.createScriptProcessor(4096, 1, 1);
  capturedSamples = 0;
  captureStartWall = Date.now();

  captureProc.onaudioprocess = (ev) => {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    const input = ev.inputBuffer.getChannelData(0);
    // meter
    let rms = 0;
    for (let i = 0; i < input.length; i++) rms += input[i] * input[i];
    rms = Math.sqrt(rms / input.length);
    const meter = document.getElementById("mic-meter");
    const pct = Math.min(100, rms * 500);
    meter.querySelector(".fill").style.width = pct + "%";
    meter.classList.remove("hot", "clip");
    if (rms > 0.3) meter.classList.add("clip");
    else if (rms > 0.15) meter.classList.add("hot");

    // downsample to 16kHz
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
    capturedSamples += pcm16k.length;

    // float32 -> int16 PCM -> base64
    const pcm16 = new Int16Array(pcm16k.length);
    for (let i = 0; i < pcm16k.length; i++) {
      const s = Math.max(-1, Math.min(1, pcm16k[i]));
      pcm16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
    }
    const bytes = new Uint8Array(pcm16.buffer);
    let bin = "";
    for (let i = 0; i < bytes.length; i++) bin += String.fromCharCode(bytes[i]);
    const b64 = btoa(bin);
    ws.send(JSON.stringify({
      type: "audio_chunk",
      data: b64,
      sample_rate: 16000,
    }));
  };
  captureSrc.connect(captureProc);
  captureProc.connect(captureCtx.destination);

  setBadge("badge-idle", "LISTENING");
  setStatus("capturing @ " + captureSr + "Hz");
  logEvent("mic started @ " + captureSr + "Hz", "ok");
}

function stopMic() {
  if (captureProc) { captureProc.disconnect(); captureProc = null; }
  if (captureSrc) { captureSrc.disconnect(); captureSrc = null; }
  if (captureCtx) { captureCtx.close().catch(()=>{}); captureCtx = null; }
  if (micStream) {
    micStream.getTracks().forEach(t => t.stop());
    micStream = null;
  }
  setBadge("badge-idle", "IDLE");
  setStatus("stopped");
  logEvent("mic stopped");
}

// ── TTS playback ─────────────────────────────────────────────────────────

function ensurePlaybackCtx() {
  if (!playbackCtx) {
    playbackCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 24000 });
    playbackGain = playbackCtx.createGain();
    playbackGain.gain.value = 1.0;
    playbackGain.connect(playbackCtx.destination);
  }
  if (playbackCtx.state === "suspended") playbackCtx.resume();
}

function playPcm16B64(b64, sampleRate) {
  ensurePlaybackCtx();
  const bin = atob(b64);
  const n = Math.floor(bin.length / 2);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  const view = new DataView(bytes.buffer, 0, n * 2);
  const float32 = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    float32[i] = view.getInt16(i * 2, true) / 32768;
  }
  const buf = playbackCtx.createBuffer(1, float32.length, sampleRate);
  buf.getChannelData(0).set(float32);
  const src = playbackCtx.createBufferSource();
  src.buffer = buf;
  src.connect(playbackGain);
  const now = playbackCtx.currentTime;
  const when = Math.max(now, nextPlayTime);
  src.start(when);
  nextPlayTime = when + float32.length / sampleRate;
}

// ── Config / refs ────────────────────────────────────────────────────────

function buildParams() {
  const p = {
    temperature: parseFloat(document.getElementById("p-temp").value),
    top_p: parseFloat(document.getElementById("p-topp").value),
    top_k: parseInt(document.getElementById("p-topk").value),
    repetition_penalty: parseFloat(document.getElementById("p-rep").value),
    chunk_sec: parseFloat(document.getElementById("p-chunk").value),
    left_context_sec: parseFloat(document.getElementById("p-leftctx").value),
    language_id: parseInt(document.getElementById("p-lang").value),
    n_threads: parseInt(document.getElementById("p-threads").value),
  };
  const instr = document.getElementById("p-instr").value.trim();
  const speaker = document.getElementById("p-speaker").value.trim();
  const ref = document.getElementById("p-ref").value;
  if (instr) p.instruction = instr;
  if (speaker) p.speaker = speaker;
  if (ref) p.ref_name = ref;
  return p;
}

function sendConfig() {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: "config", params: buildParams() }));
  }
}

async function loadRefs() {
  try {
    const resp = await fetch("/api/refs");
    const refs = await resp.json();
    const sel = document.getElementById("p-ref");
    const cur = sel.value;
    sel.innerHTML = '<option value="">none</option>';
    for (const r of refs) {
      const o = document.createElement("option");
      o.value = r.name;
      o.textContent = r.name + (r.reference_text ? " (" + r.reference_text.substring(0, 30) + ")" : "");
      sel.appendChild(o);
    }
    if ([...sel.options].some(o => o.value === cur)) sel.value = cur;
  } catch (e) {
    // refs API may not be available; ignore
  }
}

// ── Wire up ──────────────────────────────────────────────────────────────

document.getElementById("btn-start").onclick = async () => {
  document.getElementById("btn-start").disabled = true;
  document.getElementById("btn-stop").disabled = false;
  document.getElementById("transcript").innerHTML = "";
  for (const k of Object.keys(segState)) delete segState[k];
  for (const k of Object.keys(metrics)) {
    metrics[k].last = null; metrics[k].sum = 0; metrics[k].n = 0;
  }
  updateMetrics();
  await startMic();
  sendConfig();
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
  document.getElementById("transcript").innerHTML = "";
  for (const k of Object.keys(segState)) delete segState[k];
};
document.getElementById("btn-toggle-params").onclick = () => {
  const body = document.getElementById("params-body");
  const btn = document.getElementById("btn-toggle-params");
  body.classList.toggle("open");
  btn.textContent = body.classList.contains("open") ? "Hide" : "Show";
};
document.getElementById("btn-refresh-refs").onclick = loadRefs;

["p-temp","p-topp","p-rep"].forEach(id => {
  const el = document.getElementById(id);
  el.oninput = () => {
    document.getElementById(id+"-v").textContent = parseFloat(el.value).toFixed(2);
    sendConfig();
  };
});
["p-topk","p-chunk","p-leftctx","p-lang","p-threads","p-instr","p-speaker","p-ref"].forEach(id => {
  const el = document.getElementById(id);
  el.onchange = sendConfig;
  el.oninput = sendConfig;
});

connectWs();
loadRefs();
</script>
</body>
</html>"""


# ── server main ───────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Web Loopback Streaming Test")
    parser.add_argument("--tts-model-dir", required=True,
                        help="Directory with TTS GGUF models")
    parser.add_argument("--asr-model", required=True,
                        help="Path to ASR GGUF (SenseVoice / Paraformer)")
    parser.add_argument("--vad-model", required=True,
                        help="Path to VAD GGUF (FSMN-VAD)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--refs-dir", default="refs",
                        help="Directory with reference audio (optional voice clone)")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    logger.info("Loading TTS models from %s ...", args.tts_model_dir)
    tts = Qwen3TTS()
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

    logger.info("Warming up ASR Metal backend...")
    t_warmup = time.perf_counter()
    warmup = np.zeros(SAMPLE_RATE_MIC // 2, dtype=np.float32)  # 0.5s silence
    try:
        tts.transcribe_pcm(warmup)
    except Exception as exc:
        logger.warning("ASR warmup failed: %s", exc)
    logger.info("ASR warmup done in %.1fms", (time.perf_counter() - t_warmup) * 1000)

    atexit.register(lambda: (tts.free_asr_model(), tts.free_vad_model(), tts.close()))

    refs_dir = Path(args.refs_dir).resolve()
    refs_dir.mkdir(parents=True, exist_ok=True)

    app = web.Application()
    app["tts"] = tts
    app["refs_dir"] = refs_dir

    async def list_refs(request):
        refs = []
        for f in sorted(refs_dir.iterdir()):
            if f.suffix != ".json":
                continue
            try:
                data = json.loads(f.read_text("utf-8"))
                refs.append({
                    "name": data.get("name", f.stem),
                    "reference_text": data.get("reference_text", ""),
                })
            except Exception:
                pass
        return web.json_response(refs)

    app.router.add_get("/", index_handler)
    app.router.add_get("/ws", ws_handler)
    app.router.add_get("/api/refs", list_refs)

    logger.info("Serving on http://%s:%d", args.host, args.port)
    web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
