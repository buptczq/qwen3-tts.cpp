"""ctypes bindings for libqwen3_tts C API."""

import ctypes
import ctypes.util
import platform
from pathlib import Path
from typing import Callable, Optional, Union
import numpy as np

# ── C type aliases ──────────────────────────────────────────────────────────

_int32 = ctypes.c_int32
_int64 = ctypes.c_int64
_float = ctypes.c_float
_char_ptr = ctypes.c_char_p
_void_ptr = ctypes.c_void_p

# ── Struct definitions (must match qwen3_tts_c.h) ───────────────────────────

class _TtsParams(ctypes.Structure):
    _fields_ = [
        ("max_audio_tokens", _int32),
        ("temperature",      _float),
        ("top_p",            _float),
        ("top_k",            _int32),
        ("n_threads",        _int32),
        ("print_progress",   _int32),
        ("print_timing",     _int32),
        ("repetition_penalty", _float),
        ("language_id",      _int32),
        ("instruction",      _char_ptr),
        ("speaker",          _char_ptr),
    ]


class _TtsResult(ctypes.Structure):
    _fields_ = [
        ("audio",       ctypes.POINTER(_float)),
        ("audio_len",   _int32),
        ("sample_rate", _int32),
        ("success",     _int32),
        ("error_msg",   _char_ptr),
        ("t_total_ms",  _int64),
    ]


class _TtsStreamingParams(ctypes.Structure):
    _fields_ = [
        ("generation",       _TtsParams),
        ("chunk_sec",        _float),
        ("left_context_sec", _float),
        ("collect_audio",    _int32),
    ]


class _TtsModelCapabilities(ctypes.Structure):
    _fields_ = [
        ("loaded",                  _int32),
        ("supports_voice_clone",    _int32),
        ("supports_named_speakers", _int32),
        ("supports_instruction",    _int32),
        ("speaker_embedding_dim",   _int32),
        ("speaker_count",           _int32),
        ("model_kind",              _int32),
    ]


class _VadSegment(ctypes.Structure):
    _fields_ = [
        ("start_ms", _int32),
        ("end_ms",   _int32),
    ]


class _AsrParams(ctypes.Structure):
    _fields_ = [
        ("vad_model_path", _char_ptr),
        ("vad_maxseg",     _int32),
        ("keep_tags",      _int32),
        ("output_ids",     _int32),
        ("n_threads",      _int32),
    ]


class _AsrResult(ctypes.Structure):
    _fields_ = [
        ("text",           _char_ptr),
        ("token_ids",      ctypes.POINTER(_int32)),
        ("token_ids_len",  _int32),
        ("segments",       ctypes.POINTER(_VadSegment)),
        ("segments_len",   _int32),
        ("t_total_ms",     _int64),
        ("success",        _int32),
        ("error_msg",      _char_ptr),
    ]


# Model kind enum
QWEN3_TTS_MODEL_KIND_UNKNOWN = 0
QWEN3_TTS_MODEL_KIND_BASE = 1
QWEN3_TTS_MODEL_KIND_CUSTOM_VOICE = 2
QWEN3_TTS_MODEL_KIND_VOICE_DESIGN = 3

# Callback type for streaming audio chunks.
# Return nonzero to continue, zero to abort.
_AudioChunkCallback = ctypes.CFUNCTYPE(
    _int32,
    ctypes.POINTER(_float),  # samples
    _int32,                   # n_samples
    _int32,                   # sample_rate
    _void_ptr,                # user_data
)

# Progress callback type
_ProgressCallback = ctypes.CFUNCTYPE(
    None,
    _int32,    # tokens_generated
    _int32,    # max_tokens
    _void_ptr, # user_data
)


# ── Library loader ──────────────────────────────────────────────────────────

def _find_lib() -> str:
    """Locate libqwen3_tts.(dylib|so|dll)."""
    # 1. LD_LIBRARY_PATH / DYLD_LIBRARY_PATH / PATH
    name = {"Darwin": "libqwen3_tts.dylib",
            "Linux": "libqwen3_tts.so",
            "Windows": "qwen3_tts.dll"}.get(platform.system(), "libqwen3_tts.so")

    found = ctypes.util.find_library(name)
    if found:
        return found

    # 2. Relative to this file (python/qwen3_tts/../../build/)
    here = Path(__file__).resolve().parent
    for candidate in [
        here.parents[1] / "build" / name,
        here.parents[1] / "build" / ("Release" if platform.system() == "Windows" else "") / name,
    ]:
        if candidate.exists():
            return str(candidate)

    raise OSError(
        f"Cannot find {name}. Set the QWEN3_TTS_LIB_PATH environment variable "
        f"or place the library in a standard location."
    )


# ── Module-level singleton ──────────────────────────────────────────────────

_lib: Optional[ctypes.CDLL] = None


def _get_lib() -> ctypes.CDLL:
    global _lib
    if _lib is not None:
        return _lib
    lib_path = _find_lib()
    _lib = ctypes.CDLL(lib_path)
    _setup_functions(_lib)
    return _lib


def _setup_functions(lib: ctypes.CDLL) -> None:
    """Set argtypes / restype on all C API functions."""

    lib.qwen3_tts_init.restype = _void_ptr
    lib.qwen3_tts_init.argtypes = []

    lib.qwen3_tts_free.restype = None
    lib.qwen3_tts_free.argtypes = [_void_ptr]

    lib.qwen3_tts_load_models.restype = _int32
    lib.qwen3_tts_load_models.argtypes = [_void_ptr, _char_ptr]

    lib.qwen3_tts_load_models_with_name.restype = _int32
    lib.qwen3_tts_load_models_with_name.argtypes = [_void_ptr, _char_ptr, _char_ptr]

    lib.qwen3_tts_synthesize.restype = _TtsResult
    lib.qwen3_tts_synthesize.argtypes = [_void_ptr, _char_ptr, _TtsParams]

    lib.qwen3_tts_synthesize_with_voice.restype = _TtsResult
    lib.qwen3_tts_synthesize_with_voice.argtypes = [_void_ptr, _char_ptr, _char_ptr, _char_ptr, _TtsParams]

    lib.qwen3_tts_synthesize_with_speaker_embedding.restype = _TtsResult
    lib.qwen3_tts_synthesize_with_speaker_embedding.argtypes = [_void_ptr, _char_ptr, _char_ptr, _TtsParams]

    lib.qwen3_tts_synthesize_streaming.restype = _TtsResult
    lib.qwen3_tts_synthesize_streaming.argtypes = [
        _void_ptr, _char_ptr, _TtsStreamingParams,
        _AudioChunkCallback, _void_ptr,
    ]

    lib.qwen3_tts_synthesize_with_voice_streaming.restype = _TtsResult
    lib.qwen3_tts_synthesize_with_voice_streaming.argtypes = [
        _void_ptr, _char_ptr, _char_ptr, _char_ptr, _TtsStreamingParams,
        _AudioChunkCallback, _void_ptr,
    ]

    lib.qwen3_tts_synthesize_with_speaker_embedding_streaming.restype = _TtsResult
    lib.qwen3_tts_synthesize_with_speaker_embedding_streaming.argtypes = [
        _void_ptr, _char_ptr, _char_ptr, _TtsStreamingParams,
        _AudioChunkCallback, _void_ptr,
    ]

    lib.qwen3_tts_extract_speaker_embedding.restype = _int32
    lib.qwen3_tts_extract_speaker_embedding.argtypes = [_void_ptr, _char_ptr, _char_ptr]

    lib.qwen3_tts_get_model_capabilities.restype = _TtsModelCapabilities
    lib.qwen3_tts_get_model_capabilities.argtypes = [_void_ptr]

    lib.qwen3_tts_get_available_speakers.restype = _char_ptr
    lib.qwen3_tts_get_available_speakers.argtypes = [_void_ptr]

    lib.qwen3_tts_free_string.restype = None
    lib.qwen3_tts_free_string.argtypes = [_char_ptr]

    lib.qwen3_tts_free_result.restype = None
    lib.qwen3_tts_free_result.argtypes = [_TtsResult]

    lib.qwen3_tts_set_progress_callback.restype = None
    lib.qwen3_tts_set_progress_callback.argtypes = [_void_ptr, _ProgressCallback, _void_ptr]

    # ASR / VAD
    lib.qwen3_asr_transcribe.restype = _AsrResult
    lib.qwen3_asr_transcribe.argtypes = [_void_ptr, _char_ptr, _char_ptr, _AsrParams]

    lib.qwen3_vad_detect.restype = _int32
    lib.qwen3_vad_detect.argtypes = [
        _void_ptr, _char_ptr, _char_ptr, _int32,
        ctypes.POINTER(ctypes.POINTER(_VadSegment)),
        ctypes.POINTER(_int32),
    ]

    lib.qwen3_vad_free_segments.restype = None
    lib.qwen3_vad_free_segments.argtypes = [ctypes.POINTER(_VadSegment)]

    lib.qwen3_asr_free_result.restype = None
    lib.qwen3_asr_free_result.argtypes = [_AsrResult]

    lib.qwen3_asr_load_model.restype = _int32
    lib.qwen3_asr_load_model.argtypes = [_void_ptr, _char_ptr]

    lib.qwen3_asr_free_model.restype = None
    lib.qwen3_asr_free_model.argtypes = [_void_ptr]

    lib.qwen3_vad_load_model.restype = _int32
    lib.qwen3_vad_load_model.argtypes = [_void_ptr, _char_ptr]

    lib.qwen3_vad_free_model.restype = None
    lib.qwen3_vad_free_model.argtypes = [_void_ptr]

    # Streaming VAD
    lib.qwen3_vad_stream_new.restype = _void_ptr
    lib.qwen3_vad_stream_new.argtypes = [_void_ptr, _int32]

    lib.qwen3_vad_stream_feed.restype = _int32
    lib.qwen3_vad_stream_feed.argtypes = [_void_ptr, ctypes.POINTER(_float), _int32]

    lib.qwen3_vad_stream_get_segment_count.restype = _int32
    lib.qwen3_vad_stream_get_segment_count.argtypes = [_void_ptr]

    lib.qwen3_vad_stream_get_segment.restype = _int32
    lib.qwen3_vad_stream_get_segment.argtypes = [_void_ptr, _int32,
                                                  ctypes.POINTER(_int32), ctypes.POINTER(_int32)]

    lib.qwen3_vad_stream_get_open_segment.restype = _int32
    lib.qwen3_vad_stream_get_open_segment.argtypes = [_void_ptr,
                                                       ctypes.POINTER(_int32), ctypes.POINTER(_int32)]

    lib.qwen3_vad_stream_reset.restype = None
    lib.qwen3_vad_stream_reset.argtypes = [_void_ptr]

    lib.qwen3_vad_stream_free.restype = None
    lib.qwen3_vad_stream_free.argtypes = [_void_ptr]

    # Streaming ASR
    lib.qwen3_asr_transcribe_pcm.restype = _AsrResult
    lib.qwen3_asr_transcribe_pcm.argtypes = [_void_ptr, ctypes.POINTER(_float), _int32, _char_ptr, _AsrParams]

    # ── Session API ──────────────────────────────────────────────────────
    lib.qwen3_tts_session_create.restype = _void_ptr
    lib.qwen3_tts_session_create.argtypes = [_void_ptr]

    lib.qwen3_tts_session_free.restype = None
    lib.qwen3_tts_session_free.argtypes = [_void_ptr]

    lib.qwen3_tts_session_synthesize.restype = _TtsResult
    lib.qwen3_tts_session_synthesize.argtypes = [
        _void_ptr,  # ctx
        _void_ptr,  # session
        _char_ptr,  # text
        _TtsParams,  # params
    ]

    lib.qwen3_tts_session_synthesize_with_voice.restype = _TtsResult
    lib.qwen3_tts_session_synthesize_with_voice.argtypes = [
        _void_ptr,  # ctx
        _void_ptr,  # session
        _char_ptr,  # text
        _char_ptr,  # reference_audio
        _char_ptr,  # reference_text
        _TtsParams,  # params
    ]

    lib.qwen3_tts_session_synthesize_streaming.restype = _TtsResult
    lib.qwen3_tts_session_synthesize_streaming.argtypes = [
        _void_ptr,  # ctx
        _void_ptr,  # session
        _char_ptr,  # text
        _TtsStreamingParams,  # params
        _AudioChunkCallback,  # callback
        _void_ptr,  # user_data
    ]

    lib.qwen3_tts_session_synthesize_with_voice_streaming.restype = _TtsResult
    lib.qwen3_tts_session_synthesize_with_voice_streaming.argtypes = [
        _void_ptr,  # ctx
        _void_ptr,  # session
        _char_ptr,  # text
        _char_ptr,  # reference_audio
        _char_ptr,  # reference_text
        _TtsStreamingParams,  # params
        _AudioChunkCallback,  # callback
        _void_ptr,  # user_data
    ]


# ── Pythonic wrapper ────────────────────────────────────────────────────────

class TTSModelCapabilities:
    """Pythonic view of model capabilities."""
    def __init__(self, caps: _TtsModelCapabilities):
        self.loaded = bool(caps.loaded)
        self.supports_voice_clone = bool(caps.supports_voice_clone)
        self.supports_named_speakers = bool(caps.supports_named_speakers)
        self.supports_instruction = bool(caps.supports_instruction)
        self.speaker_embedding_dim = caps.speaker_embedding_dim
        self.speaker_count = caps.speaker_count
        self.model_kind = caps.model_kind

    def __repr__(self) -> str:
        return (
            f"TTSModelCapabilities(loaded={self.loaded}, "
            f"voice_clone={self.supports_voice_clone}, "
            f"named_speakers={self.supports_named_speakers}, "
            f"instruction={self.supports_instruction}, "
            f"speaker_dim={self.speaker_embedding_dim}, "
            f"speaker_count={self.speaker_count}, "
            f"model_kind={self.model_kind})"
        )


class Qwen3TTSSession:
    """Per-client inference session for concurrent TTS.

    Create from Qwen3TTS.create_session(). Each session has its own
    scheduler, KV caches, and scratch buffers. Multiple sessions can
    share the same model weights safely from different threads.
    """

    def __init__(self, lib, session_ptr):
        self._lib = lib
        self._session = session_ptr

    def close(self):
        if getattr(self, '_session', None):
            self._lib.qwen3_tts_session_free(self._session)
            self._session = None

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class Qwen3TTS:
    """Pythonic wrapper around libqwen3_tts C API.

    Usage
    -----
    >>> tts = Qwen3TTS()
    >>> tts.load_models("path/to/models")
    >>> success, audio, sr = tts.synthesize("Hello world")
    >>> # streaming
    >>> def on_chunk(samples: np.ndarray, sr: int) -> bool:
    ...     play(samples)  # your logic
    ...     return True    # continue
    >>> tts.synthesize_streaming("Hello world", on_chunk)
    """

    def __init__(self, lib_path: Optional[Union[str, Path]] = None):
        if lib_path is not None:
            global _lib
            _lib = ctypes.CDLL(str(lib_path))
            _setup_functions(_lib)
        self._ctx = _get_lib().qwen3_tts_init()
        self._lib = _get_lib()

    # ── context manager ─────────────────────────────────────────────────

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self) -> None:
        if self._ctx is not None:
            self._lib.qwen3_tts_free(self._ctx)
            self._ctx = None

    # ── model loading ───────────────────────────────────────────────────

    def load_models(self, model_dir: Union[str, Path],
                    model_name: str = "") -> bool:
        """Load all TTS models from *model_dir*."""
        d = str(model_dir).encode("utf-8")
        if model_name:
            n = model_name.encode("utf-8")
            return bool(self._lib.qwen3_tts_load_models_with_name(self._ctx, d, n))
        return bool(self._lib.qwen3_tts_load_models(self._ctx, d))

    # ── sessions ────────────────────────────────────────────────────────

    def create_session(self) -> Qwen3TTSSession:
        """Create a new inference session for concurrent use.

        Each session has its own scheduler, KV caches, and scratch buffers.
        Multiple sessions can safely share the same model weights.
        """
        session_ptr = self._lib.qwen3_tts_session_create(self._ctx)
        if not session_ptr:
            raise RuntimeError("Failed to create TTS session")
        return Qwen3TTSSession(self._lib, session_ptr)

    # ── capabilities ────────────────────────────────────────────────────

    @property
    def capabilities(self) -> TTSModelCapabilities:
        caps = self._lib.qwen3_tts_get_model_capabilities(self._ctx)
        return TTSModelCapabilities(caps)

    @property
    def available_speakers(self) -> list[str]:
        raw = self._lib.qwen3_tts_get_available_speakers(self._ctx)
        if not raw:
            return []
        text = raw.decode("utf-8")
        self._lib.qwen3_tts_free_string(raw)
        return [s for s in text.split("\n") if s]

    # ── internal helpers ────────────────────────────────────────────────

    @staticmethod
    def _build_params(
        max_audio_tokens: int = 4096,
        temperature: float = 0.9,
        top_p: float = 1.0,
        top_k: int = 50,
        n_threads: int = 4,
        print_progress: bool = False,
        print_timing: bool = False,
        repetition_penalty: float = 1.05,
        language_id: int = 2050,
        instruction: Optional[str] = None,
        speaker: Optional[str] = None,
    ) -> _TtsParams:
        return _TtsParams(
            max_audio_tokens,
            temperature,
            top_p,
            top_k,
            n_threads,
            int(print_progress),
            int(print_timing),
            repetition_penalty,
            language_id,
            instruction.encode("utf-8") if instruction else None,
            speaker.encode("utf-8") if speaker else None,
        )

    @staticmethod
    def _build_streaming_params(
        chunk_sec: float = 1.0,
        left_context_sec: float = 2.0,
        collect_audio: bool = False,
        **gen_kw,
    ) -> _TtsStreamingParams:
        gen = Qwen3TTS._build_params(**gen_kw)
        return _TtsStreamingParams(gen, chunk_sec, left_context_sec, int(collect_audio))

    def _result_to_numpy(self, res: _TtsResult) -> tuple:
        """Convert C result to (success, audio_array, sample_rate, error_msg, t_total_ms)."""
        success = bool(res.success)
        sr = res.sample_rate
        err = res.error_msg.decode("utf-8") if res.error_msg else ""
        t_ms = res.t_total_ms

        if success and res.audio and res.audio_len > 0:
            buf = (ctypes.c_float * res.audio_len).from_address(
                ctypes.addressof(res.audio.contents)
            )
            audio = np.frombuffer(buf, dtype=np.float32).copy()
        else:
            audio = np.array([], dtype=np.float32)

        self._lib.qwen3_tts_free_result(res)
        return success, audio, sr, err, t_ms

    # ── batch synthesis ─────────────────────────────────────────────────

    def synthesize(self, text: str, **params) -> tuple:
        """Synthesize speech from text.

        Returns
        -------
        (success, audio: np.ndarray[float32], sample_rate: int, error_msg: str, t_total_ms: int)
        """
        p = self._build_params(**params)
        res = self._lib.qwen3_tts_synthesize(self._ctx, text.encode("utf-8"), p)
        return self._result_to_numpy(res)

    def synthesize_with_voice(self, text: str, reference_audio: Union[str, Path],
                              reference_text: Optional[str] = None,
                              **params) -> tuple:
        """Synthesize speech with voice cloning from a reference audio file."""
        p = self._build_params(**params)
        ref = str(reference_audio).encode("utf-8")
        ref_text = reference_text.encode("utf-8") if reference_text else None
        res = self._lib.qwen3_tts_synthesize_with_voice(self._ctx, text.encode("utf-8"), ref, ref_text, p)
        return self._result_to_numpy(res)

    def synthesize_with_speaker_embedding(self, text: str,
                                           speaker_embedding_file: Union[str, Path],
                                           **params) -> tuple:
        """Synthesize speech from a precomputed speaker embedding file."""
        p = self._build_params(**params)
        emb = str(speaker_embedding_file).encode("utf-8")
        res = self._lib.qwen3_tts_synthesize_with_speaker_embedding(
            self._ctx, text.encode("utf-8"), emb, p
        )
        return self._result_to_numpy(res)

    # ── streaming synthesis ─────────────────────────────────────────────

    def synthesize_streaming(
        self,
        text: str,
        on_audio_chunk: Callable[[np.ndarray, int], bool],
        **params,
    ) -> tuple:
        """Streaming synthesis. *on_audio_chunk(samples, sample_rate)* is called
        for each decoded chunk. Return ``True`` to continue or ``False`` to abort.

        Returns the same tuple as ``synthesize()``. If *collect_audio* is True
        the returned audio array contains the full waveform; otherwise it is empty.
        """
        return self._streaming_impl(
            self._lib.qwen3_tts_synthesize_streaming,
            text, on_audio_chunk,
            extra_args=(),
            params=params,
        )

    def synthesize_with_voice_streaming(
        self,
        text: str,
        reference_audio: Union[str, Path],
        on_audio_chunk: Callable[[np.ndarray, int], bool],
        reference_text: Optional[str] = None,
        **params,
    ) -> tuple:
        """Streaming synthesis with voice cloning."""
        return self._streaming_impl(
            self._lib.qwen3_tts_synthesize_with_voice_streaming,
            text, on_audio_chunk,
            extra_args=(str(reference_audio), reference_text),
            params=params,
        )

    def synthesize_with_speaker_embedding_streaming(
        self,
        text: str,
        speaker_embedding_file: Union[str, Path],
        on_audio_chunk: Callable[[np.ndarray, int], bool],
        **params,
    ) -> tuple:
        """Streaming synthesis with a precomputed speaker embedding."""
        return self._streaming_impl(
            self._lib.qwen3_tts_synthesize_with_speaker_embedding_streaming,
            text, on_audio_chunk,
            extra_args=(str(speaker_embedding_file),),
            params=params,
        )

    def _streaming_impl(self, c_api_func, text, on_audio_chunk,
                        extra_args=(), params=None):
        """Common streaming path: set up C callback and call the C API."""
        sp = self._build_streaming_params(**(params or {}))

        abort_flag = [False]

        @_AudioChunkCallback
        def _c_callback(samples_ptr, n_samples, sample_rate, user_data):
            if abort_flag[0]:
                return 0
            if n_samples <= 0:
                return 1
            buf = (ctypes.c_float * n_samples).from_address(
                ctypes.addressof(samples_ptr.contents)
            )
            arr = np.frombuffer(buf, dtype=np.float32).copy()
            try:
                cont = on_audio_chunk(arr, sample_rate)
            except Exception:
                abort_flag[0] = True
                return 0
            if not cont:
                abort_flag[0] = True
                return 0
            return 1

        text_enc = text.encode("utf-8")

        # Build positional args for the C function call
        c_args = [self._ctx, text_enc]
        for a in extra_args:
            if a is None:
                c_args.append(None)
            elif isinstance(a, str):
                c_args.append(a.encode("utf-8"))
            else:
                c_args.append(a)
        c_args.extend([sp, _c_callback, None])

        res = c_api_func(*c_args)
        return self._result_to_numpy(res)

    # ── session-aware streaming synthesis ──────────────────────────────

    def synthesize_streaming_session(
        self,
        session: Qwen3TTSSession,
        text: str,
        on_audio_chunk: Callable[[np.ndarray, int], bool],
        **params,
    ) -> tuple:
        """Session-aware streaming synthesis.

        Same as synthesize_streaming but uses a dedicated session
        for thread-safe concurrent inference.
        """
        return self._streaming_impl_session(
            self._lib.qwen3_tts_session_synthesize_streaming,
            session, text, on_audio_chunk,
            extra_args=(),
            params=params,
        )

    def synthesize_with_voice_streaming_session(
        self,
        session: Qwen3TTSSession,
        text: str,
        reference_audio: Union[str, Path],
        on_audio_chunk: Callable[[np.ndarray, int], bool],
        reference_text: Optional[str] = None,
        **params,
    ) -> tuple:
        """Session-aware streaming synthesis with voice cloning.

        Same as synthesize_with_voice_streaming but uses a dedicated session
        for thread-safe concurrent inference.
        """
        return self._streaming_impl_session(
            self._lib.qwen3_tts_session_synthesize_with_voice_streaming,
            session, text, on_audio_chunk,
            extra_args=(str(reference_audio), reference_text),
            params=params,
        )

    def _streaming_impl_session(self, c_api_func, session, text, on_audio_chunk,
                                extra_args=(), params=None):
        """Common session-aware streaming path: set up C callback and call the C API."""
        sp = self._build_streaming_params(**(params or {}))

        abort_flag = [False]

        @_AudioChunkCallback
        def _c_callback(samples_ptr, n_samples, sample_rate, user_data):
            if abort_flag[0]:
                return 0
            if n_samples <= 0:
                return 1
            buf = (ctypes.c_float * n_samples).from_address(
                ctypes.addressof(samples_ptr.contents)
            )
            arr = np.frombuffer(buf, dtype=np.float32).copy()
            try:
                cont = on_audio_chunk(arr, sample_rate)
            except Exception:
                abort_flag[0] = True
                return 0
            if not cont:
                abort_flag[0] = True
                return 0
            return 1

        text_enc = text.encode("utf-8")

        # Build positional args for the C function call
        # Session-aware functions have (ctx, session, text, ...extras, params, callback, user_data)
        c_args = [self._ctx, session._session, text_enc]
        for a in extra_args:
            if a is None:
                c_args.append(None)
            elif isinstance(a, str):
                c_args.append(a.encode("utf-8"))
            else:
                c_args.append(a)
        c_args.extend([sp, _c_callback, None])

        res = c_api_func(*c_args)
        return self._result_to_numpy(res)

    # ── utility ─────────────────────────────────────────────────────────

    def extract_speaker_embedding(self, reference_audio: Union[str, Path],
                                   output_path: Union[str, Path]) -> bool:
        """Extract speaker embedding from a reference audio file."""
        ref = str(reference_audio).encode("utf-8")
        out = str(output_path).encode("utf-8")
        return bool(self._lib.qwen3_tts_extract_speaker_embedding(self._ctx, ref, out))

    def set_progress_callback(self, callback: Optional[Callable[[int, int], None]]) -> None:
        """Set a progress callback ``fn(tokens_generated, max_tokens)``."""
        if callback is None:
            self._lib.qwen3_tts_set_progress_callback(self._ctx, None, None)
            return

        @_ProgressCallback
        def _cb(tokens, max_tokens, user_data):
            callback(tokens, max_tokens)

        self._lib.qwen3_tts_set_progress_callback(self._ctx, _cb, None)

    # ── ASR / VAD ────────────────────────────────────────────────────────

    def load_asr_model(self, model_gguf: Union[str, Path]) -> bool:
        """Pre-load and cache an ASR model to avoid reloading on every transcribe call.

        Model type is auto-detected from path (path contains "paraformer" -> Paraformer,
        else SenseVoice). Returns True on success.
        """
        return bool(self._lib.qwen3_asr_load_model(
            self._ctx, str(model_gguf).encode("utf-8")))

    def free_asr_model(self) -> None:
        """Free the cached ASR model."""
        self._lib.qwen3_asr_free_model(self._ctx)

    def transcribe(
        self,
        audio_path: Union[str, Path],
        model_gguf: Optional[Union[str, Path]] = None,
        vad_model: Optional[Union[str, Path]] = None,
        vad_maxseg: int = 30000,
        keep_tags: bool = False,
        output_ids: bool = False,
        n_threads: int = 8,
    ) -> dict:
        """Transcribe an audio file using SenseVoice or Paraformer.

        Parameters
        ----------
        audio_path : path to WAV file (any sample rate)
        model_gguf : path to ASR model GGUF, or None to use cached model
                     (must call load_asr_model() first if None)
        vad_model  : optional path to FSMN-VAD GGUF for long audio segmentation
        vad_maxseg : max VAD segment duration in ms
        keep_tags  : keep <|...|> meta tags (SenseVoice only)
        output_ids : return token IDs instead of text
        n_threads  : CPU threads

        Returns
        -------
        dict with keys:
            success (bool), text (str), token_ids (list[int]),
            segments (list[dict(start_ms, end_ms)]),
            t_total_ms (int), error_msg (str)
        """
        p = _AsrParams()
        p.vad_model_path = str(vad_model).encode("utf-8") if vad_model else None
        p.vad_maxseg = vad_maxseg
        p.keep_tags = int(keep_tags)
        p.output_ids = int(output_ids)
        p.n_threads = n_threads

        model_path = str(model_gguf).encode("utf-8") if model_gguf else None
        res = self._lib.qwen3_asr_transcribe(
            self._ctx,
            str(audio_path).encode("utf-8"),
            model_path,
            p,
        )

        result = {
            "success": bool(res.success),
            "text": res.text.decode("utf-8") if res.text else "",
            "token_ids": [],
            "segments": [],
            "t_total_ms": res.t_total_ms,
            "error_msg": res.error_msg.decode("utf-8") if res.error_msg else "",
        }

        if res.token_ids and res.token_ids_len > 0:
            buf = (ctypes.c_int32 * res.token_ids_len).from_address(
                ctypes.addressof(res.token_ids.contents)
            )
            result["token_ids"] = list(buf)

        if res.segments and res.segments_len > 0:
            for i in range(res.segments_len):
                result["segments"].append({
                    "start_ms": res.segments[i].start_ms,
                    "end_ms": res.segments[i].end_ms,
                })

        self._lib.qwen3_asr_free_result(res)
        return result

    def detect_vad(
        self,
        audio_path: Union[str, Path],
        vad_gguf: Union[str, Path],
        maxseg_ms: int = 30000,
    ) -> list[dict]:
        """Detect speech segments using FSMN-VAD.

        Returns
        -------
        list of dicts with keys: start_ms, end_ms
        """
        segs_ptr = ctypes.POINTER(_VadSegment)()
        segs_len = _int32()

        ok = self._lib.qwen3_vad_detect(
            self._ctx,
            str(audio_path).encode("utf-8"),
            str(vad_gguf).encode("utf-8"),
            maxseg_ms,
            ctypes.byref(segs_ptr),
            ctypes.byref(segs_len),
        )

        segments = []
        if ok and segs_len.value > 0 and segs_ptr:
            for i in range(segs_len.value):
                segments.append({
                    "start_ms": segs_ptr[i].start_ms,
                    "end_ms": segs_ptr[i].end_ms,
                })
            self._lib.qwen3_vad_free_segments(segs_ptr)

        return segments

    def load_vad_model(self, vad_gguf: Union[str, Path]) -> bool:
        """Pre-load and cache a VAD model to avoid reloading on every detect call."""
        return bool(self._lib.qwen3_vad_load_model(
            self._ctx, str(vad_gguf).encode("utf-8")))

    def free_vad_model(self) -> None:
        """Free the cached VAD model."""
        self._lib.qwen3_vad_free_model(self._ctx)

    def create_vad_stream(self, max_seg_ms: int = 30000) -> 'VadStream':
        """Create a streaming VAD context. Requires a cached VAD model."""
        return VadStream(self._lib, self._ctx, max_seg_ms)

    def transcribe_pcm(
        self,
        pcm: np.ndarray,
        model_gguf: Optional[Union[str, Path]] = None,
        keep_tags: bool = False,
        output_ids: bool = False,
        n_threads: int = 8,
    ) -> dict:
        """Transcribe PCM samples (f32, 16kHz mono) using cached ASR model.

        Parameters
        ----------
        pcm : numpy array of float32 samples at 16kHz mono
        model_gguf : path to ASR model, or None to use cached model
        keep_tags : keep <|...|> meta tags (SenseVoice only)
        output_ids : return token IDs instead of text
        n_threads : CPU threads

        Returns
        -------
        dict with keys: success, text, token_ids, t_total_ms, error_msg
        """
        pcm = np.ascontiguousarray(pcm, dtype=np.float32)
        p = _AsrParams()
        p.vad_model_path = None
        p.vad_maxseg = 30000
        p.keep_tags = int(keep_tags)
        p.output_ids = int(output_ids)
        p.n_threads = n_threads

        model_path = str(model_gguf).encode("utf-8") if model_gguf else None
        res = self._lib.qwen3_asr_transcribe_pcm(
            self._ctx,
            pcm.ctypes.data_as(ctypes.POINTER(_float)),
            pcm.size,
            model_path,
            p,
        )

        result = {
            "success": bool(res.success),
            "text": res.text.decode("utf-8") if res.text else "",
            "token_ids": [],
            "t_total_ms": res.t_total_ms,
            "error_msg": res.error_msg.decode("utf-8") if res.error_msg else "",
        }

        if res.token_ids and res.token_ids_len > 0:
            buf = (ctypes.c_int32 * res.token_ids_len).from_address(
                ctypes.addressof(res.token_ids.contents)
            )
            result["token_ids"] = list(buf)

        self._lib.qwen3_asr_free_result(res)
        return result


class VadStream:
    """Streaming VAD context. Feed PCM chunks and get speech segments incrementally."""

    def __init__(self, lib, ctx, max_seg_ms: int = 30000):
        self._lib = lib
        self._stream = lib.qwen3_vad_stream_new(ctx, max_seg_ms)
        if not self._stream:
            raise RuntimeError("Failed to create VAD stream (is VAD model loaded?)")

    def __del__(self):
        self.close()

    def close(self):
        if getattr(self, '_stream', None):
            self._lib.qwen3_vad_stream_free(self._stream)
            self._stream = None

    def feed(self, pcm: np.ndarray) -> bool:
        """Feed PCM samples (f32, 16kHz mono). Returns True on success."""
        pcm = np.ascontiguousarray(pcm, dtype=np.float32)
        return bool(self._lib.qwen3_vad_stream_feed(
            self._stream,
            pcm.ctypes.data_as(ctypes.POINTER(_float)),
            pcm.size,
        ))

    def get_segments(self) -> list[dict]:
        """Get all completed speech segments."""
        count = self._lib.qwen3_vad_stream_get_segment_count(self._stream)
        segments = []
        start = _int32()
        end = _int32()
        for i in range(count):
            if self._lib.qwen3_vad_stream_get_segment(
                self._stream, i, ctypes.byref(start), ctypes.byref(end)
            ):
                segments.append({"start_ms": start.value, "end_ms": end.value})
        return segments

    def get_open_segment(self) -> Optional[dict]:
        """Get the current in-progress segment, if any."""
        start = _int32()
        end = _int32()
        if self._lib.qwen3_vad_stream_get_open_segment(
            self._stream, ctypes.byref(start), ctypes.byref(end)
        ):
            return {"start_ms": start.value, "end_ms": end.value}
        return None

    def reset(self) -> None:
        """Reset the stream state (clear all segments)."""
        self._lib.qwen3_vad_stream_reset(self._stream)
