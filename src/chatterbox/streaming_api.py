"""Streaming API for Chatterbox TTS models.

This module exposes a FastAPI application that allows a locally hosted LLM to
open a session, stream text tokens, and receive streamed audio responses. The
API supports both the English-only and multilingual models and provides a
buffering switch that controls whether the server waits for punctuation before
speaking.
"""

from __future__ import annotations

import asyncio
import io
import re
import uuid
from dataclasses import dataclass, field
from threading import Lock
from typing import Dict, Iterable, List, Optional, Tuple, Union

import torch
import torchaudio
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from chatterbox.models.t3.modules.cond_enc import T3Cond

from .mtl_tts import SUPPORTED_LANGUAGES, ChatterboxMultilingualTTS
from .tts import ChatterboxTTS


def _detect_device() -> str:
    """Return the best available torch device for inference."""

    if torch.cuda.is_available():
        return "cuda"

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"

    return "cpu"


DEVICE = _detect_device()


app = FastAPI(title="Chatterbox Streaming API", version="1.0.0")


# ---------------------------------------------------------------------------
# Session configuration and state helpers
# ---------------------------------------------------------------------------


@dataclass
class SessionConfig:
    """Configuration that controls how a session generates speech."""

    multilingual: bool = False
    language_id: Optional[str] = None
    buffer_until_punctuation: bool = True
    audio_prompt_path: Optional[str] = None
    exaggeration: float = 0.5
    temperature: float = 0.8
    cfg_weight: float = 0.5
    repetition_penalty: float = 1.2
    min_p: float = 0.05
    top_p: float = 1.0


@dataclass
class SessionState:
    """Mutable state for an active session."""

    session_id: str
    tts_key: str
    config: SessionConfig
    pending_text: str = ""
    conds: Optional[object] = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


SESSION_REGISTRY: Dict[str, SessionState] = {}
SESSION_REGISTRY_LOCK = Lock()


MODEL_CACHE: Dict[str, Union[ChatterboxTTS, ChatterboxMultilingualTTS]] = {}
MODEL_LOCKS: Dict[str, asyncio.Lock] = {
    "english": asyncio.Lock(),
    "multilingual": asyncio.Lock(),
}
MODEL_LOAD_LOCK = asyncio.Lock()


# Regex used to chunk buffered text at punctuation boundaries. The look-ahead
# ensures that decimal numbers and abbreviations without trailing whitespace
# aren't split prematurely.
PUNCTUATION_PATTERN = re.compile(
    r"(.+?[\.!\?,;:\u3002\uFF01\uFF1F\uFF0C\uFF1B\uFF1A、]+)(?=\s|$|[\"'”’])",
    flags=re.UNICODE,
)


class SessionCreateRequest(BaseModel):
    multilingual: bool = False
    language_id: Optional[str] = None
    audio_prompt_path: Optional[str] = None
    buffer_until_punctuation: bool = True
    exaggeration: float = 0.5
    temperature: float = 0.8
    cfg_weight: float = 0.5
    repetition_penalty: float = 1.2
    min_p: float = 0.05
    top_p: float = 1.0


class SessionCreateResponse(BaseModel):
    session_id: str
    sample_rate: int
    multilingual: bool
    language_id: Optional[str]
    buffer_until_punctuation: bool


class BufferToggleRequest(BaseModel):
    enabled: bool


class BufferToggleResponse(BaseModel):
    session_id: str
    buffer_until_punctuation: bool


class TokenStreamRequest(BaseModel):
    tokens: Union[str, List[str]]
    is_final: bool = False


def _clone_tensor(value: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if value is None:
        return None
    return value.detach().cpu().clone()


def _clone_nested(value):
    if isinstance(value, dict):
        return {k: _clone_nested(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clone_nested(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_clone_nested(v) for v in value)
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    return value


def _clone_conditionals(conds) -> Optional[object]:
    """Deep-copy a Conditionals object onto the CPU."""

    if conds is None:
        return None

    cond_type = type(conds)
    t3: T3Cond = conds.t3
    cloned_t3 = T3Cond(
        speaker_emb=_clone_tensor(getattr(t3, "speaker_emb", None)),
        clap_emb=_clone_tensor(getattr(t3, "clap_emb", None)),
        cond_prompt_speech_tokens=_clone_tensor(getattr(t3, "cond_prompt_speech_tokens", None)),
        cond_prompt_speech_emb=_clone_tensor(getattr(t3, "cond_prompt_speech_emb", None)),
        emotion_adv=_clone_tensor(getattr(t3, "emotion_adv", None)),
    )

    cloned_gen = _clone_nested(conds.gen)

    return cond_type(cloned_t3, cloned_gen)


def _prepare_conditionals_for_model(conds, device: str):
    """Move cloned conditionals onto the specified device."""

    if conds is None:
        return None

    conds_for_device = _clone_conditionals(conds)
    conds_for_device.to(device=device)
    return conds_for_device


async def _load_model(key: str):
    loop = asyncio.get_running_loop()

    def _load():
        if key == "multilingual":
            return ChatterboxMultilingualTTS.from_pretrained(device=DEVICE)
        return ChatterboxTTS.from_pretrained(device=DEVICE)

    return await loop.run_in_executor(None, _load)


async def _get_model(key: str):
    if key in MODEL_CACHE:
        return MODEL_CACHE[key]

    async with MODEL_LOAD_LOCK:
        if key not in MODEL_CACHE:
            MODEL_CACHE[key] = await _load_model(key)
    return MODEL_CACHE[key]


def _register_session(state: SessionState) -> None:
    with SESSION_REGISTRY_LOCK:
        SESSION_REGISTRY[state.session_id] = state


def _get_session(session_id: str) -> SessionState:
    with SESSION_REGISTRY_LOCK:
        session = SESSION_REGISTRY.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    return session


def _remove_session(session_id: str) -> None:
    with SESSION_REGISTRY_LOCK:
        SESSION_REGISTRY.pop(session_id, None)


def _collect_punctuated_segments(buffer: str) -> Tuple[List[str], str]:
    """Split the buffer into fully punctuated segments and leftover text."""

    segments: List[str] = []
    remaining = buffer

    while remaining:
        match = PUNCTUATION_PATTERN.search(remaining)
        if not match:
            break

        segment = match.group(1).strip()
        if segment:
            segments.append(segment)
        remaining = remaining[match.end():].lstrip()

    return segments, remaining


def _chunk_bytes(payload: bytes, chunk_size: int = 4096) -> Iterable[bytes]:
    view = memoryview(payload)
    for start in range(0, len(view), chunk_size):
        yield bytes(view[start : start + chunk_size])


async def _initialize_session_conds(state: SessionState) -> None:
    model = await _get_model(state.tts_key)
    lock = MODEL_LOCKS[state.tts_key]

    async with lock:
        original = _clone_conditionals(model.conds)

        loop = asyncio.get_running_loop()

        def _prepare():
            if state.config.audio_prompt_path:
                model.prepare_conditionals(
                    state.config.audio_prompt_path,
                    exaggeration=state.config.exaggeration,
                )
            return _clone_conditionals(model.conds)

        conds_cpu = await loop.run_in_executor(None, _prepare)

        if original is not None:
            restored = _prepare_conditionals_for_model(original, model.device)
            model.conds = restored
        else:
            model.conds = None

        state.conds = conds_cpu


async def _synthesize_segment(state: SessionState, text: str) -> bytes:
    model = await _get_model(state.tts_key)
    lock = MODEL_LOCKS[state.tts_key]

    async with lock:
        original = _clone_conditionals(model.conds)

        session_conds = state.conds or original
        if session_conds is None:
            raise HTTPException(status_code=500, detail="TTS model conditionals are not initialized")

        working_conds = _prepare_conditionals_for_model(session_conds, model.device)
        model.conds = working_conds

        loop = asyncio.get_running_loop()

        def _generate() -> Tuple[bytes, Optional[object]]:
            if state.config.multilingual:
                wav = model.generate(
                    text,
                    language_id=state.config.language_id,
                    audio_prompt_path=None,
                    exaggeration=state.config.exaggeration,
                    cfg_weight=state.config.cfg_weight,
                    temperature=state.config.temperature,
                    repetition_penalty=state.config.repetition_penalty,
                    min_p=state.config.min_p,
                    top_p=state.config.top_p,
                )
            else:
                wav = model.generate(
                    text,
                    audio_prompt_path=None,
                    exaggeration=state.config.exaggeration,
                    cfg_weight=state.config.cfg_weight,
                    temperature=state.config.temperature,
                    repetition_penalty=state.config.repetition_penalty,
                    min_p=state.config.min_p,
                    top_p=state.config.top_p,
                )

            buffer = io.BytesIO()
            torchaudio.save(buffer, wav.cpu(), model.sr, format="wav")
            payload = buffer.getvalue()
            return payload, _clone_conditionals(model.conds)

        audio_bytes, updated_conds = await loop.run_in_executor(None, _generate)
        state.conds = updated_conds

        if original is not None:
            restored = _prepare_conditionals_for_model(original, model.device)
            model.conds = restored
        else:
            model.conds = None

    return audio_bytes


def _normalize_tokens(tokens: Union[str, List[str]]) -> str:
    if isinstance(tokens, list):
        return "".join(tokens)
    return tokens or ""


def _prepare_segments(state: SessionState, new_text: str, is_final: bool) -> List[str]:
    config = state.config
    segments: List[str] = []

    if config.buffer_until_punctuation:
        state.pending_text += new_text
        punctuated, remainder = _collect_punctuated_segments(state.pending_text)
        segments.extend(segment for segment in punctuated if segment.strip())
        state.pending_text = remainder
        if is_final and state.pending_text.strip():
            segments.append(state.pending_text.strip())
            state.pending_text = ""
    else:
        combined = (state.pending_text + new_text).strip()
        state.pending_text = ""
        if combined:
            segments.append(combined)

    if is_final:
        state.pending_text = ""

    return segments


@app.post("/api/v1/sessions", response_model=SessionCreateResponse)
async def create_session(request: SessionCreateRequest):
    language_id: Optional[str] = None
    tts_key = "multilingual" if request.multilingual else "english"

    if request.multilingual:
        language_id = (request.language_id or "en").lower()
        if language_id not in SUPPORTED_LANGUAGES:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported language_id '{language_id}'.",
            )
    elif request.language_id:
        raise HTTPException(status_code=400, detail="language_id is only valid for multilingual sessions")

    state = SessionState(
        session_id=str(uuid.uuid4()),
        tts_key=tts_key,
        config=SessionConfig(
            multilingual=request.multilingual,
            language_id=language_id,
            buffer_until_punctuation=request.buffer_until_punctuation,
            audio_prompt_path=request.audio_prompt_path,
            exaggeration=request.exaggeration,
            temperature=request.temperature,
            cfg_weight=request.cfg_weight,
            repetition_penalty=request.repetition_penalty,
            min_p=request.min_p,
            top_p=request.top_p,
        ),
    )

    await _initialize_session_conds(state)
    _register_session(state)

    model = await _get_model(tts_key)

    return SessionCreateResponse(
        session_id=state.session_id,
        sample_rate=model.sr,
        multilingual=request.multilingual,
        language_id=language_id,
        buffer_until_punctuation=state.config.buffer_until_punctuation,
    )


@app.post("/api/v1/sessions/{session_id}/buffer", response_model=BufferToggleResponse)
async def toggle_buffer(session_id: str, request: BufferToggleRequest):
    state = _get_session(session_id)
    async with state.lock:
        state.config.buffer_until_punctuation = request.enabled
        # When buffering is disabled, we deliberately leave any pending text
        # untouched so that the next stream request flushes it immediately.
    return BufferToggleResponse(
        session_id=session_id,
        buffer_until_punctuation=state.config.buffer_until_punctuation,
    )


@app.post("/api/v1/sessions/{session_id}/stream")
async def stream_tokens(session_id: str, request: TokenStreamRequest):
    state = _get_session(session_id)

    async with state.lock:
        new_text = _normalize_tokens(request.tokens)
        segments = _prepare_segments(state, new_text, request.is_final)

    if not segments:
        return Response(status_code=204)

    async def audio_generator():
        for segment in segments:
            audio_bytes = await _synthesize_segment(state, segment)
            for chunk in _chunk_bytes(audio_bytes):
                yield chunk

    return StreamingResponse(audio_generator(), media_type="audio/wav")


@app.delete("/api/v1/sessions/{session_id}", status_code=204)
async def close_session(session_id: str):
    _remove_session(session_id)
    return Response(status_code=204)

