"""Murf TTS plugin for Hermes.

Implements :class:`agent.tts_provider.TTSProvider` (from the Hermes
checkout, ``agent/tts_provider.py``) for `Murf <https://murf.ai>`_,
using the Falcon streaming model over Murf's WebSocket API for
low-latency synthesis, with a batch ``synthesize()`` fallback built on
top of the same stream.

Murf API reference:

- https://murf.ai/api/docs/api-reference/text-to-speech/stream-input
- https://murf.ai/api/docs/text-to-speech/web-sockets
- https://murf.ai/api/docs/text-to-speech/web-sockets/context-id
- https://murf.ai/api/docs/text-to-speech-models/falcon-2
- https://murf.ai/api/docs/voices-styles/overview

Behavior notes:

1. **Auth.** The API key is sent as the ``api_key`` query parameter on
   the WebSocket handshake (see ``_QUERY_KEY_API_KEY`` below).
2. **``model``.** ``falcon-2`` (default) or ``gen2``.
3. **``voice_config`` fields.** ``voiceId`` is always sent. ``locale``,
   ``style``, ``rate``, ``pitch``, and ``variation`` are sent only when
   explicitly passed via ``**extra`` — never as defaults.
4. **REST voices endpoint.** ``list_voices()`` calls
   ``GET /v1/speech/voices`` at ``https://api.murf.ai`` (override via
   ``MURF_API_BASE_URL``) and fails soft — it returns ``[]`` and logs a
   warning on error, so it never blocks the streaming/synthesis path.

:meth:`TTSProvider.stream` is a synchronous generator per the ABC, so
this module uses ``websockets.sync.client`` (requires ``websockets>=12``)
rather than an async websocket client.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import uuid
from typing import Any, Dict, Iterator, List, Optional
from urllib.parse import urlencode

from agent.tts_provider import DEFAULT_OUTPUT_FORMAT, TTSProvider

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WS_HOST = "global.api.murf.ai"
WS_URL = f"wss://{WS_HOST}/v1/speech/stream-input"

REST_BASE_URL = os.environ.get("MURF_API_BASE_URL", "https://api.murf.ai")
VOICES_ENDPOINT = f"{REST_BASE_URL}/v1/speech/voices"

DEFAULT_MODEL = "falcon-2"
DEFAULT_SAMPLE_RATE = 48000
DEFAULT_CHANNEL_TYPE = "MONO"

# Query param name used for API-key auth on the WS handshake.
_QUERY_KEY_API_KEY = "api_key"

# Murf's documented WS `format` enum. `opus` (the ABC's stream()
# default) is not one of these — see _normalize_format().
_MURF_FORMATS = {"MP3", "FLAC", "WAV", "ALAW", "ULAW", "OGG", "PCM"}


def _normalize_format(fmt: Optional[str]) -> str:
    """Map an ABC-level format hint to a Murf WS `format` enum value.

    Murf doesn't support Opus directly. Hermes's own voice-bubble
    pipeline already re-encodes to Opus when `voice_compatible` is
    True, so requesting WAV from Murf and letting Hermes transcode is
    the closest equivalent rather than a lossy guess.
    """
    if not fmt:
        return "WAV"
    upper = fmt.strip().upper()
    if upper in _MURF_FORMATS:
        return upper
    return "WAV"


def _build_ws_url(api_key: str, *, model: str, sample_rate: int, fmt: str) -> str:
    params = {
        _QUERY_KEY_API_KEY: api_key,
        "model": model,
        "sample_rate": str(sample_rate),
        "channel_type": DEFAULT_CHANNEL_TYPE,
        "format": fmt,
    }
    return f"{WS_URL}?{urlencode(params)}"


class MurfTTSProvider(TTSProvider):
    """Murf Falcon streaming TTS backend."""

    @property
    def name(self) -> str:
        return "murf"

    @property
    def display_name(self) -> str:
        return "Murf"

    @property
    def voice_compatible(self) -> bool:
        return True

    def is_available(self) -> bool:
        if not os.environ.get("MURF_API_KEY", "").strip():
            return False
        try:
            import importlib.util

            return importlib.util.find_spec("websockets") is not None
        except Exception:
            return False

    # -- voices / models ----------------------------------------------------

    def list_voices(self) -> List[Dict[str, Any]]:
        # Falcon-2 and Gen2 each have their own voice catalog on Murf's
        # side (confirmed via a live "Invalid voice_id" error, which
        # names the exact query param: GET /v1/speech/voices?model=FALCON).
        # Without it, this endpoint silently returns some other/default
        # catalog that omits voices real for the model actually in use —
        # confirmed missing at least two real, working Falcon voices
        # (en-IN-palak, en-IN-anisha) when this filter was absent.
        model_env = os.environ.get("MURF_TTS_MODEL", "").strip() or DEFAULT_MODEL
        model_query = "GEN2" if model_env.lower() == "gen2" else "FALCON"

        api_key = os.environ.get("MURF_API_KEY", "").strip()
        if not api_key:
            return []
        try:
            import urllib.request
            from urllib.parse import urlencode as _urlencode

            req = urllib.request.Request(
                f"{VOICES_ENDPOINT}?{_urlencode({'model': model_query})}",
                headers={"api-key": api_key, "Accept": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001 - list_voices() must fail soft
            logger.warning("murf: failed to list voices (%s): %s", VOICES_ENDPOINT, exc)
            return []

        if isinstance(data, list):
            raw_voices = data
        elif isinstance(data, dict):
            raw_voices = data.get("voices") or data.get("data") or []
        else:
            raw_voices = []

        voices: List[Dict[str, Any]] = []
        for v in raw_voices:
            if not isinstance(v, dict):
                continue
            voice_id = v.get("voiceId") or v.get("voice_id") or v.get("id")
            if not voice_id:
                continue
            voices.append(
                {
                    "id": voice_id,
                    "display": v.get("displayName") or v.get("display_name") or voice_id,
                    "language": v.get("locale") or v.get("language"),
                    "gender": (v.get("gender") or "").lower() or None,
                    "preview_url": v.get("previewUrl") or v.get("preview_url"),
                }
            )
        return voices

    def list_models(self) -> List[Dict[str, Any]]:
        return [
            {"id": "falcon-2", "display": "Falcon 2 (ultra-low latency, under 100ms time-to-first-audio)"},
            {"id": "gen2", "display": "Gen2 (higher quality, standard latency)"},
        ]

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": self.display_name,
            "badge": "paid",
            "tag": "Ultra-low-latency Falcon streaming TTS",
            "env_vars": [
                {
                    "key": "MURF_API_KEY",
                    "prompt": "Murf API key",
                    "url": "https://murf.ai/api",
                },
            ],
        }

    # -- synthesis ------------------------------------------------------

    def _require_api_key(self) -> str:
        api_key = os.environ.get("MURF_API_KEY", "").strip()
        if not api_key:
            raise ValueError(
                "MURF_API_KEY environment variable not set. "
                "Get your API key at https://murf.ai/api"
            )
        return api_key

    def _resolve_voice(self, voice: Optional[str], extra: Dict[str, Any]) -> str:
        voice_id = voice or extra.get("voice_id") or os.environ.get("MURF_DEFAULT_VOICE", "").strip()
        if not voice_id:
            raise ValueError(
                "No Murf voice specified. Pass voice=<voiceId>, or set "
                "MURF_DEFAULT_VOICE, or call list_voices() to pick one."
            )
        return voice_id

    def stream(
        self,
        text: str,
        *,
        voice: Optional[str] = None,
        model: Optional[str] = None,
        format: str = "opus",
        **extra: Any,
    ) -> Iterator[bytes]:
        try:
            from websockets.sync.client import connect
        except ImportError as exc:
            raise ImportError(
                "The murf plugin requires the 'websockets' package "
                "(>=12). Install it with: pip install 'websockets>=12'"
            ) from exc

        api_key = self._require_api_key()
        voice_id = self._resolve_voice(voice, extra)
        ws_model = model or os.environ.get("MURF_TTS_MODEL", "").strip() or DEFAULT_MODEL
        sample_rate = int(extra.get("sample_rate") or DEFAULT_SAMPLE_RATE)
        murf_format = _normalize_format(extra.get("murf_format") or format)
        context_id = extra.get("context_id") or uuid.uuid4().hex

        url = _build_ws_url(api_key, model=ws_model, sample_rate=sample_rate, fmt=murf_format)

        voice_config: Dict[str, Any] = {"voiceId": voice_id}
        for key in ("locale", "style", "rate", "pitch", "variation"):
            if extra.get(key) is not None:
                voice_config[key] = extra[key]

        logger.debug("murf: connecting (model=%s, voice=%s, sample_rate=%s)", ws_model, voice_id, sample_rate)

        with connect(url, open_timeout=10, close_timeout=5) as ws:
            ws.send(json.dumps({"voice_config": voice_config, "context_id": context_id}))
            ws.send(json.dumps({"text": text, "context_id": context_id, "end": True}))

            for raw in ws:
                try:
                    msg = json.loads(raw)
                except (TypeError, ValueError):
                    logger.debug("murf: ignoring non-JSON WS frame")
                    continue
                if not isinstance(msg, dict):
                    continue

                if msg.get("error"):
                    raise RuntimeError(f"Murf WS error: {msg['error']}")

                audio_b64 = msg.get("audio")
                if audio_b64:
                    yield base64.b64decode(audio_b64)

                if msg.get("final"):
                    break

    def open_live_session(
        self,
        *,
        voice: Optional[str] = None,
        model: Optional[str] = None,
        format: str = "pcm",
        **extra: Any,
    ) -> "MurfLiveSession":
        """Open one persistent WebSocket connection for streaming multiple
        text segments as a single continuous spoken turn.

        Murf's own context-id docs describe exactly this pattern for text
        arriving in parts from an LLM: send several ``{"text": ...,
        "context_id": same_id}`` messages over one connection, with
        ``end: true`` only on the last one — ``end`` closes that context,
        not the connection (source:
        https://murf.ai/api/docs/text-to-speech/web-sockets/context-id).
        :meth:`stream` reconnects per call, which is correct for a single
        one-shot utterance but pays a full connection-setup cost per
        sentence for a multi-sentence reply fed in incrementally — this
        is the low-latency alternative for that case. Optional: callers
        without a use for it keep using :meth:`stream`/:meth:`synthesize`
        unchanged.
        """
        from websockets.sync.client import connect

        api_key = self._require_api_key()
        voice_id = self._resolve_voice(voice, extra)
        ws_model = model or os.environ.get("MURF_TTS_MODEL", "").strip() or DEFAULT_MODEL
        sample_rate = int(extra.get("sample_rate") or DEFAULT_SAMPLE_RATE)
        murf_format = _normalize_format(extra.get("murf_format") or format)
        context_id = extra.get("context_id") or uuid.uuid4().hex

        url = _build_ws_url(api_key, model=ws_model, sample_rate=sample_rate, fmt=murf_format)

        voice_config: Dict[str, Any] = {"voiceId": voice_id}
        for key in ("locale", "style", "rate", "pitch", "variation"):
            if extra.get(key) is not None:
                voice_config[key] = extra[key]

        logger.debug(
            "murf: opening live session (model=%s, voice=%s, sample_rate=%s)",
            ws_model, voice_id, sample_rate,
        )
        ws = connect(url, open_timeout=10, close_timeout=5)
        try:
            ws.send(json.dumps({"voice_config": voice_config, "context_id": context_id}))
        except Exception:
            ws.close()
            raise
        return MurfLiveSession(ws, context_id)

    def synthesize(
        self,
        text: str,
        output_path: str,
        *,
        voice: Optional[str] = None,
        model: Optional[str] = None,
        speed: Optional[float] = None,
        format: str = DEFAULT_OUTPUT_FORMAT,
        **extra: Any,
    ) -> str:
        # `speed` maps to Murf's `rate` field; pass `rate=` via **extra to
        # control it. The ABC allows providers to ignore `speed`, so it's a
        # no-op here rather than an assumed mapping.
        if speed is not None:
            logger.debug("murf: ignoring speed=%r; pass rate= via extra instead", speed)

        with open(output_path, "wb") as f:
            for chunk in self.stream(text, voice=voice, model=model, format=format, **extra):
                f.write(chunk)
        return output_path


class MurfLiveSession:
    """One persistent WebSocket connection streaming multiple text
    segments as a single continuous spoken turn.

    Returned by :meth:`MurfTTSProvider.open_live_session`. Per Murf's own
    concurrency model (their WS supports multiplexed context_ids on one
    connection), exactly one thread should call :meth:`send_text` and a
    single (possibly different) thread should iterate :meth:`iter_audio`
    — matches the underlying ``websockets.sync`` connection's own
    contract of one sender / one receiver, never two callers of the same
    method concurrently.
    """

    def __init__(self, ws, context_id: str):
        self._ws = ws
        self._context_id = context_id

    def send_text(self, text: str, *, end: bool = False) -> None:
        """Send the next chunk of text for this turn. Set ``end=True`` on
        the final chunk — per Murf's context-id docs this closes the
        *context*, not the connection."""
        self._ws.send(json.dumps({
            "text": text, "context_id": self._context_id, "end": end,
        }))

    def iter_audio(self) -> Iterator[bytes]:
        """Yield decoded audio chunks in arrival order (Murf guarantees
        output order matches input order) until the final marker for
        this context's ``end: true`` segment is received."""
        for raw in self._ws:
            try:
                msg = json.loads(raw)
            except (TypeError, ValueError):
                continue
            if not isinstance(msg, dict):
                continue
            if msg.get("error"):
                raise RuntimeError(f"Murf WS error: {msg['error']}")
            audio_b64 = msg.get("audio")
            if audio_b64:
                yield base64.b64decode(audio_b64)
            if msg.get("final"):
                break

    def close(self) -> None:
        try:
            self._ws.close()
        except Exception:
            pass


def register(ctx) -> None:
    ctx.register_tts_provider(MurfTTSProvider())
