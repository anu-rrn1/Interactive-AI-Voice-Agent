"""
AssemblyAI Universal-Streaming speech-to-text.

A thin client for the v3 streaming WebSocket: send raw PCM16 audio, read
``Begin`` / ``Turn`` / ``Termination`` events.
"""

import asyncio
import json
import logging
import os
from collections.abc import AsyncIterator
from urllib.parse import urlencode

import websockets
from fastapi import HTTPException
from websockets.asyncio.client import ClientConnection, connect

logger = logging.getLogger(__name__)


# ── Configuration ────────────────────────────────────────────────────────────

ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
STREAMING_URL = "wss://streaming.assemblyai.com/v3/ws"
SAMPLE_RATE = 16000  # The browser sends 16 kHz mono PCM16.
CONNECT_TIMEOUT_S = 15
CLOSE_TIMEOUT_S = 3

# Multilingual model: English, Spanish, French, German, Italian and Portuguese,
# with language detection.
STREAMING_MODEL = os.getenv("ASSEMBLYAI_STREAMING_MODEL") or "universal-streaming-multilingual"

# How long the user must stay silent before their turn counts as finished.
TURN_SILENCE_MS = int(os.getenv("TURN_SILENCE_MS") or 3000)


# ── Streaming session ────────────────────────────────────────────────────────

class AssemblyAIStream:
    """A live AssemblyAI Universal-Streaming session.

    Send audio with :meth:`send_audio` and read events from :meth:`events`.
    A turn ends only after ``TURN_SILENCE_MS`` of silence: both the semantic
    end-of-turn path and the max-silence path are set to that value.

    Args:
        ws: An open connection on which the ``Begin`` message was already received.
    """

    def __init__(self, ws: ClientConnection) -> None:
        self._ws = ws

    @classmethod
    async def connect(cls) -> "AssemblyAIStream":
        """Opens a streaming session.

        Returns:
            A ready-to-use stream.

        Raises:
            HTTPException: 500 if the API key is missing, 401 if it is rejected,
                502 if the connection fails.
        """
        if not ASSEMBLYAI_API_KEY:
            logger.error("AssemblyAI API key is not configured.")
            raise HTTPException(status_code=500, detail="AssemblyAI API key is not configured.")

        params = {
            "sample_rate": SAMPLE_RATE,
            "encoding": "pcm_s16le",
            "format_turns": "true",
            "speech_model": STREAMING_MODEL,
            "language_detection": "true",
            "min_end_of_turn_silence_when_confident": TURN_SILENCE_MS,
            "max_turn_silence": TURN_SILENCE_MS,
        }
        uri = f"{STREAMING_URL}?{urlencode(params)}"

        try:
            ws = await connect(
                uri,
                additional_headers={"Authorization": ASSEMBLYAI_API_KEY},
                open_timeout=CONNECT_TIMEOUT_S,
            )
            first = json.loads(await asyncio.wait_for(ws.recv(), timeout=CONNECT_TIMEOUT_S))
        except (OSError, asyncio.TimeoutError, websockets.WebSocketException) as e:
            raise HTTPException(
                status_code=502, detail=f"AssemblyAI streaming connection failed: {type(e).__name__}: {e}"
            )

        if first.get("type") != "Begin":
            await ws.close()
            error = first.get("error") or first
            error_text = str(error).lower()
            status = 401 if "unauthorized" in error_text or "api key" in error_text else 502
            raise HTTPException(status_code=status, detail=f"AssemblyAI streaming error: {error}")

        logger.info(f"AssemblyAI streaming session started ({STREAMING_MODEL}, turn silence {TURN_SILENCE_MS}ms).")
        return cls(ws)

    async def send_audio(self, chunk: bytes) -> None:
        """Sends one chunk of 16 kHz mono PCM16 audio."""
        await self._ws.send(chunk)

    async def events(self) -> AsyncIterator[dict]:
        """Yields AssemblyAI messages (``Turn``, ``Termination``, ...) until the session closes.

        Raises:
            HTTPException: If AssemblyAI reports an error or drops the connection.
        """
        try:
            async for raw in self._ws:
                if isinstance(raw, bytes):
                    continue
                message = json.loads(raw)
                if "error" in message and message.get("type") != "Turn":
                    raise HTTPException(status_code=502, detail=f"AssemblyAI streaming error: {message['error']}")
                yield message
        except websockets.ConnectionClosedError as e:
            reason = e.rcvd.reason if e.rcvd else e
            raise HTTPException(status_code=502, detail=f"AssemblyAI streaming connection closed: {reason}")

    async def close(self) -> None:
        """Ends the session gracefully; never raises."""
        try:
            await self._ws.send(json.dumps({"type": "Terminate"}))
            await asyncio.wait_for(self._ws.close(), timeout=CLOSE_TIMEOUT_S)
        except Exception:
            pass
