"""
Murf text-to-speech.

Uses Murf's streaming endpoint, which runs the low-latency FALCON model and has its
own voice catalogue. Voices are validated against that catalogue, and failed
requests fall back to the default voice so a reply is never left silent.
"""

import asyncio
import logging
import os

import httpx
from fastapi import HTTPException
from murf import Murf

from personas import DEFAULT_VOICE_ID

logger = logging.getLogger(__name__)


# ── Configuration ────────────────────────────────────────────────────────────

MURF_API_KEY = os.getenv("MURF_API_KEY")
MAX_MURF_TEXT_LENGTH = 3000  # Murf's per-request character limit.
AUDIO_FORMAT = "MP3"
AUDIO_SAMPLE_RATE = 24000
VOICES_URL = "https://api.murf.ai/v1/speech/voices"
VOICES_TIMEOUT_S = 20

_murf_client: Murf | None = None
_voice_ids: set[str] | None = None  # FALCON voice ids; None until loaded, or if loading failed.
_voices_lock = asyncio.Lock()


# ── Voice catalogue ──────────────────────────────────────────────────────────

def _get_client() -> Murf:
    """Returns the shared Murf SDK client."""
    global _murf_client
    if _murf_client is None:
        _murf_client = Murf(api_key=MURF_API_KEY)
    return _murf_client


async def load_voice_catalogue() -> set[str] | None:
    """Fetches and caches the streaming (FALCON) voice ids.

    Returns:
        The set of voice ids, or ``None`` if the catalogue is unavailable
        (in which case voices are not validated).
    """
    global _voice_ids
    if _voice_ids is not None or not MURF_API_KEY:
        return _voice_ids

    async with _voices_lock:
        if _voice_ids is None:
            try:
                async with httpx.AsyncClient(timeout=VOICES_TIMEOUT_S) as client:
                    res = await client.get(
                        VOICES_URL, params={"model": "FALCON"}, headers={"api-key": MURF_API_KEY}
                    )
                    res.raise_for_status()
                _voice_ids = {voice["voiceId"] for voice in res.json() if voice.get("voiceId")}
                logger.info(f"Loaded {len(_voice_ids)} Murf streaming voices.")
            except Exception as e:
                logger.warning(f"Could not fetch the Murf voice list; skipping voice validation: {e}")
                return None
    return _voice_ids


async def resolve_voice(voice_id: str | None) -> str:
    """Checks a voice against Murf's streaming catalogue.

    Args:
        voice_id: The requested voice; ``None`` selects the default voice.

    Returns:
        ``voice_id`` if streaming supports it, otherwise ``DEFAULT_VOICE_ID``.
    """
    voice_id = voice_id or DEFAULT_VOICE_ID
    catalogue = await load_voice_catalogue()
    if catalogue is not None and voice_id not in catalogue:
        logger.warning(
            f"Murf voice '{voice_id}' isn't available for streaming; falling back to '{DEFAULT_VOICE_ID}'."
        )
        return DEFAULT_VOICE_ID
    return voice_id


# ── Synthesis ────────────────────────────────────────────────────────────────

def _is_auth_error(error: Exception) -> bool:
    """True if a Murf SDK error looks like a bad or missing API key."""
    message = str(error)
    return "401" in message or "api key" in message.lower()


def _synthesize_blocking(text: str, voice_id: str, rate: int, pitch: int) -> bytes:
    """Calls the (blocking) Murf SDK and returns the complete audio clip."""
    kwargs = {"text": text, "voice_id": voice_id, "format": AUDIO_FORMAT, "sample_rate": AUDIO_SAMPLE_RATE}
    if rate:
        kwargs["rate"] = rate
    if pitch:
        kwargs["pitch"] = pitch
    return b"".join(_get_client().text_to_speech.stream(**kwargs))


async def synthesize_speech(text: str, voice_id: str, rate: int = 0, pitch: int = 0) -> bytes:
    """Converts one sentence to MP3 audio without blocking the event loop.

    If the first attempt fails, it retries once with the default voice and a
    natural delivery before giving up.

    Args:
        text: The sentence to speak (truncated to Murf's character limit).
        voice_id: A Murf FALCON voice id.
        rate: Speaking rate adjustment, -50..50.
        pitch: Pitch adjustment, -50..50.

    Returns:
        The MP3 audio bytes.

    Raises:
        HTTPException: If the Murf API key is missing, or if audio generation fails.
    """
    if not MURF_API_KEY:
        logger.error("Murf API key is not configured.")
        raise HTTPException(status_code=500, detail="Murf API key is not configured.")

    text = text[:MAX_MURF_TEXT_LENGTH]
    try:
        try:
            audio = await asyncio.to_thread(_synthesize_blocking, text, voice_id, rate, pitch)
        except Exception as e:
            already_default = voice_id == DEFAULT_VOICE_ID and not rate and not pitch
            if already_default or _is_auth_error(e):
                raise
            logger.warning(
                f"Murf failed with voice='{voice_id}', rate={rate}, pitch={pitch} ({e}); retrying with defaults."
            )
            audio = await asyncio.to_thread(_synthesize_blocking, text, DEFAULT_VOICE_ID, 0, 0)

        if not audio:
            raise HTTPException(status_code=502, detail="Murf API returned no audio.")
        return audio
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Murf API error: {e}")
        status = 401 if _is_auth_error(e) else 502
        raise HTTPException(status_code=status, detail=f"Failed to generate audio with Murf API: {e}")
