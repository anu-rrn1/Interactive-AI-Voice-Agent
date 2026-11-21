"""
Google Gemini streaming replies.

Streams reply text over server-sent events, with retries for transient errors and
request hedging against Gemini's occasional multi-second latency spikes.
"""

import asyncio
import json
import logging
import os
from collections.abc import AsyncIterator

import httpx
from fastapi import HTTPException

logger = logging.getLogger(__name__)


# ── Configuration ────────────────────────────────────────────────────────────

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Lite models answer in ~1-2 s; full Flash models measured 13-50 s, too slow for voice.
GEMINI_MODEL = os.getenv("GEMINI_MODEL") or "gemini-3.1-flash-lite"

# If Gemini hasn't started answering after this many seconds, a backup request is
# sent and whichever answers first wins. Set to 0 to disable hedging.
HEDGE_AFTER_S = float(os.getenv("GEMINI_HEDGE_AFTER_S") or 2.5)

MAX_ATTEMPTS = 3
INITIAL_RETRY_DELAY_S = 1.0
REQUEST_TIMEOUT_S = 30
CONNECT_TIMEOUT_S = 10

_client: httpx.AsyncClient | None = None
_DONE = object()  # Sentinel marking the end of a hedged request's stream.


# ── HTTP client ──────────────────────────────────────────────────────────────

def _get_client() -> httpx.AsyncClient:
    """Returns the shared client, so the TLS connection is reused across turns (~100-200 ms saved)."""
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=httpx.Timeout(REQUEST_TIMEOUT_S, connect=CONNECT_TIMEOUT_S))
    return _client


async def close_client() -> None:
    """Closes the shared HTTP client (called on app shutdown)."""
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


# ── Public API ───────────────────────────────────────────────────────────────

async def stream_gemini_reply(chat_history: list[dict], system_prompt: str) -> AsyncIterator[str]:
    """Streams Gemini's reply text chunk by chunk, hedging against slow responses.

    A second identical request starts if the first is silent for ``HEDGE_AFTER_S``;
    only the request that answers first is used, and the other is cancelled.

    Args:
        chat_history: ``[{"role": "user" | "model", "content": str}, ...]``, ending
            with the user's turn.
        system_prompt: Persona instructions, sent as Gemini's ``systemInstruction``.

    Yields:
        Reply text chunks as they arrive.

    Raises:
        HTTPException: If every request fails.
    """
    queue: asyncio.Queue = asyncio.Queue()

    async def run(index: int) -> None:
        try:
            async for chunk in _stream_once(chat_history, system_prompt):
                await queue.put((index, chunk))
            await queue.put((index, _DONE))
        except Exception as e:  # Handed to the consumer below.
            await queue.put((index, e))

    tasks = [asyncio.create_task(run(0))]
    winner: int | None = None
    try:
        while True:
            hedge_pending = winner is None and len(tasks) == 1 and HEDGE_AFTER_S > 0
            try:
                index, item = await asyncio.wait_for(queue.get(), HEDGE_AFTER_S if hedge_pending else None)
            except asyncio.TimeoutError:
                logger.info(f"Gemini silent for {HEDGE_AFTER_S}s; sending a backup request.")
                tasks.append(asyncio.create_task(run(1)))
                continue

            if winner is None:
                others_running = any(not t.done() for i, t in enumerate(tasks) if i != index)
                if isinstance(item, Exception) and others_running:
                    continue  # Let the other request answer.
                winner = index
                for i, task in enumerate(tasks):
                    if i != index:
                        task.cancel()

            if index != winner:
                continue
            if item is _DONE:
                return
            if isinstance(item, Exception):
                raise item
            yield item
    finally:
        for task in tasks:
            task.cancel()


# ── Single request ───────────────────────────────────────────────────────────

def _build_payload(chat_history: list[dict], system_prompt: str) -> dict:
    """Builds the ``streamGenerateContent`` request body."""
    return {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": [
            {"role": "user" if turn["role"] == "user" else "model", "parts": [{"text": turn["content"]}]}
            for turn in chat_history
        ],
        # Thinking adds seconds of latency and voice replies don't need it. The setting
        # is dropped automatically for models that reject it.
        "generationConfig": {"thinkingConfig": {"thinkingBudget": 0}},
    }


async def _stream_once(chat_history: list[dict], system_prompt: str) -> AsyncIterator[str]:
    """Streams Gemini's reply for one request, retrying transient failures.

    Retries happen only before the first chunk is yielded, so the caller never
    sees duplicated text.

    Args:
        chat_history: See :func:`stream_gemini_reply`.
        system_prompt: See :func:`stream_gemini_reply`.

    Yields:
        Text chunks as they arrive (thought parts are skipped).

    Raises:
        HTTPException: 500 if the API key is missing, 401 if it is rejected,
            502 for API errors or blocked prompts, 504 on timeout.
    """
    if not GEMINI_API_KEY:
        logger.error("Gemini API key is not configured.")
        raise HTTPException(status_code=500, detail="Gemini API key is not configured.")

    payload = _build_payload(chat_history, system_prompt)
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:streamGenerateContent?alt=sse"
    headers = {"x-goog-api-key": GEMINI_API_KEY}

    delay = INITIAL_RETRY_DELAY_S
    attempt = 0
    while True:
        attempt += 1
        yielded = False
        try:
            async with _get_client().stream("POST", url, json=payload, headers=headers) as res:
                if res.status_code >= 400:
                    body = (await res.aread()).decode(errors="replace")
                    if res.status_code == 400 and "generationConfig" in payload:
                        logger.info(f"{GEMINI_MODEL} rejected the thinking setting; retrying without it.")
                        payload.pop("generationConfig")
                        attempt -= 1  # Not a real failure.
                        continue

                    detail = f"Gemini API error: HTTP {res.status_code} - Response: {body[:500]}"
                    retryable = res.status_code == 429 or res.status_code >= 500
                    if retryable and attempt < MAX_ATTEMPTS:
                        logger.warning(f"{detail} (attempt {attempt}/{MAX_ATTEMPTS}); retrying.")
                        await asyncio.sleep(delay)
                        delay *= 2
                        continue
                    status = 401 if res.status_code in (401, 403) else 502
                    raise HTTPException(status_code=status, detail=f"Failed to get response from Gemini API: {detail}")

                async for line in res.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = json.loads(line[len("data: "):])
                    candidate = (data.get("candidates") or [{}])[0]
                    parts = candidate.get("content", {}).get("parts", [])
                    text = "".join(part.get("text", "") for part in parts if not part.get("thought"))
                    if text:
                        yielded = True
                        yield text
                    if data.get("promptFeedback", {}).get("blockReason"):
                        raise HTTPException(
                            status_code=502, detail=f"Gemini blocked the request: {data['promptFeedback']}"
                        )
                return
        except HTTPException:
            raise
        except httpx.TimeoutException:
            if yielded or attempt >= MAX_ATTEMPTS:
                raise HTTPException(status_code=504, detail="Gemini API request timed out.")
            logger.warning(f"Gemini timed out (attempt {attempt}/{MAX_ATTEMPTS}); retrying.")
        except httpx.HTTPError as e:
            if yielded or attempt >= MAX_ATTEMPTS:
                raise HTTPException(
                    status_code=502, detail=f"Failed to get response from Gemini API: {type(e).__name__}: {e}"
                )
            logger.warning(f"Gemini connection error {e!r} (attempt {attempt}/{MAX_ATTEMPTS}); retrying.")
        await asyncio.sleep(delay)
        delay *= 2
