"""
Real-time voice conversation over a WebSocket.

Pipeline for each user turn::

    browser mic (16 kHz PCM16)
      -> AssemblyAI streaming STT (the turn ends after the silence window)
      -> Gemini streams "[mood: X]" followed by the reply text
      -> the reply is split into sentences
      -> Murf voices the sentences in parallel
      -> MP3 clips are sent back in order and played by the browser

The user can interrupt at any time: the browser stops playback and sends
``{"type": "interrupt"}``, which cancels the reply in progress. A newly finished
user turn cancels it as well.
"""

import asyncio
import base64
import json
import logging
import re
import time
from datetime import datetime, timedelta, timezone

from fastapi import HTTPException, WebSocket, WebSocketDisconnect

from personas import (
    DEFAULT_PERSONA,
    DEFAULT_TARGET_LANGUAGE,
    FIRST_TURN_RULE,
    MOOD_DELIVERY,
    MOOD_EMOJI,
    MOODS,
    PERSONAS,
    TRANSLATOR_FIRST_TURN_RULE,
    TRANSLATOR_LANGUAGES,
    TRANSLATOR_PROMPT_TEMPLATE,
)
from services.assemblyai_service import AssemblyAIStream
from services.gemini_service import stream_gemini_reply
from services.murf_service import resolve_voice, synthesize_speech
from services.serpapi_service import search_web

logger = logging.getLogger(__name__)


# ── Configuration ────────────────────────────────────────────────────────────

MAX_PARALLEL_TTS = 3
"""Number of sentences synthesized at the same time (they are still sent in order)."""

MIN_SENTENCE_LENGTH = 20
"""Shorter sentences are merged with the next one so Murf isn't called for tiny clips."""

ECHO_THRESHOLD = 0.6
"""Share of heard words that must match the assistant's reply to count as speaker echo."""

MOOD_TAG_MAX_LENGTH = 60
"""Characters to wait for a "[mood: X]" tag before treating the text as plain reply."""

MAX_UTC_OFFSET_MINUTES = 14 * 60
"""Largest real-world UTC offset; anything beyond it from the browser is ignored."""


# ── Text helpers ─────────────────────────────────────────────────────────────

class MoodTagParser:
    """Strips the leading ``[mood: X]`` tag from a streamed reply and remembers ``X``.

    Attributes:
        mood: The detected mood, or ``"neutral"`` if the tag is missing or unknown.
        done: True once the tag has been consumed (or ruled out).
    """

    _TAG = re.compile(r"^\s*\[\s*mood\s*:\s*([a-zA-Z]+)\s*\]\s*", re.IGNORECASE)

    def __init__(self) -> None:
        self._buffer = ""
        self.done = False
        self.mood = "neutral"

    def feed(self, chunk: str) -> str | None:
        """Consumes one streamed chunk.

        Args:
            chunk: The next piece of model output.

        Returns:
            Reply text with the tag removed (possibly an empty string), or ``None``
            while the parser is still waiting to see the whole tag.
        """
        if self.done:
            return chunk

        self._buffer += chunk
        match = self._TAG.match(self._buffer)
        if match:
            mood = match.group(1).lower()
            self.mood = mood if mood in MOODS else "neutral"
            return self._finish(self._buffer[match.end():])

        stripped = self._buffer.lstrip()
        no_tag_possible = (
            (stripped and not stripped.startswith("["))
            or "]" in self._buffer
            or len(self._buffer) > MOOD_TAG_MAX_LENGTH
        )
        if no_tag_possible:
            return self._finish(self._buffer)
        return None

    def flush(self) -> str:
        """Returns any buffered text once the stream has ended."""
        return "" if self.done else self._finish(self._buffer)

    def _finish(self, rest: str) -> str:
        """Marks the tag as handled and returns the remaining reply text."""
        self.done = True
        self._buffer = ""
        return rest


class SentenceSplitter:
    """Groups streamed text into sentences so each can be voiced as soon as it is complete.

    The very first chunk may also end at a comma, semicolon or colon: a shorter
    first clip means the assistant starts speaking sooner.

    Args:
        min_length: Minimum characters per emitted sentence; shorter ones are
            merged with the following sentence.
    """

    _BOUNDARY = re.compile(r"[.!?।。！？]+[\"'”’)\]]*\s+")
    _FIRST_BOUNDARY = re.compile(r"(?:[.!?।。！？]+[\"'”’)\]]*|[,;:，、])\s+")

    def __init__(self, min_length: int = MIN_SENTENCE_LENGTH) -> None:
        self._buffer = ""
        self._min_length = min_length
        self._emitted_any = False

    def feed(self, text: str) -> list[str]:
        """Adds streamed text and returns any sentences that are now complete.

        Args:
            text: The next piece of reply text.

        Returns:
            Complete sentences, in order. May be empty.
        """
        self._buffer += text
        sentences: list[str] = []
        start = 0

        if not self._emitted_any:
            for match in self._FIRST_BOUNDARY.finditer(self._buffer):
                candidate = self._buffer[:match.end()].strip()
                if len(candidate) >= self._min_length:
                    sentences.append(candidate)
                    start = match.end()
                    self._emitted_any = True
                    break
            if not self._emitted_any:
                return []

        for match in self._BOUNDARY.finditer(self._buffer, start):
            candidate = self._buffer[start:match.end()].strip()
            if len(candidate) >= self._min_length:
                sentences.append(candidate)
                start = match.end()

        self._buffer = self._buffer[start:]
        return sentences

    def flush(self) -> list[str]:
        """Returns whatever text is left once the stream has ended."""
        rest, self._buffer = self._buffer.strip(), ""
        self._emitted_any = True
        return [rest] if rest else []


_MARKDOWN_SYMBOLS = re.compile(r"[*_#`~]+")
_WORD = re.compile(r"[\w']+", re.UNICODE)


def clean_for_speech(text: str) -> str:
    """Removes markdown symbols the model may still emit.

    Args:
        text: A sentence of reply text.

    Returns:
        The text without ``* _ # ` ~`` characters, stripped of surrounding whitespace.
    """
    return _MARKDOWN_SYMBOLS.sub("", text).strip()


def _words(text: str) -> list[str]:
    """Splits text into lowercase words."""
    return _WORD.findall(text.lower())


def looks_like_echo(heard: str, spoken: str, threshold: float = ECHO_THRESHOLD) -> bool:
    """Detects the speakers leaking into the microphone.

    Args:
        heard: What speech-to-text picked up.
        spoken: What the assistant is saying or just said.
        threshold: Share of heard words that must appear in ``spoken``.

    Returns:
        True if ``heard`` is mostly the assistant's own words.
    """
    heard_words = _words(heard)
    if not heard_words or not spoken:
        return False
    spoken_words = set(_words(spoken))
    hits = sum(1 for word in heard_words if word in spoken_words)
    return hits / len(heard_words) >= threshold


def current_datetime_rule(utc_offset_minutes: int | None = None, timezone_name: str | None = None) -> str:
    """Builds the prompt line that tells the model the current date and time.

    The model has no clock of its own, so without this it guesses a date from its
    training data.

    Args:
        utc_offset_minutes: The user's offset from UTC (e.g. 330 for India);
            ``None`` uses the server's local time.
        timezone_name: The user's IANA time zone name (e.g. ``"Asia/Kolkata"``), for display.

    Returns:
        A sentence such as "The current date and time for the user is Friday,
        21 November 2026, 3:48 PM (Asia/Kolkata, UTC+05:30)."
    """
    if utc_offset_minutes is None:
        now = datetime.now().astimezone()
    else:
        now = datetime.now(timezone(timedelta(minutes=utc_offset_minutes)))

    offset = now.strftime("%z")
    offset_text = f"UTC{offset[:3]}:{offset[3:]}"
    zone = f"{timezone_name}, {offset_text}" if timezone_name else offset_text
    hour = now.strftime("%I").lstrip("0")
    stamp = f"{now.strftime('%A')}, {now.day} {now.strftime('%B %Y')}, {hour}:{now.strftime('%M %p')}"
    return (
        f"The current date and time for the user is {stamp} ({zone}). "
        "Use it whenever the user asks about the date, the time or anything relative to today."
    )


def _parse_utc_offset(value: object) -> int | None:
    """Validates the browser's UTC offset in minutes; returns ``None`` if invalid."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    minutes = int(value)
    return minutes if abs(minutes) <= MAX_UTC_OFFSET_MINUTES else None


# ── Persona resolution ───────────────────────────────────────────────────────

def resolve_mode(persona_id: str | None, target_language: str | None) -> tuple[dict, str, str]:
    """Resolves the persona (and translator language) the browser asked for.

    Args:
        persona_id: A key of ``PERSONAS``; ``None`` selects the default persona.
        target_language: A key of ``TRANSLATOR_LANGUAGES``; only used by the translator.

    Returns:
        A tuple of ``(persona_config, system_prompt, mode_key)``. ``mode_key``
        identifies the mode, e.g. ``"storyteller"`` or ``"translator:es"``.

    Raises:
        ValueError: If the persona or the target language is unknown.
    """
    persona_id = persona_id or DEFAULT_PERSONA
    persona = PERSONAS.get(persona_id)
    if persona is None:
        raise ValueError(f"Unknown persona '{persona_id}'.")

    if persona_id == "translator":
        code = target_language or DEFAULT_TARGET_LANGUAGE
        language = TRANSLATOR_LANGUAGES.get(code)
        if language is None:
            raise ValueError(f"Unsupported target language '{code}'.")
        return (
            {
                **persona,
                "id": persona_id,
                "voice_id": language["voice_id"],
                "first_turn_rule": TRANSLATOR_FIRST_TURN_RULE.format(label=language["label"]),
            },
            TRANSLATOR_PROMPT_TEMPLATE.format(label=language["label"]),
            f"translator:{code}",
        )

    return {**persona, "id": persona_id, "first_turn_rule": FIRST_TURN_RULE}, persona["system_prompt"], persona_id


# ── Conversation ─────────────────────────────────────────────────────────────

class Conversation:
    """Drives one browser connection: speech-to-text, replies, audio and interruptions.

    Args:
        websocket: The accepted browser WebSocket.
        session_id: The browser's session id; history is stored under it.
        histories: Shared ``session_id -> chat turns`` store.
        session_modes: Shared ``session_id -> mode_key`` store, used to reset
            history when the persona changes.
    """

    def __init__(
        self,
        websocket: WebSocket,
        session_id: str,
        histories: dict[str, list[dict]],
        session_modes: dict[str, str],
    ) -> None:
        self.ws = websocket
        self.session_id = session_id
        self.persona: dict = {}
        self.system_prompt = ""
        self.stt: AssemblyAIStream | None = None

        self._histories = histories
        self._session_modes = session_modes
        self._response_task: asyncio.Task | None = None
        self._turn_counter = 0
        self._last_turn_order = -1
        self._assistant_text = ""      # What the assistant is saying or just said (echo filter).
        self._client_playing = False   # Reported by the browser.
        self._carry_over = ""          # User text whose reply was cancelled before any text.
        self._utc_offset_minutes: int | None = None  # The user's time zone, from the browser.
        self._timezone_name: str | None = None
        self._send_lock = asyncio.Lock()

    @property
    def history(self) -> list[dict]:
        """This session's chat turns, as ``{"role", "content"}`` dicts."""
        return self._histories.setdefault(self.session_id, [])

    # ── Messaging ──

    async def send(self, message: dict) -> None:
        """Sends a JSON message to the browser, ignoring a closed connection."""
        async with self._send_lock:
            try:
                await self.ws.send_json(message)
            except (RuntimeError, WebSocketDisconnect):
                pass

    async def send_error(self, technical: str, message: str | None = None) -> None:
        """Sends an error to the browser.

        Args:
            technical: Technical details, shown behind the "View issue" link.
            message: Optional friendly text; the browser picks one if omitted.
        """
        payload = {"type": "error", "technical": technical}
        if message:
            payload["message"] = message
        await self.send(payload)

    def _log(self, text: str) -> None:
        """Logs a message prefixed with the short session id."""
        logger.info(f"[{self.session_id[:8]}] {text}")

    # ── Lifecycle ──

    async def run(self) -> None:
        """Runs the conversation until the browser disconnects or ends it."""
        stt_task: asyncio.Task | None = None
        try:
            # 1. The browser first says which persona to use.
            first = await self.ws.receive_json()
            if first.get("type") != "config" or not await self._apply_config(first, initial=True):
                return

            # 2. Open the live transcription session.
            try:
                self.stt = await AssemblyAIStream.connect()
            except HTTPException as e:
                await self.send_error(str(e.detail))
                return
            await self.send({"type": "ready"})
            stt_task = asyncio.create_task(self._stt_loop())

            # 3. Relay mic audio and control messages until the browser leaves.
            await self._client_loop(stt_task)
        except WebSocketDisconnect:
            pass
        finally:
            await self._cancel_response()
            if stt_task:
                stt_task.cancel()
            if self.stt:
                await self.stt.close()
            self._log("Conversation ended.")

    async def _client_loop(self, stt_task: asyncio.Task) -> None:
        """Forwards mic audio to STT and handles the browser's control messages."""
        while not stt_task.done():
            message = await self.ws.receive()
            if message["type"] == "websocket.disconnect":
                return

            if message.get("bytes") is not None:
                try:
                    await self.stt.send_audio(message["bytes"])
                except Exception:
                    return  # STT closed; _stt_loop reports why.
                continue

            if message.get("text") is None:
                continue
            try:
                data = json.loads(message["text"])
            except ValueError:
                continue

            kind = data.get("type")
            if kind == "config":
                await self._apply_config(data)
            elif kind == "interrupt":
                if self._response_task and not self._response_task.done():
                    self._log("User interrupted the assistant.")
                await self._cancel_response()
                self._client_playing = False
            elif kind == "playback":
                self._client_playing = bool(data.get("playing"))
            elif kind == "stop":
                return

    async def _apply_config(self, data: dict, initial: bool = False) -> bool:
        """Applies a persona/language choice, resetting history when the mode changes.

        Args:
            data: The browser's ``config`` message.
            initial: True for the first config of the connection.

        Returns:
            False if the choice was invalid (an error has been sent to the browser).
        """
        try:
            persona, prompt, mode_key = resolve_mode(data.get("persona"), data.get("target_language"))
        except ValueError as e:
            await self.send_error(str(e), "⚙️ That option isn't supported. Please refresh the page and try again.")
            return False

        self._utc_offset_minutes = _parse_utc_offset(data.get("utc_offset_minutes"))
        timezone_name = data.get("timezone")
        self._timezone_name = timezone_name[:64] if isinstance(timezone_name, str) and timezone_name else None

        self.persona, self.system_prompt = persona, prompt
        if self._session_modes.get(self.session_id) != mode_key:
            had_history = bool(self._histories.get(self.session_id))
            await self._cancel_response()
            self._histories[self.session_id] = []
            self._session_modes[self.session_id] = mode_key
            self._carry_over = ""
            self._log(f"Mode set to '{mode_key}' (history reset).")
            if had_history or not initial:
                await self.send({"type": "history_reset", "persona": persona["name"]})
        return True

    # ── Speech-to-text events ──

    async def _stt_loop(self) -> None:
        """Reads AssemblyAI events until the session ends or fails."""
        try:
            async for event in self.stt.events():
                if event.get("type") == "Turn":
                    await self._on_turn_event(event)
                elif event.get("type") == "Termination":
                    break
        except HTTPException as e:
            await self.send_error(str(e.detail))
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.exception("Speech-to-text loop failed")
            await self.send_error(f"Speech-to-text failed: {type(e).__name__}: {e}")

    async def _on_turn_event(self, event: dict) -> None:
        """Handles one AssemblyAI ``Turn`` event: live partials, or a finished user turn."""
        transcript = (event.get("transcript") or "").strip()

        if not event.get("end_of_turn"):
            if transcript:
                await self.send({"type": "partial", "text": transcript})
            return
        if not event.get("turn_is_formatted"):
            return  # A formatted copy of this final turn follows.
        if event.get("turn_order", -1) <= self._last_turn_order:
            return
        self._last_turn_order = event.get("turn_order", self._last_turn_order + 1)

        assistant_busy = self._client_playing or (self._response_task and not self._response_task.done())
        if not transcript or (assistant_busy and looks_like_echo(transcript, self._assistant_text)):
            if transcript:
                self._log(f"Ignored echo of the assistant's voice: '{transcript}'")
            await self.send({"type": "turn_discarded"})
            return

        language = event.get("language_code")
        self._log(f"User ({language}): '{transcript}'")
        await self.send({"type": "user_turn", "text": transcript, "language": language})

        await self._cancel_response()
        self._turn_counter += 1
        self._response_task = asyncio.create_task(self._respond(transcript, self._turn_counter))

    # ── Assistant reply ──

    async def _cancel_response(self) -> None:
        """Cancels the reply in progress, if any, and waits for it to stop."""
        task = self._response_task
        if task and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        self._response_task = None

    async def _with_search_results(self, user_turn: dict, turn_id: int) -> dict:
        """Searches the web for the user's question and attaches the results for the model.

        A failed search never fails the turn: the user sees a friendly notice and the
        model is told to answer without live results. Only the plain question is kept
        in the chat history, so old results don't pile up in later prompts.

        Args:
            user_turn: The user's ``{"role", "content"}`` turn.
            turn_id: Id of the reply being prepared.

        Returns:
            A copy of the turn whose content includes the search results.
        """
        query = user_turn["content"]
        await self.send({"type": "search", "turn_id": turn_id, "status": "searching", "query": query})
        started_at = time.perf_counter()
        try:
            results = await search_web(query)
        except HTTPException as e:
            self._log(f"Web search failed: {e.detail}")
            await self.send_error(str(e.detail))
            await self.send({"type": "search", "turn_id": turn_id, "status": "failed"})
            note = "(Live web search is unavailable right now. Say so briefly, then answer from your own knowledge.)"
            return {"role": "user", "content": f"{query}\n\n{note}"}

        self._log(f"Web search: {len(results.sources)} sources in {time.perf_counter() - started_at:.2f}s for '{query}'")
        await self.send({
            "type": "search",
            "turn_id": turn_id,
            "status": "done",
            "sources": [{"title": s.title, "source": s.source, "link": s.link} for s in results.sources],
        })
        return {
            "role": "user",
            "content": f"{query}\n\nLive Google search results for this message:\n{results.context}",
        }

    async def _respond(self, user_text: str, turn_id: int) -> None:
        """Streams one reply: text deltas, then MP3 audio for each sentence, in order.

        Args:
            user_text: What the user said this turn.
            turn_id: Id that tags every message of this reply for the browser.
        """
        if self._carry_over:
            user_text = f"{self._carry_over} {user_text}"
            self._carry_over = ""

        user_turn = {"role": "user", "content": user_text}
        reply = ""
        started = False
        started_at = time.perf_counter()
        first_audio_at: list[float] = []

        audio_queue: asyncio.Queue = asyncio.Queue()
        tts_tasks: list[asyncio.Task] = []
        semaphore = asyncio.Semaphore(MAX_PARALLEL_TTS)
        voice_id = self.persona.get("voice_id")
        delivery = MOOD_DELIVERY["neutral"]
        parser, splitter = MoodTagParser(), SentenceSplitter()

        async def synthesize(sentence: str) -> bytes:
            async with semaphore:
                return await synthesize_speech(sentence, voice_id, delivery["rate"], delivery["pitch"])

        def queue_sentences(sentences: list[str]) -> None:
            for sentence in sentences:
                sentence = clean_for_speech(sentence)
                if not sentence:
                    continue
                task = asyncio.create_task(synthesize(sentence))
                # Errors are re-raised where the task is awaited; this only silences
                # "exception never retrieved" for tasks dropped after an interruption.
                task.add_done_callback(lambda t: t.cancelled() or t.exception())
                tts_tasks.append(task)
                audio_queue.put_nowait((task, sentence))

        async def send_audio_in_order() -> None:
            seq = 0
            while (item := await audio_queue.get()) is not None:
                task, sentence = item
                audio = await task
                if not first_audio_at:
                    first_audio_at.append(time.perf_counter() - started_at)
                await self.send({
                    "type": "audio",
                    "turn_id": turn_id,
                    "seq": seq,
                    "text": sentence,
                    "format": "mp3",
                    "data": base64.b64encode(audio).decode("ascii"),
                })
                seq += 1

        async def start_if_needed() -> None:
            nonlocal started, voice_id, delivery
            if started:
                return
            started = True
            mood = parser.mood
            voice_id = await resolve_voice(self.persona.get("voice_id"))
            if not self.persona.get("neutral_delivery"):
                delivery = MOOD_DELIVERY.get(mood, MOOD_DELIVERY["neutral"])
            self._log(
                f"Mood: {mood} -> voice {voice_id}, {delivery['label']} "
                f"(rate {delivery['rate']}, pitch {delivery['pitch']})"
            )
            await self.send({
                "type": "assistant_start",
                "turn_id": turn_id,
                "mood": mood,
                "mood_emoji": MOOD_EMOJI.get(mood, "😐"),
                "voice_id": voice_id,
                "delivery": delivery["label"],
            })

        async def handle_text(text: str) -> None:
            nonlocal reply
            await start_if_needed()
            if text:
                reply += text
                self._assistant_text = reply
                await self.send({"type": "assistant_delta", "turn_id": turn_id, "text": text})
                queue_sentences(splitter.feed(text))

        system_prompt = f"{self.system_prompt} {current_datetime_rule(self._utc_offset_minutes, self._timezone_name)}"
        # The first reply in a persona (history is reset on every switch) opens with a
        # one-sentence introduction of the persona's role.
        if not self.history:
            system_prompt = f"{system_prompt} {self.persona['first_turn_rule']}"

        sender = asyncio.create_task(send_audio_in_order())
        try:
            self._assistant_text = ""
            model_turn_input = user_turn
            if self.persona.get("web_search"):
                model_turn_input = await self._with_search_results(user_turn, turn_id)

            async for chunk in stream_gemini_reply(self.history + [model_turn_input], system_prompt):
                text = parser.feed(chunk)
                if text is not None:
                    await handle_text(text)
            await handle_text(parser.flush())
            queue_sentences(splitter.flush())
            if not reply.strip():
                raise HTTPException(status_code=502, detail="Gemini returned an empty reply.")

            audio_queue.put_nowait(None)
            await sender

            await self.send({"type": "assistant_end", "turn_id": turn_id, "text": reply})
            self.history.extend([user_turn, {"role": "model", "content": reply}])
            first_audio = first_audio_at[0] if first_audio_at else 0
            self._log(
                f"Reply done: first audio {first_audio:.2f}s, "
                f"total {time.perf_counter() - started_at:.2f}s: '{reply[:80]}'"
            )
        except asyncio.CancelledError:
            if reply.strip():
                self.history.extend([user_turn, {"role": "model", "content": reply + " (interrupted)"}])
            else:
                self._carry_over = user_text
            await self.send({"type": "assistant_interrupted", "turn_id": turn_id})
            raise
        except HTTPException as e:
            self._carry_over = ""
            await self.send_error(str(e.detail))
            await self.send({"type": "assistant_end", "turn_id": turn_id, "text": reply, "failed": True})
        except Exception as e:
            logger.exception("Reply generation failed")
            await self.send_error(f"An internal server error occurred: {type(e).__name__}: {e}")
            await self.send({"type": "assistant_end", "turn_id": turn_id, "text": reply, "failed": True})
        finally:
            sender.cancel()
            for task in tts_tasks:
                task.cancel()
