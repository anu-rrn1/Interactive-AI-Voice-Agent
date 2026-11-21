"""
FastAPI entry point for the real-time voice agent.

Endpoints:
    GET  /, /app.js, /style.css             The web page (so one service hosts everything).
    GET  /personas                         Personas and translator languages for the UI.
    WS   /ws/conversation/{session_id}      Live voice conversation (see ``conversation.py``).

Run with::

    python -m uvicorn main:app --port 8000
"""

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv

# Environment variables must be loaded before the services are imported:
# they read their API keys at import time.
load_dotenv()

# Logging must be configured before importing modules that log; otherwise their
# first log call installs a WARNING-level default and hides the INFO timing lines.
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

from fastapi import FastAPI, WebSocket  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402

from conversation import Conversation  # noqa: E402
from personas import DEFAULT_PERSONA, DEFAULT_TARGET_LANGUAGE, PERSONAS, TRANSLATOR_LANGUAGES  # noqa: E402
from schemas import LanguageInfo, PersonaInfo, PersonasResponse  # noqa: E402
from services.gemini_service import GEMINI_MODEL, close_client  # noqa: E402
from services.murf_service import load_voice_catalogue  # noqa: E402


# ── Configuration ────────────────────────────────────────────────────────────

REQUIRED_ENV_VARS = ("ASSEMBLYAI_API_KEY", "GEMINI_API_KEY", "MURF_API_KEY")

FRONTEND_DIR = Path(__file__).resolve().parent

OPTIONAL_ENV_VARS = ("SERPAPI_API_KEY",)  # Only the Web Researcher persona needs it.

for var in REQUIRED_ENV_VARS:
    if not os.getenv(var):
        logger.warning(f"Environment variable '{var}' not found. Some features may not work.")
for var in OPTIONAL_ENV_VARS:
    if not os.getenv(var):
        logger.info(f"Optional environment variable '{var}' not set; the Web Researcher persona can't search.")

# In-memory session state. A production deployment would use a database instead.
chat_histories: dict[str, list[dict]] = {}
"""Chat turns per session id."""

session_modes: dict[str, str] = {}
"""Mode key (persona, plus target language for the translator) per session id."""


# ── Application ──────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Warms the Murf voice cache on startup and closes the Gemini client on shutdown."""
    logger.info(f"Using Gemini model: {GEMINI_MODEL}")
    await load_voice_catalogue()
    yield
    await close_client()


app = FastAPI(
    title="Voice Chat Agent API",
    description=(
        "Real-time voice agent with personas, emotion-aware speech and live translation "
        "(AssemblyAI, Gemini, Murf)."
    ),
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict this in production.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Frontend ─────────────────────────────────────────────────────────────────
# Only these three files are served. The project folder is deliberately not mounted
# as static files, so `.env` and the source code can never be downloaded.

@app.get("/", include_in_schema=False)
async def index_page() -> FileResponse:
    """Serves the web page."""
    return FileResponse(FRONTEND_DIR / "index.html", media_type="text/html")


@app.get("/app.js", include_in_schema=False)
async def app_script() -> FileResponse:
    """Serves the frontend script."""
    return FileResponse(FRONTEND_DIR / "app.js", media_type="application/javascript")


@app.get("/style.css", include_in_schema=False)
async def stylesheet() -> FileResponse:
    """Serves the stylesheet."""
    return FileResponse(FRONTEND_DIR / "style.css", media_type="text/css")


# ── API ──────────────────────────────────────────────────────────────────────

@app.get("/personas", response_model=PersonasResponse)
async def list_personas() -> PersonasResponse:
    """Lists the available personas and translator target languages.

    Returns:
        The personas, languages and their defaults, used to fill the UI dropdowns.
    """
    return PersonasResponse(
        personas=[
            PersonaInfo(id=persona_id, name=persona["name"], description=persona["description"])
            for persona_id, persona in PERSONAS.items()
        ],
        languages=[
            LanguageInfo(code=code, label=language["label"])
            for code, language in TRANSLATOR_LANGUAGES.items()
        ],
        default_persona=DEFAULT_PERSONA,
        default_language=DEFAULT_TARGET_LANGUAGE,
    )


@app.websocket("/ws/conversation/{session_id}")
async def conversation_ws(websocket: WebSocket, session_id: str) -> None:
    """Runs a real-time voice conversation.

    Browser -> server:
        ``{"type": "config", "persona", "target_language", "timezone",
        "utc_offset_minutes"}`` first, then binary
        16 kHz PCM16 mic audio, plus ``interrupt``, ``playback``, ``config`` and
        ``stop`` control messages.

    Server -> browser:
        ``ready``, ``partial``, ``user_turn``, ``turn_discarded``, ``search``, ``assistant_start``,
        ``assistant_delta``, ``audio`` (base64 MP3 per sentence), ``assistant_end``,
        ``assistant_interrupted``, ``history_reset`` and ``error``.

    Args:
        websocket: The incoming WebSocket connection.
        session_id: The browser's session id, which keys the chat history.
    """
    await websocket.accept()
    logger.info(f"[{session_id[:8]}] Conversation connected.")
    await Conversation(websocket, session_id, chat_histories, session_modes).run()
