"""
SerpAPI Google web search.

Used by the Web Researcher persona: the user's question is searched before Gemini
answers, and the top results are passed to Gemini as context.
"""

import logging
import os
from dataclasses import dataclass

import httpx
from fastapi import HTTPException

logger = logging.getLogger(__name__)


# ── Configuration ────────────────────────────────────────────────────────────

SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
SEARCH_URL = "https://serpapi.com/search.json"
SEARCH_TIMEOUT_S = 10
MAX_ORGANIC_RESULTS = 5
MAX_NEWS_RESULTS = 3


# ── Results ──────────────────────────────────────────────────────────────────

@dataclass
class SearchSource:
    """One web result shown to the user as a source."""

    title: str
    source: str
    link: str


@dataclass
class SearchResults:
    """Search results formatted for the model, plus the sources for the UI."""

    context: str
    sources: list[SearchSource]


# ── Public API ───────────────────────────────────────────────────────────────

async def search_web(query: str) -> SearchResults:
    """Searches Google through SerpAPI.

    Args:
        query: What to search for (the user's spoken question).

    Returns:
        The answer box, knowledge graph, top news and organic results as plain
        text for the model, and the list of sources.

    Raises:
        HTTPException: 500 if the API key is missing, 401 if it is rejected,
            429 if the search quota is used up, 504 on timeout, 502 otherwise.
    """
    if not SERPAPI_API_KEY:
        logger.error("SerpAPI API key is not configured.")
        raise HTTPException(status_code=500, detail="SerpAPI API key is not configured.")

    params = {"engine": "google", "q": query, "api_key": SERPAPI_API_KEY, "hl": "en", "num": MAX_ORGANIC_RESULTS}
    try:
        async with httpx.AsyncClient(timeout=SEARCH_TIMEOUT_S) as client:
            res = await client.get(SEARCH_URL, params=params)
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="SerpAPI web search timed out.")
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"SerpAPI web search failed: {type(e).__name__}: {e}")

    data = res.json() if res.headers.get("content-type", "").startswith("application/json") else {}
    if res.status_code >= 400 or data.get("error"):
        error = data.get("error") or res.text[:300]
        status = res.status_code if res.status_code in (401, 429) else 502
        if "invalid api key" in str(error).lower():
            status = 401
        raise HTTPException(status_code=status, detail=f"SerpAPI web search error: HTTP {res.status_code} - {error}")

    return _format_results(data)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _format_results(data: dict) -> SearchResults:
    """Turns a SerpAPI response into model context and a list of sources."""
    lines: list[str] = []
    sources: list[SearchSource] = []

    answer = data.get("answer_box") or {}
    answer_text = answer.get("answer") or answer.get("snippet") or answer.get("result")
    if answer_text:
        lines.append(f"Direct answer: {answer_text}")

    graph = data.get("knowledge_graph") or {}
    if graph.get("description"):
        lines.append(f"About {graph.get('title', 'this')}: {graph['description']}")

    for item in (data.get("top_stories") or [])[:MAX_NEWS_RESULTS]:
        source = _source_name(item)
        lines.append(f"News ({source}, {item.get('date', 'recent')}): {item.get('title', '')}")
        sources.append(SearchSource(item.get("title", ""), source, item.get("link", "")))

    for item in (data.get("organic_results") or [])[:MAX_ORGANIC_RESULTS]:
        source = _source_name(item)
        date = f", {item['date']}" if item.get("date") else ""
        lines.append(f"{item.get('title', '')} ({source}{date}): {item.get('snippet', '')}")
        sources.append(SearchSource(item.get("title", ""), source, item.get("link", "")))

    return SearchResults(context="\n".join(lines) or "No results found.", sources=sources)


def _source_name(item: dict) -> str:
    """The publisher's name for a result, falling back to its domain."""
    source = item.get("source")
    if isinstance(source, dict):
        source = source.get("name")
    if source:
        return str(source)
    link = item.get("link", "")
    return link.split("/")[2].removeprefix("www.") if link.count("/") >= 2 else "web"
