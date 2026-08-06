"""Custom tools handed to the deep agent.

`internet_search` is the Tavily tool from notebooks 1 and 4 — the one custom
tool the course adds on top of the built-in planning / filesystem / task tools.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Literal


@lru_cache(maxsize=1)
def _tavily_client():
    from tavily import TavilyClient

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        msg = "TAVILY_API_KEY is not set — add it to .env to enable web search."
        raise RuntimeError(msg)
    return TavilyClient(api_key=api_key)


def internet_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
) -> dict[str, Any]:
    """Run a web search and return the raw Tavily results.

    Args:
        query: What to search for.
        max_results: How many results to return (keep it small; write anything
            bulky to a file instead of carrying it in the conversation).
        topic: Search vertical to bias the results towards.
        include_raw_content: Include the scraped page text. Only ask for this
            when you intend to save it to a file.
    """
    return _tavily_client().search(
        query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic=topic,
    )
