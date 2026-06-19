# agent/tools/web_search.py
import os
from exa_py import Exa
from langchain_core.tools import tool

_exa_client = None

def _get_exa():
    global _exa_client
    if _exa_client is None:
        from config import EXA_API_KEY
        _exa_client = Exa(api_key=EXA_API_KEY)
    return _exa_client

@tool
def web_search(query: str, num_results: int = 5) -> str:
    """
    Search the web using Exa deep research API.
    Use for: locating datasets, surveying published baselines, domain knowledge.
    Returns a formatted string of results.
    """
    exa = _get_exa()
    results = exa.search_and_contents(
        query,
        num_results=num_results,
        use_autoprompt=True,
        text={"max_characters": 1000},
    )
    formatted = []
    for r in results.results:
        formatted.append(f"**{r.title}**\n{r.url}\n{r.text[:500]}\n")
    return "\n---\n".join(formatted) if formatted else "No results found."
