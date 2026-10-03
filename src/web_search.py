import feedparser
import urllib.parse
from typing import List, Dict
import re
import logging
from duckduckgo_search import DDGS

logger = logging.getLogger(__name__)

def search_live_evidence(query: str, max_results: int = 8) -> List[Dict]:
    """
    Searches DuckDuckGo and Google News RSS for live evidence on a query.
    Returns a unified list of sources.
    """
    results = []
    
    # DuckDuckGo Search (HTML/Web)
    try:
        with DDGS() as ddgs:
            # We use text search
            ddg_results = list(ddgs.text(query, max_results=max_results))
            for r in ddg_results:
                results.append({
                    "title": r.get('title', ''),
                    "url": r.get('href', ''),
                    "source": _extract_domain(r.get('href', '')),
                    "snippet": r.get('body', ''),
                    "date": "Unknown"
                })
    except Exception as e:
        logger.warning(f"DDG search failed: {e}")

    # Fallback/Additional: Google News RSS
    if len(results) < max_results:
        try:
            encoded_query = urllib.parse.quote(query)
            rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
            feed = feedparser.parse(rss_url)
            for entry in feed.entries[:max_results]:
                results.append({
                    "title": entry.title,
                    "url": entry.link,
                    "source": entry.source.title if hasattr(entry, 'source') else "Google News",
                    "snippet": entry.title, # RSS snippets aren't always available nicely
                    "date": entry.published if hasattr(entry, 'published') else "Unknown"
                })
        except Exception as e:
            logger.warning(f"News RSS search failed: {e}")
            
    # Deduplicate by URL
    seen_urls = set()
    unique_results = []
    for r in results:
        if r['url'] not in seen_urls:
            seen_urls.add(r['url'])
            unique_results.append(r)
            if len(unique_results) >= max_results:
                break
    
    return unique_results

def _extract_domain(url: str) -> str:
    try:
        from urllib.parse import urlparse
        return urlparse(url).netloc.replace('www.', '')
    except:
        return 'Unknown Source'
