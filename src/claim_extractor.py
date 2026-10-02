import urllib.parse
from typing import List, Dict

def extract_primary_claim(text: str) -> str:
    """
    Very lightweight extraction.
    If it's short, it's the claim itself.
    If it's long, taking the first sentence or two.
    """
    text = text.strip()
    if not text:
        return ""
    
    # Simple heuristic to extract the first substantial sentence
    sentences = text.replace('\n', ' ').split('. ')
    if len(sentences) == 1:
        return sentences[0][:200]
    
    claim = sentences[0]
    if len(claim.split()) < 5 and len(sentences) > 1:
        claim += ". " + sentences[1]
    
    # Clean up
    return claim[:300].strip()

def build_search_query(claim: str) -> str:
    """
    Converts a claim into an effective keyword search query.
    Removes common stop words for better search density if needed, 
    but modern search engines handle raw text well.
    """
    # Limit length so it doesn't break limits
    return claim[:100]
