import os
import requests
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3:latest")
OLLAMA_TIMEOUT = int(os.environ.get("OLLAMA_TIMEOUT", "30"))

def extract_primary_claim(text: str) -> str:
    """
    Extracts the primary factual claim from the provided text.
    If the text is short, it is returned as the claim.
    If it is long, it uses the local discrete Ollama LLM to synthesize the core factual assertion dynamically.
    """
    text = text.strip()
    if not text:
        return ""
        
    words = text.split()
    if len(words) <= 30:
        # For typical headlines, the headline is the claim.
        return text
        
    # For longer text, attempt LLM extraction
    try:
        endpoint = f"{OLLAMA_BASE_URL}/api/generate"
        prompt = (
            "You are a factual claim extraction tool. Extract the single main factual claim from the following news article text. "
            "Output ONLY the claim as a single concise sentence. Do not provide explanations or supplementary text.\n\n"
            f"Article Text: {text}"
        )
        
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False
        }
        
        response = requests.post(endpoint, json=payload, timeout=OLLAMA_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        claim = data.get("response", "").strip()
        
        # If Ollama hallucinates massive amounts of text, fallback to truncation defensively
        if claim and len(claim.split()) <= 50:
            return claim.strip('"\'')
            
    except Exception as e:
        logger.warning(f"Ollama claim extraction failed or timed out: {e}")
        
    # Heuristic fallback
    sentences = text.replace('\n', ' ').split('. ')
    if len(sentences) == 1:
        return sentences[0][:200]
    
    claim = sentences[0]
    if len(claim.split()) < 5 and len(sentences) > 1:
        claim += ". " + sentences[1]
    
    return claim[:300].strip()

def build_search_query(claim: str) -> str:
    """
    Converts a claim into an effective keyword search query dynamically.
    Instead of passing whole sentences, we strip common stop words to
    ensure search providers like DDG retrieve relevant specific facts.
    """
    import string
    
    stops = {"a", "an", "the", "and", "or", "but", "is", "are", "was", "were", 
             "in", "on", "at", "to", "for", "with", "by", "about", "against",
             "between", "into", "through", "during", "before", "after", "above",
             "below", "from", "up", "down", "of", "off", "over", "under", "again",
             "further", "then", "once", "here", "there", "when", "where", "why",
             "how", "all", "any", "both", "each", "few", "more", "most", "other",
             "some", "such", "no", "nor", "not", "only", "own", "same", "so", 
             "than", "too", "very", "can", "will", "just", "should", "now",
             "successfully", "its", "it", "they", "them", "their", "this", "that"}
             
    clean = claim.translate(str.maketrans('', '', string.punctuation)).lower()
    words = clean.split()
    
    keywords = [w for w in words if w not in stops and len(w) > 2]
    
    # Take up to 6 of the most distinct keywords to form the query
    out_query = " ".join(keywords[:6])
    if not out_query:
        return claim[:100].strip()
        
    return out_query

