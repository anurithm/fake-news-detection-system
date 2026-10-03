import os
import requests
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# Default config using standard env variables or defaults
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3:latest")
OLLAMA_TIMEOUT = int(os.environ.get("OLLAMA_TIMEOUT", "30"))

SYSTEM_PROMPT = (
    "You are an explanation assistant for a fake-news detection system. "
    "Do not independently determine whether a claim is true. "
    "Explain the existing machine-learning prediction and evidence verification using only the information supplied to you. "
    "Do not invent facts, sources, URLs, dates, evidence, or browsing activity. "
    "Clearly distinguish between model prediction, retrieved evidence, and uncertainty."
)


def get_ai_explanation(
    news_text: str,
    prediction: str,
    confidence: float,
    evidence_result: Optional[Dict[str, Any]] = None
) -> Optional[str]:
    """
    Calls the local Ollama API to generate an explanation of the ML and evidence results.
    Returns the explanation string, or None if the request fails.
    """
    endpoint = f"{OLLAMA_BASE_URL}/api/generate"
    
    # Construct the user prompt integrating all the gathered verification data.
    prompt_lines = [
        "Please provide an AI Explanation based on the following results:",
        f"Input News Text: {news_text}",
        f"ML Prediction: {prediction}",
        f"ML Confidence: {confidence*100:.2f}%",
    ]
    
    if evidence_result:
        prompt_lines.append(f"Evidence Status: {evidence_result.get('status', 'Unknown')}")
        prompt_lines.append(f"Evidence Strength: {evidence_result.get('strength', 'Unknown')}")
        prompt_lines.append(f"Main Claim Analyzed: {evidence_result.get('claim', 'Unknown')}")
        
        supporting = evidence_result.get('supporting', [])
        if supporting:
            prompt_lines.append("Supporting Evidence Snippets:")
            for s in supporting:
                prompt_lines.append(f"- {s.get('source', 'Unknown')}: {s.get('snippet', '')}")
                
        conflicting = evidence_result.get('conflicting', [])
        if conflicting:
            prompt_lines.append("Conflicting Evidence Snippets:")
            for c in conflicting:
                prompt_lines.append(f"- {c.get('source', 'Unknown')}: {c.get('snippet', '')}")
    
    prompt = "\\n".join(prompt_lines)
    
    payload = {
        "model": OLLAMA_MODEL,
        "system": SYSTEM_PROMPT,
        "prompt": prompt,
        "stream": False
    }
    
    try:
        response = requests.post(endpoint, json=payload, timeout=OLLAMA_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        return data.get("response", "")
    except requests.exceptions.RequestException as e:
        logger.error(f"Ollama API request failed: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error when calling Ollama API: {e}")
        return None
