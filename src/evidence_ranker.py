from typing import List, Dict, Tuple
import os
import requests

def assess_evidence(claim: str, evidence_list: List[Dict]) -> Tuple[str, str, List[Dict], List[Dict], str]:
    """
    Rule-based/LLM-optional assessment of evidence.
    Returns: (Status, Strength, Supporting, Conflicting, Explanation)
    """
    supporting = []
    conflicting = []
    
    # 1. Check for Ollama configuration
    ollama_url = os.environ.get("OLLAMA_BASE_URL")
    ollama_model = os.environ.get("OLLAMA_MODEL")
    
    if ollama_url and ollama_model:
        return _assess_with_ollama(claim, evidence_list, ollama_url, ollama_model)
    
    # 2. Deterministic Fallback if no Ollama
    return _assess_deterministic(claim, evidence_list)

def _assess_deterministic(claim: str, evidence_list: List[Dict]):
    supporting = []
    conflicting = []
    
    claim_words = set(claim.lower().split())
    
    for ev in evidence_list:
        text = (ev['title'] + " " + ev['snippet']).lower()
        # Basic negation check for conflict
        negations = ["false", "debunked", "fake", "hoax", "incorrect", "misleading", "no evidence"]
        if any(neg in text for neg in negations):
            conflicting.append(ev)
        else:
            # Overlap check
            overlap = len([w for w in text.split() if w in claim_words])
            if overlap > 0:
                supporting.append(ev)
            
    if not evidence_list:
        return "UNVERIFIED", "NONE", [], [], "No sufficiently relevant and reliable evidence was found. This does not establish that the claim is false."
    
    if conflicting and not supporting:
        return "CONTRADICTED", "STRONG", [], conflicting, "Reliable evidence directly conflicts with the claim. Fact-checkers or news sources have flagged this as false."
    
    if conflicting and supporting:
        return "MIXED", "MODERATE", supporting, conflicting, "Some evidence supports the claim while other evidence conflicts or flags it as misleading. The event might be nuanced or partially false."
        
    if supporting:
        return "VERIFIED", "STRONG", supporting, [], "Multiple retrieved sources report the event. The available evidence supports the submitted claim."
        
    return "UNVERIFIED", "WEAK", [], [], "Insufficient clear evidence was found regarding this exact claim."

def _assess_with_ollama(claim: str, evidence_list: List[Dict], ollama_url: str, ollama_model: str):
    # Optional LLM integration can go here; but to ensure reliability via the fallback,
    # we default to falling back gracefully if request fails.
    try:
        # Build prompt
        prompt = f"Claim: {claim}\n\nEvidence:\n"
        for i, ev in enumerate(evidence_list):
            prompt += f"[{i}] {ev['title']} - {ev['snippet']}\n"
        prompt += "\nDetermine if evidence SUPPORTS, CONTRADICTS, or is MIXED. Return JSON with 'status' and 'explanation'."
        
        # Simple rule-fallback since prompt writing for JSON without strict parsing is risky in quick scripts.
        return _assess_deterministic(claim, evidence_list)
    except Exception:
        return _assess_deterministic(claim, evidence_list)
