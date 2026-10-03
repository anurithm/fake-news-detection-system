from typing import List, Dict, Tuple
import os
import requests
import json
import re

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3:latest")
OLLAMA_TIMEOUT = int(os.environ.get("OLLAMA_TIMEOUT", "60")) # slightly longer for harder analysis

def assess_evidence(claim: str, evidence_list: List[Dict]) -> Tuple[str, str, List[Dict], List[Dict], str]:
    """
    Delegates assessment strictly to Ollama to determine verification status inherently from independent search results.
    Returns: (Status, Strength, Supporting, Conflicting, Explanation)
    """
    if not evidence_list:
         return "UNVERIFIED", "NONE", [], [], "No sufficiently relevant and reliable evidence was found to support or contradict this claim. This does NOT automatically mean the claim is false."
    
    return _assess_with_ollama(claim, evidence_list)

def _assess_with_ollama(claim: str, evidence_list: List[Dict]):
    supporting = []
    conflicting = []
    
    # Provide the exact snippets to the LLM to judge strictly
    prompt = (
        "You are an unbiased fact-checker analyzing evidence against a claim.\n"
        f"CLAIM: {claim}\n\nEVIDENCE:\n"
    )
    for i, ev in enumerate(evidence_list):
        prompt += f"[{i}] SOURCE: {ev['source']} | TEXT: {ev['title']} - {ev['snippet']}\n"
        
    prompt += (
        "\nIMPORTANT RULES:\n"
        "1. Do NOT use outside knowledge. The verdict MUST be based purely on the provided EVIDENCE.\n"
        "2. If multiple independent sources confidently confirm the specific facts of the claim, the verdict is VERIFIED.\n"
        "3. If evidence securely states the claim is false, debunked, misleading, or a hoax, the verdict is CONTRADICTED.\n"
        "4. If there is insufficient evidence, missing confirmation, or the search results are unrelated topics, the verdict is UNVERIFIED.\n"
        "5. If reliable sources genuinely conflict dynamically, return MIXED.\n"
        "\nYou MUST respond with ONLY a valid, parseable JSON block containing exactly three keys:\n"
        '- "status" (must be exactly one of: "VERIFIED", "CONTRADICTED", "MIXED", "UNVERIFIED")\n'
        '- "strength" (must be "STRONG", "MODERATE", "WEAK", or "NONE")\n'
        '- "explanation" (a brief summary of why based on the evidence snippets, noting if evidence conflicted or was insufficient)\n'
        "DO NOT add markdown formatting blocks, commentary, or text outside of the raw JSON object."
    )
    
    try:
        endpoint = f"{OLLAMA_BASE_URL}/api/generate"
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "format": "json"
        }
        
        response = requests.post(endpoint, json=payload, timeout=OLLAMA_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        
        text_resp = data.get("response", "").strip()
        
        # Clean any potential markdown wrapping from LLM
        if text_resp.startswith("```json"):
            text_resp = text_resp[7:]
        if text_resp.startswith("```"):
            text_resp = text_resp[3:]
        if text_resp.endswith("```"):
            text_resp = text_resp[:-3]
            
        parsed = json.loads(text_resp.strip())
        
        status = parsed.get("status", "UNVERIFIED").strip().upper()
        if status not in ["VERIFIED", "CONTRADICTED", "MIXED", "UNVERIFIED"]:
            status = "UNVERIFIED"
            
        strength = parsed.get("strength", "WEAK").strip().upper()
        explanation = parsed.get("explanation", "The AI determined this status based on the retrieved evidence.")
        
        # The LLM doesn't easily return specific index splits dependably, so we assign lists based on text overlap dynamically
        # Since the LLM rendered the verdict overall, we categorize snippets loosely for the UI
        # A true implementation would have the LLM return indexes, but we use a regex fallback for snippet classification for UI display.
        for ev in evidence_list:
            text = (ev['title'] + " " + ev['snippet']).lower()
            negations = ["false", "debunked", "fake", "hoax", "incorrect", "misleading", "rumor", "no evidence"]
            if status in ["CONTRADICTED", "MIXED"] and any(neg in text for neg in negations):
                conflicting.append(ev)
            elif status in ["VERIFIED", "MIXED"]:
                supporting.append(ev)
            else:
                # If unverified, they aren't strong enough.
                pass
                
        # If the LLM declared contradictory but heuristic didn't catch the negations, place them all in conflicting just for display.
        if status == "CONTRADICTED" and not conflicting:
            conflicting = evidence_list
        if status == "VERIFIED" and not supporting:
            supporting = evidence_list
            
        return status, strength, supporting, conflicting, explanation

    except Exception as e:
        # Fallback if the JSON parsing or LLM connection strictly fails
        return _assess_deterministic(claim, evidence_list)

def _assess_deterministic(claim: str, evidence_list: List[Dict]):
    supporting = []
    conflicting = []
    
    claim_words = set(w for w in claim.lower().split() if len(w) > 3)
    
    for ev in evidence_list:
        text = (ev['title'] + " " + ev['snippet']).lower()
        negations = ["false", "debunked", "fake", "hoax", "incorrect", "misleading", "no evidence"]
        if any(neg in text for neg in negations):
            conflicting.append(ev)
        else:
            overlap = len([w for w in text.split() if w in claim_words])
            if overlap > 0:
                supporting.append(ev)
                
    if conflicting and not supporting:
        return "CONTRADICTED", "STRONG", [], conflicting, "Fallback rule evaluated reliable sources as contradicting the claim."
    if conflicting and supporting:
        return "MIXED", "MODERATE", supporting, conflicting, "Fallback rule evaluated sources as mixed."
    if supporting:
        if len(supporting) > 1:
            return "VERIFIED", "STRONG", supporting, [], "Fallback rule confirmed multiple sources matching the claim."
        else:
            # Single source shouldn't be auto-verified
            return "UNVERIFIED", "WEAK", supporting, [], "Only one matching source was found. This is insufficient to definitively verify."
            
    return "UNVERIFIED", "WEAK", [], [], "Insufficient clear evidence was found regarding this exact claim."
