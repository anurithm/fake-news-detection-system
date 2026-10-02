from typing import Dict, Any
import logging
from src.claim_extractor import extract_primary_claim, build_search_query
from src.web_search import search_live_evidence
from src.evidence_ranker import assess_evidence

logger = logging.getLogger(__name__)

def run_verification_pipeline(text: str) -> Dict[str, Any]:
    """
    Runs the full evidence verification pipeline:
    1. Extract main claim.
    2. Build query.
    3. Search web/RSS.
    4. Rank/Compare evidence.
    """
    
    # 1. Claim Extraction
    claim = extract_primary_claim(text)
    if not claim:
        return {
            "status": "UNVERIFIED",
            "strength": "NONE",
            "claim": "No clear claim extracted.",
            "explanation": "Could not identify a clear factual claim to verify.",
            "supporting": [],
            "conflicting": []
        }
        
    # 2. Search
    query = build_search_query(claim)
    evidence_list = search_live_evidence(query, max_results=6)
    
    # 3. Assess
    status, strength, supporting, conflicting, explanation = assess_evidence(claim, evidence_list)
    
    return {
        "status": status,
        "strength": strength,
        "claim": claim,
        "explanation": explanation,
        "supporting": supporting,
        "conflicting": conflicting
    }
