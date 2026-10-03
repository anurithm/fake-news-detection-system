import json
from src.claim_extractor import extract_primary_claim, build_search_query
from src.web_search import search_live_evidence
from src.evidence_ranker import assess_evidence
from src.evidence_verifier import run_verification_pipeline
from src.predict import predict_text

tests = [
    "India’s Chandrayaan-3 successfully landed near the Moon’s south pole.",
    "India won the cricket match after scoring more runs than its opponent.",
    "Scientists successfully confirmed that the earth is actually perfectly flat and rests on turtles.",
    "Manchester United formally announced they are signing Lionel Messi and Cristiano Ronaldo on a free transfer yesterday.",
    "SpaceX successfully launched its massive Starship rocket on a test flight from its Texas facility.",
    "The new 5G network towers are secretly transmitting mind-control radio waves into the water supply."
]

print("Running dynamic evidence verifications...\n")
for t in tests:
    print("-" * 60)
    print(f"INPUT: {t}")
    
    # Run the exact pipeline used in app.py
    ev_result = run_verification_pipeline(t)
    
    # Check the underlying ML just for reference context
    ml = predict_text(t)
    
    print(f"\nML Pipeline Predicts: {ml['prediction']} ({ml['confidence']*100:.1f}%)")
    
    status = ev_result['status']
    print(f"EVIDENCE EVALUATION STATUS: {status}")
    print(f"CLAIM EXTRACTED: {ev_result['claim']}")
    print(f"EXPLANATION: {ev_result['explanation']}")
    
    final = "REAL" if status == "VERIFIED" else "FAKE" if status == "CONTRADICTED" else "UNVERIFIED"
    print(f"==> FINAL APP VERDICT: {final}")
    print("\n")
