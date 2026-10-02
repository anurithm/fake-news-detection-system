import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent))
from src.evidence_verifier import run_verification_pipeline

test_cases = [
    {
        "title": "1. Indian pilot flydubai",
        "text": "Indian pilot tells PM Modi he opened flydubai cockpit door during attack"
    },
    {
        "title": "2. World food prices",
        "text": "World food prices near four-year high in September, UN says"
    },
    {
        "title": "3. Suspicious Claim",
        "text": "Scientists confirm that drinking water after midnight causes humans to lose all their memories."
    }
]

def run_evidence_tests():
    print("Running Evidence Verification Tests...\\n" + "="*50)
    for test in test_cases:
        print(f"\\n[{test['title']}]")
        print(f"Input: {test['text']}")
        try:
            result = run_verification_pipeline(test["text"])
            print(f"Status: {result['status']}")
            print(f"Strength: {result['strength']}")
            print(f"Supporting: {len(result['supporting'])}")
            print(f"Conflicting: {len(result['conflicting'])}")
            print(f"Explanation: {result['explanation']}")
        except Exception as e:
            print(f"FAILED on test {test['title']} - {e}")
            
if __name__ == "__main__":
    run_evidence_tests()
