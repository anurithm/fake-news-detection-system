import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ollama_client import get_ai_explanation

def test_ollama():
    news_text = "India’s Chandrayaan-3 successfully landed near the Moon’s south pole."
    prediction = "REAL"
    confidence = 0.94
    evidence_result = {
        "status": "VERIFIED",
        "strength": "STRONG",
        "claim": "Chandrayaan-3 landed on Moon's south pole",
        "supporting": [{"source": "Space News", "snippet": "ISRO's Chandrayaan-3 successfully touched down."}],
        "conflicting": []
    }
    
    print("Sending request to Ollama...")
    result = get_ai_explanation(news_text, prediction, confidence, evidence_result)
    
    if result:
        print("\nSUCCESS! Response from Ollama:\n")
        print(result)
    else:
        print("\nFAILED: Received None or an error.")

if __name__ == "__main__":
    test_ollama()
