import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Override base URL to forcefully cause a failure
os.environ["OLLAMA_BASE_URL"] = "http://localhost:59999"

from src.ollama_client import get_ai_explanation

def test_ollama_failure():
    result = get_ai_explanation("test", "REAL", 0.99, {})
    if result is None:
        print("\nSUCCESS! Handled failure gracefully by returning None")
    else:
        print("\nFAILED! Unexpectedly returned some result instead of None")

if __name__ == "__main__":
    test_ollama_failure()
