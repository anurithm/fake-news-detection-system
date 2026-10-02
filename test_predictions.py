import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.predict import predict_text

test_cases = [
    {
        "title": "Suzuki Test",
        "text": "Suzuki asks India suppliers to shorten production week in push for quality"
    },
    {
        "title": "1. New REAL-style headline",
        "text": "Federal Reserve announces 0.25% interest rate hike amidst inflation concerns."
    },
    {
        "title": "2. New FAKE-style headline",
        "text": "SHOCKING: Aliens finally land in New York, Mayor offers them free hot dogs and pizza!"
    },
    {
        "title": "3. Technology news",
        "text": "Apple unveils the new M4 chip with major performance improvements for machine learning tasks."
    },
    {
        "title": "4. Science news",
        "text": "NASA's James Webb Space Telescope captures unprecedented details of a distant exoplanet atmosphere."
    },
    {
        "title": "5. Business news",
        "text": "Global markets rally as tech stocks surge following better than expected quarterly earnings reports."
    },
    {
        "title": "6. Politics/news",
        "text": "The Senate passes the new infrastructure bill after marathon overnight voting session."
    },
    {
        "title": "7. Sports news",
        "text": "Manchester City secures the Premier League title on the final day of the season."
    },
    {
        "title": "8. Health news",
        "text": "New study suggests that a balanced diet consisting of Mediterranean foods significantly reduces heart disease risks."
    },
    {
        "title": "9. A short headline",
        "text": "Local bakery wins national award."
    },
    {
        "title": "10. A long article",
        "text": "The rapid advancement of artificial intelligence has led to both excitement and apprehension across various sectors. While some experts praise the potential for unprecedented economic growth and medical breakthroughs, others issue stark warnings about job displacement and ethical concerns. The debate intensified recently following the release of a highly capable generative model that demonstrated reasoning capabilities previously thought unique to humans. Lawmakers in multiple countries are now scrambling to draft regulations that aim to balance innovation with public safety. Despite the regulatory uncertainty, venture capital investment in AI startups continues to break records, signaling strong market confidence in the technology's long-term viability."
    }
]

def run_tests():
    print("Running dynamic input verification tests...\n" + "="*50)
    for i, test in enumerate(test_cases):
        try:
            result = predict_text(test["text"])
            print(f"\n[{test['title']}]")
            print(f"Input: {test['text'][:100]}...")
            print(f"Prediction: {result['prediction']}")
            print(f"Confidence: {result['confidence'] * 100:.2f}%")
            print(f"Probs: REAL={result['label_probs'].get('REAL', 0):.4f}, FAKE={result['label_probs'].get('FAKE', 0):.4f}")
        except Exception as e:
            print(f"FAILED on test {test['title']} - {e}")

if __name__ == "__main__":
    run_tests()
