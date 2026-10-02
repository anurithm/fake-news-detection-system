"""
test_predict.py
---------------
Tests for the prediction pipeline.
Skips gracefully when models have not been trained yet.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.predict import models_ready, get_model_info


@pytest.mark.skipif(not models_ready(), reason="Models not trained yet - run python src/train.py first")
class TestPredictText:
    def test_real_prediction_structure(self):
        from src.predict import predict_text
        result = predict_text(
            "Scientists discover new planet orbiting a distant star using the James Webb Space Telescope."
        )
        assert "prediction" in result
        assert "confidence" in result
        assert "model_name" in result
        assert "label_probs" in result
        assert result["prediction"] in {"REAL", "FAKE"}
        assert 0.0 <= result["confidence"] <= 1.0

    def test_confidence_is_float(self):
        from src.predict import predict_text
        result = predict_text("Government announces new tax policies for the next fiscal year.")
        assert isinstance(result["confidence"], float)

    def test_empty_text_raises(self):
        from src.predict import predict_text
        with pytest.raises(ValueError):
            predict_text("")

    def test_whitespace_only_raises(self):
        from src.predict import predict_text
        with pytest.raises(ValueError):
            predict_text("   ")

    def test_label_probs_sum_near_one(self):
        from src.predict import predict_text
        result = predict_text("Local government approves new budget for infrastructure projects this year.")
        total = sum(result["label_probs"].values())
        assert abs(total - 1.0) < 0.02  # allow small floating-point error


class TestModelsReady:
    def test_returns_bool(self):
        val = models_ready()
        assert isinstance(val, bool)


class TestGetModelInfo:
    def test_returns_dict_or_none(self):
        info = get_model_info()
        assert info is None or isinstance(info, dict)

    @pytest.mark.skipif(not models_ready(), reason="Models not trained yet")
    def test_trained_info_has_keys(self):
        info = get_model_info()
        assert info is not None
        assert "best_model_name" in info
        assert "best_model_f1" in info
