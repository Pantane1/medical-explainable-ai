"""
Tests for the model training and prediction pipeline.
Run with: pytest tests/test_models.py -v
"""
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.explainable_model import ExplainableMedicalAI

FEATURE_NAMES = ["age", "sex", "cp", "bp", "chol", "fbs", "thalach", "exang", "oldpeak"]


@pytest.fixture
def trained_model():
    rng = np.random.default_rng(0)
    X = rng.random((200, 9))
    y = (X[:, 0] + X[:, 2] > 1.0).astype(int)
    model = ExplainableMedicalAI(model_type="decision_tree")
    model.train(X, y, FEATURE_NAMES)
    return model, X, y


def test_supported_model_types():
    for mt in ("decision_tree", "random_forest", "logistic"):
        m = ExplainableMedicalAI(model_type=mt)
        assert m.model is not None


def test_invalid_model_type():
    with pytest.raises(ValueError):
        ExplainableMedicalAI(model_type="xgboost")


def test_train_and_predict(trained_model):
    model, X, y = trained_model
    preds = model.predict(X[:5])
    assert preds.shape == (5,)
    assert set(preds).issubset({0, 1})


def test_predict_proba(trained_model):
    model, X, _ = trained_model
    probas = model.predict_proba(X[:10])
    assert probas.shape == (10,)
    assert np.all((probas >= 0) & (probas <= 1))


def test_evaluate(trained_model):
    model, X, y = trained_model
    metrics = model.evaluate(X, y)
    assert "accuracy" in metrics
    assert "roc_auc" in metrics
    assert 0 <= metrics["accuracy"] <= 1
    assert 0 <= metrics["roc_auc"] <= 1


def test_feature_importance(trained_model):
    model, _, _ = trained_model
    imp = model.feature_importance()
    assert len(imp) == len(FEATURE_NAMES)
    assert all(v >= 0 for v in imp.values())


def test_save_and_load(trained_model, tmp_path):
    model, X, _ = trained_model
    path = str(tmp_path / "model.pkl")
    model.save(path)
    loaded = ExplainableMedicalAI.load(path)
    np.testing.assert_array_equal(model.predict(X[:3]), loaded.predict(X[:3]))


def test_single_instance_prediction(trained_model):
    model, X, _ = trained_model
    instance = X[0]
    pred = model.predict(instance)
    assert pred.shape == (1,)
