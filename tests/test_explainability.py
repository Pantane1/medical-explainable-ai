"""
Tests for SHAP, LIME, counterfactual, and feature importance modules.
Run with: pytest tests/test_explainability.py -v
"""
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.explainable_model import ExplainableMedicalAI

FEATURE_NAMES = ["age", "sex", "cp", "bp", "chol", "fbs", "thalach", "exang", "oldpeak"]


@pytest.fixture(scope="module")
def components():
    rng = np.random.default_rng(1)
    X   = rng.random((300, 9))
    y   = (X[:, 0] + X[:, 2] > 1.0).astype(int)
    model = ExplainableMedicalAI(model_type="decision_tree")
    model.train(X, y, FEATURE_NAMES)
    return model, X, y


def test_feature_importance_analyzer(components):
    from src.explainability.feature_importance import FeatureImportanceAnalyzer
    model, X, y = components
    fia = FeatureImportanceAnalyzer(model.model, FEATURE_NAMES)
    imp = fia.intrinsic_importance()
    assert len(imp) == len(FEATURE_NAMES)
    assert all(v >= 0 for v in imp.values())


def test_permutation_importance(components):
    from src.explainability.feature_importance import FeatureImportanceAnalyzer
    model, X, y = components
    fia = FeatureImportanceAnalyzer(model.model, FEATURE_NAMES)
    perm = fia.permutation_importance(X, y, n_repeats=3)
    assert len(perm) == len(FEATURE_NAMES)


def test_counterfactual_generation(components):
    from src.explainability.counterfactuals import CounterfactualExplainer
    model, X, _ = components
    cfe = CounterfactualExplainer(model.model, FEATURE_NAMES)
    instance = X[0]
    cfs = cfe.generate(instance, n_counterfactuals=2)
    # Either CFs are found or instance is already target class
    assert isinstance(cfs, list)
    for cf in cfs:
        assert "changed_feature" in cf
        assert "original_value" in cf
        assert "counterfactual_value" in cf


def test_counterfactual_text(components):
    from src.explainability.counterfactuals import CounterfactualExplainer
    model, X, _ = components
    cfe = CounterfactualExplainer(model.model, FEATURE_NAMES)
    cfs = cfe.generate(X[0], n_counterfactuals=1)
    text = cfe.to_clinical_text(cfs)
    assert isinstance(text, str)
    assert len(text) > 0
