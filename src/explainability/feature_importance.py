"""
Feature Importance — global and permutation-based importance methods.
Complements SHAP for model-level interpretability.
"""
import logging
import numpy as np

logger = logging.getLogger(__name__)


class FeatureImportanceAnalyzer:
    def __init__(self, model, feature_names: list):
        self.model = model
        self.feature_names = feature_names

    def intrinsic_importance(self) -> dict:
        """Return built-in feature importances (trees) or coefficients (linear)."""
        if hasattr(self.model, "feature_importances_"):
            raw = self.model.feature_importances_
        elif hasattr(self.model, "coef_"):
            raw = np.abs(self.model.coef_[0])
        else:
            logger.warning("Model has no intrinsic feature importance.")
            return {}
        total = raw.sum()
        normalized = raw / total if total > 0 else raw
        ranked = sorted(
            zip(self.feature_names, normalized),
            key=lambda x: x[1], reverse=True
        )
        return {f: round(float(v), 5) for f, v in ranked}

    def permutation_importance(self, X, y, n_repeats: int = 10) -> dict:
        """Compute permutation-based importance on a held-out set."""
        from sklearn.inspection import permutation_importance
        result = permutation_importance(self.model, X, y,
                                        n_repeats=n_repeats, random_state=42)
        ranked = sorted(
            zip(self.feature_names, result.importances_mean),
            key=lambda x: x[1], reverse=True
        )
        return {f: round(float(v), 5) for f, v in ranked}

    def summary(self, X=None, y=None) -> dict:
        out = {"intrinsic": self.intrinsic_importance()}
        if X is not None and y is not None:
            out["permutation"] = self.permutation_importance(X, y)
        return out
