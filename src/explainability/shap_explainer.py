"""
SHAP Explainer — computes SHAP values for tree-based and linear models.
Provides both per-instance and global feature importance.
"""
import logging
import numpy as np

logger = logging.getLogger(__name__)


class SHAPExplainer:
    def __init__(self, model, feature_names: list, train_data):
        self.model = model
        self.feature_names = feature_names
        self.train_data = train_data
        self._explainer = None
        self._init_explainer()

    def _init_explainer(self):
        try:
            import shap
            if hasattr(self.model, "feature_importances_"):
                # Tree-based model
                self._explainer = shap.TreeExplainer(self.model)
            else:
                # Linear or other
                self._explainer = shap.LinearExplainer(self.model, self.train_data)
            logger.info("SHAP explainer initialised.")
        except ImportError:
            logger.warning("shap not installed — SHAP explanations unavailable.")

    def explain(self, instance) -> dict:
        """Return SHAP values for a single instance as a named dict."""
        if self._explainer is None:
            return {}
        import shap
        instance = np.atleast_2d(instance)
        shap_values = self._explainer.shap_values(instance)

        # For binary classifiers, shap_values is a list [class0, class1]
        if isinstance(shap_values, list):
            vals = shap_values[1][0]
        else:
            vals = shap_values[0]

        result = {
            "shap_values": dict(zip(self.feature_names, vals.tolist())),
            "expected_value": float(
                self._explainer.expected_value[1]
                if isinstance(self._explainer.expected_value, (list, np.ndarray))
                else self._explainer.expected_value
            ),
            "top_features": self._top_features(vals),
        }
        return result

    def _top_features(self, shap_vals, n: int = 5) -> list:
        idx = np.argsort(np.abs(shap_vals))[::-1][:n]
        return [
            {"feature": self.feature_names[i],
             "shap_value": round(float(shap_vals[i]), 4),
             "direction": "increases_risk" if shap_vals[i] > 0 else "decreases_risk"}
            for i in idx
        ]

    def global_importance(self, X) -> dict:
        """Compute mean absolute SHAP values across a dataset."""
        if self._explainer is None:
            return {}
        import shap
        shap_values = self._explainer.shap_values(X)
        if isinstance(shap_values, list):
            vals = shap_values[1]
        else:
            vals = shap_values
        mean_abs = np.mean(np.abs(vals), axis=0)
        return dict(zip(self.feature_names, mean_abs.tolist()))

    def save_force_plot(self, instance, output_path: str = "logs/shap_force.html"):
        """Save an interactive SHAP force plot to HTML."""
        try:
            import shap
            import matplotlib
            matplotlib.use("Agg")
            exp = self.explain(instance)
            shap.save_html(output_path,
                           shap.force_plot(exp["expected_value"],
                                           list(exp["shap_values"].values()),
                                           feature_names=self.feature_names))
            logger.info(f"SHAP force plot saved to {output_path}")
        except Exception as e:
            logger.error(f"Could not save force plot: {e}")
