"""
LIME Explainer — Local Interpretable Model-Agnostic Explanations.
Approximates model behaviour in the neighbourhood of a specific instance.
"""
import logging
import numpy as np

logger = logging.getLogger(__name__)


class LIMEExplainer:
    def __init__(self, model, feature_names: list, train_data,
                 class_names: list = None, num_features: int = 8):
        self.model = model
        self.feature_names = feature_names
        self.train_data = np.array(train_data)
        self.class_names = class_names or ["Negative", "Positive"]
        self.num_features = num_features
        self._explainer = None
        self._init_explainer()

    def _init_explainer(self):
        try:
            import lime.lime_tabular
            self._explainer = lime.lime_tabular.LimeTabularExplainer(
                self.train_data,
                feature_names=self.feature_names,
                class_names=self.class_names,
                mode="classification",
                random_state=42,
            )
            logger.info("LIME explainer initialised.")
        except ImportError:
            logger.warning("lime not installed — LIME explanations unavailable.")

    def explain(self, instance) -> dict:
        """Return LIME explanation for a single instance."""
        if self._explainer is None:
            return {}
        instance = np.array(instance).flatten()
        exp = self._explainer.explain_instance(
            instance,
            self.model.predict_proba,
            num_features=self.num_features,
        )
        feature_weights = exp.as_list()
        return {
            "feature_weights": feature_weights,
            "score": exp.score,                         # local fidelity R²
            "intercept": exp.intercept[1],
            "prediction_local": exp.local_pred[0],
            "narrative": self._to_narrative(feature_weights, instance),
        }

    def _to_narrative(self, weights: list, instance) -> str:
        """Convert LIME weights to a plain-English clinical sentence."""
        pos = [(f, w) for f, w in weights if w > 0]
        neg = [(f, w) for f, w in weights if w < 0]
        pos_str = ", ".join(f[0] for f in pos[:3]) if pos else "none"
        neg_str = ", ".join(f[0] for f in neg[:2]) if neg else "none"
        return (
            f"Risk-increasing factors: {pos_str}. "
            f"Protective factors: {neg_str}."
        )

    def save_html(self, instance, output_path: str = "logs/lime_explanation.html"):
        """Save LIME explanation as a standalone HTML file."""
        if self._explainer is None:
            return
        instance = np.array(instance).flatten()
        exp = self._explainer.explain_instance(
            instance, self.model.predict_proba, num_features=self.num_features)
        exp.save_to_file(output_path)
        logger.info(f"LIME HTML saved to {output_path}")
