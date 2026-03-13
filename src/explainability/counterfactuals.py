"""
Counterfactual Explanations — "What would need to change to alter the prediction?"
Generates minimal actionable changes for clinicians.
"""
import logging
import numpy as np

logger = logging.getLogger(__name__)


class CounterfactualExplainer:
    def __init__(self, model, feature_names: list, feature_ranges: dict = None):
        self.model = model
        self.feature_names = feature_names
        # feature_ranges: {feature_name: (min, max)} for clipping generated CFs
        self.feature_ranges = feature_ranges or {}

    def generate(self, instance, target_class: int = 0,
                 n_counterfactuals: int = 3, step_pct: float = 0.05) -> list:
        """
        Simple greedy counterfactual search:
        perturb one feature at a time until the predicted class flips.
        Returns up to n_counterfactuals minimal-change alternatives.
        """
        instance = np.array(instance, dtype=float).flatten()
        current_pred = int(self.model.predict(instance.reshape(1, -1))[0])

        if current_pred == target_class:
            logger.info("Instance already predicted as target class.")
            return []

        results = []
        # Try perturbing each feature in both directions
        for idx, fname in enumerate(self.feature_names):
            for direction in [1, -1]:
                cf = instance.copy()
                fmin, fmax = self.feature_ranges.get(fname, (-np.inf, np.inf))
                step = abs(instance[idx]) * step_pct + 1e-3

                for _ in range(100):          # max 100 steps per feature
                    cf[idx] += direction * step
                    cf[idx] = np.clip(cf[idx], fmin, fmax)
                    new_pred = int(self.model.predict(cf.reshape(1, -1))[0])
                    if new_pred == target_class:
                        delta = cf[idx] - instance[idx]
                        results.append({
                            "changed_feature": fname,
                            "original_value": round(float(instance[idx]), 3),
                            "counterfactual_value": round(float(cf[idx]), 3),
                            "delta": round(float(delta), 3),
                            "direction": "increase" if delta > 0 else "decrease",
                            "counterfactual_instance": cf.tolist(),
                        })
                        break

            if len(results) >= n_counterfactuals:
                break

        # Sort by smallest absolute delta
        results.sort(key=lambda r: abs(r["delta"]))
        return results[:n_counterfactuals]

    def to_clinical_text(self, counterfactuals: list) -> str:
        if not counterfactuals:
            return "No actionable counterfactual found."
        lines = ["To change the prediction, consider the following adjustments:"]
        for i, cf in enumerate(counterfactuals, 1):
            lines.append(
                f"  {i}. {cf['direction'].capitalize()} {cf['changed_feature']} "
                f"from {cf['original_value']} to {cf['counterfactual_value']} "
                f"(Δ {cf['delta']:+.3f})"
            )
        return "\n".join(lines)
