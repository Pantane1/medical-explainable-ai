"""
Guidelines Validator — loads structured clinical guidelines from JSON
and checks predictions for adherence and conflicts.
"""
import json
import logging
import os

logger = logging.getLogger(__name__)


class GuidelinesValidator:
    def __init__(self, guidelines_path: str = "data/raw/clinical_guidelines.json"):
        self.guidelines = []
        if os.path.exists(guidelines_path):
            self._load(guidelines_path)
        else:
            logger.warning(f"Guidelines file not found at {guidelines_path}. "
                           "Using empty ruleset.")

    def _load(self, path: str):
        with open(path) as f:
            data = json.load(f)
        self.guidelines = data.get("guidelines", [])
        logger.info(f"Loaded {len(self.guidelines)} guidelines from {path}")

    def check(self, prediction: int, features: dict) -> list:
        """Return list of guideline conflicts for the given prediction + features."""
        conflicts = []
        for g in self.guidelines:
            # Each guideline has: id, description, condition_key,
            #   condition_op, condition_value, applies_to_prediction
            applies = g.get("applies_to_prediction")
            if applies is not None and applies != prediction:
                continue
            key = g.get("condition_key")
            op  = g.get("condition_op", "gt")
            val = g.get("condition_value")
            feat_val = features.get(key)
            if feat_val is None:
                continue
            triggered = False
            if op == "gt"  and feat_val > val:  triggered = True
            elif op == "lt" and feat_val < val:  triggered = True
            elif op == "eq" and feat_val == val: triggered = True
            elif op == "ne" and feat_val != val: triggered = True
            if triggered:
                conflicts.append({
                    "guideline_id": g["id"],
                    "description": g["description"],
                    "severity": g.get("severity", "warning"),
                    "source": g.get("source", ""),
                })
        return conflicts
