"""
Clinical Decision Support — validates AI predictions against
evidence-based medical guidelines (ACC/AHA, ESC, USPSTF, FDA).
"""
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rule definitions (extend this dict to add new guidelines)
# ---------------------------------------------------------------------------
CARDIOVASCULAR_RULES = [
    {
        "id": "CV-001",
        "source": "FDA Safety Communication",
        "level": "contraindication",
        "check": lambda f: f.get("age", 99) < 18,
        "message": "Patient is a minor — aspirin is contraindicated (Reye's syndrome risk).",
    },
    {
        "id": "CV-002",
        "source": "ACC/AHA Cholesterol Guidelines 2018 · Class I",
        "level": "recommendation",
        "check": lambda f: f.get("cholesterol", 0) > 240,
        "message": "Cholesterol > 240 mg/dL — statin therapy strongly recommended.",
    },
    {
        "id": "CV-003",
        "source": "ESC Guidelines 2019 · Class I, LOE A",
        "level": "warning",
        "check": lambda f: f.get("exercise_angina", 0) == 1,
        "message": "Exercise-induced angina present — cardiology referral indicated.",
    },
    {
        "id": "CV-004",
        "source": "JNC 8 2014 · Class I, LOE A",
        "level": "warning",
        "check": lambda f: f.get("resting_bp", 0) > 140,
        "message": "Resting BP > 140 mmHg — antihypertensive therapy consideration.",
    },
    {
        "id": "CV-005",
        "source": "AHA Scientific Statement 2020",
        "level": "info",
        "check": lambda f: (
            f.get("max_hr", 999) < (220 - f.get("age", 50)) * 0.70
        ),
        "message": "Max HR below 70% age-predicted — investigate chronotropic incompetence.",
    },
    {
        "id": "CV-006",
        "source": "ACC/AHA 2023 · Class I, LOE B-R",
        "level": "info",
        "check": lambda f: f.get("age", 0) > 40 and f.get("chest_pain_type", -1) >= 0,
        "message": "Age > 40 with chest pain — stress testing evaluation warranted.",
    },
]


class ClinicalDecisionSupport:
    def __init__(self, ruleset: list = None):
        self.rules = ruleset or CARDIOVASCULAR_RULES

    def validate(self, prediction, features: dict) -> dict:
        """
        Run all rules against the feature dict.
        Returns a structured validation result.
        """
        findings = []
        has_contraindication = False

        for rule in self.rules:
            try:
                triggered = rule["check"](features)
            except Exception as e:
                logger.warning(f"Rule {rule['id']} evaluation error: {e}")
                triggered = False

            if triggered:
                finding = {
                    "rule_id": rule["id"],
                    "level": rule["level"],
                    "message": rule["message"],
                    "source": rule["source"],
                }
                findings.append(finding)
                if rule["level"] == "contraindication":
                    has_contraindication = True

        return {
            "prediction": int(prediction[0]) if hasattr(prediction, "__len__") else int(prediction),
            "guideline_adherent": not has_contraindication,
            "has_contraindication": has_contraindication,
            "findings": findings,
            "total_flags": len(findings),
        }

    def add_rule(self, rule: dict):
        """Dynamically add a new clinical rule at runtime."""
        required = {"id", "source", "level", "check", "message"}
        if not required.issubset(rule.keys()):
            raise ValueError(f"Rule must contain keys: {required}")
        self.rules.append(rule)
