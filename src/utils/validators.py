"""
Input Validators — validate and sanitise patient feature dictionaries
before they reach the model. Raises clear errors for clinicians / API callers.
"""

FEATURE_CONSTRAINTS = {
    "age":                  {"min": 0,   "max": 120, "type": float},
    "sex":                  {"values": [0, 1],        "type": int},
    "chest_pain_type":      {"values": [0, 1, 2, 3],  "type": int},
    "resting_bp":           {"min": 60,  "max": 250,  "type": float},
    "cholesterol":          {"min": 50,  "max": 700,  "type": float},
    "fasting_blood_sugar":  {"values": [0, 1],        "type": int},
    "max_hr":               {"min": 40,  "max": 250,  "type": float},
    "exercise_angina":      {"values": [0, 1],        "type": int},
    "oldpeak":              {"min": 0.0, "max": 10.0, "type": float},
}


class FeatureValidator:

    def validate(self, features: dict) -> dict:
        """
        Validate a feature dict. Returns cleaned dict or raises ValueError.
        """
        errors = []
        cleaned = {}

        for name, constraints in FEATURE_CONSTRAINTS.items():
            if name not in features:
                errors.append(f"Missing required feature: '{name}'")
                continue

            raw = features[name]

            # Type coercion
            try:
                val = constraints["type"](raw)
            except (TypeError, ValueError):
                errors.append(f"'{name}' must be {constraints['type'].__name__}, got {raw!r}")
                continue

            # Range check
            if "min" in constraints and val < constraints["min"]:
                errors.append(f"'{name}' = {val} is below minimum {constraints['min']}")
            elif "max" in constraints and val > constraints["max"]:
                errors.append(f"'{name}' = {val} is above maximum {constraints['max']}")

            # Allowed values check
            if "values" in constraints and val not in constraints["values"]:
                errors.append(f"'{name}' must be one of {constraints['values']}, got {val}")

            cleaned[name] = val

        if errors:
            raise ValueError("Feature validation failed:\n  " + "\n  ".join(errors))

        return cleaned

    def to_array(self, features: dict):
        """Return validated features as an ordered numpy array."""
        import numpy as np
        cleaned = self.validate(features)
        return np.array([cleaned[k] for k in FEATURE_CONSTRAINTS], dtype=float)

    def feature_order(self) -> list:
        return list(FEATURE_CONSTRAINTS.keys())
