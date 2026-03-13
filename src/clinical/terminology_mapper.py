"""
Terminology Mapper — translates raw feature names / codes into
human-readable clinical terminology (ICD-10, SNOMED CT, LOINC stubs).
"""

FEATURE_DISPLAY_NAMES = {
    "age": "Age (years)",
    "sex": "Biological Sex",
    "chest_pain_type": "Chest Pain Type",
    "resting_bp": "Resting Blood Pressure (mmHg)",
    "cholesterol": "Serum Cholesterol (mg/dL)",
    "fasting_blood_sugar": "Fasting Blood Sugar",
    "max_hr": "Maximum Heart Rate Achieved (bpm)",
    "exercise_angina": "Exercise-Induced Angina",
    "oldpeak": "ST Depression (Oldpeak)",
    "st_slope": "Slope of Peak Exercise ST Segment",
}

CATEGORICAL_VALUE_MAP = {
    "sex": {0: "Female", 1: "Male"},
    "chest_pain_type": {
        0: "Typical Angina",
        1: "Atypical Angina",
        2: "Non-Anginal Pain",
        3: "Asymptomatic",
    },
    "fasting_blood_sugar": {0: "< 120 mg/dL (Normal)", 1: "≥ 120 mg/dL (Elevated)"},
    "exercise_angina": {0: "No", 1: "Yes"},
    "st_slope": {0: "Upsloping", 1: "Flat", 2: "Downsloping"},
}

PREDICTION_LABELS = {0: "Low Cardiac Risk", 1: "Elevated Cardiac Risk"}


class TerminologyMapper:
    def display_name(self, feature_key: str) -> str:
        return FEATURE_DISPLAY_NAMES.get(feature_key, feature_key.replace("_", " ").title())

    def display_value(self, feature_key: str, raw_value) -> str:
        mapping = CATEGORICAL_VALUE_MAP.get(feature_key)
        if mapping:
            return mapping.get(int(raw_value), str(raw_value))
        return str(raw_value)

    def prediction_label(self, prediction: int) -> str:
        return PREDICTION_LABELS.get(prediction, f"Class {prediction}")

    def humanize_features(self, feature_dict: dict) -> dict:
        """Return a new dict with display names and values."""
        return {
            self.display_name(k): self.display_value(k, v)
            for k, v in feature_dict.items()
        }
