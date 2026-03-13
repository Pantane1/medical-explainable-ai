"""
Pydantic-style schema documentation for the MedXAI API.
Used for documentation and optional runtime validation.
"""

PREDICT_REQUEST_SCHEMA = {
    "type": "object",
    "required": ["patient_id", "features"],
    "properties": {
        "patient_id": {"type": "string", "example": "PT-00123"},
        "features": {
            "type": "object",
            "required": [
                "age", "sex", "chest_pain_type", "resting_bp",
                "cholesterol", "fasting_blood_sugar", "max_hr",
                "exercise_angina", "oldpeak",
            ],
            "properties": {
                "age":                  {"type": "number", "minimum": 0,   "maximum": 120},
                "sex":                  {"type": "integer", "enum": [0, 1]},
                "chest_pain_type":      {"type": "integer", "enum": [0, 1, 2, 3]},
                "resting_bp":           {"type": "number", "minimum": 60,  "maximum": 250},
                "cholesterol":          {"type": "number", "minimum": 50,  "maximum": 700},
                "fasting_blood_sugar":  {"type": "integer", "enum": [0, 1]},
                "max_hr":               {"type": "number", "minimum": 40,  "maximum": 250},
                "exercise_angina":      {"type": "integer", "enum": [0, 1]},
                "oldpeak":              {"type": "number", "minimum": 0.0, "maximum": 10.0},
            },
        },
    },
}

PREDICT_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "report_id":    {"type": "string"},
        "generated_at": {"type": "string", "format": "date-time"},
        "patient_id":   {"type": "string"},
        "prediction": {
            "type": "object",
            "properties": {
                "class":           {"type": "integer"},
                "label":           {"type": "string"},
                "confidence":      {"type": "number"},
                "confidence_pct":  {"type": "string"},
                "ci_95": {
                    "type": "object",
                    "properties": {
                        "lower": {"type": "number"},
                        "upper": {"type": "number"},
                    },
                },
            },
        },
        "key_factors":      {"type": "array"},
        "lime_narrative":   {"type": "string"},
        "guideline_check":  {"type": "object"},
        "recommendations":  {"type": "array", "items": {"type": "string"}},
        "disclaimer":       {"type": "string"},
    },
}
