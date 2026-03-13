# MedXAI API Documentation

Base URL: `http://localhost:5000`

---

## Endpoints

### `GET /health`
Liveness check.

**Response 200**
```json
{ "status": "ok", "version": "2.1.0" }
```

---

### `GET /model/info`
Returns the active model type, feature names, and global importances.

---

### `POST /predict`
Run a full prediction with explainability for a single patient.

**Request Body**
```json
{
  "patient_id": "PT-00123",
  "features": {
    "age": 55,
    "sex": 1,
    "chest_pain_type": 3,
    "resting_bp": 140,
    "cholesterol": 265,
    "fasting_blood_sugar": 1,
    "max_hr": 115,
    "exercise_angina": 1,
    "oldpeak": 2.3
  }
}
```

**Feature Reference**

| Feature              | Type  | Range / Values          | Description                     |
|----------------------|-------|-------------------------|---------------------------------|
| age                  | float | 0–120                   | Patient age in years            |
| sex                  | int   | 0=Female, 1=Male        | Biological sex                  |
| chest_pain_type      | int   | 0–3                     | 0=Typical, 1=Atypical, 2=Non-anginal, 3=Asymptomatic |
| resting_bp           | float | 60–250 mmHg             | Resting blood pressure          |
| cholesterol          | float | 50–700 mg/dL            | Serum cholesterol               |
| fasting_blood_sugar  | int   | 0=Normal, 1=High        | Fasting BS ≥ 120 mg/dL = 1     |
| max_hr               | float | 40–250 bpm              | Maximum heart rate achieved     |
| exercise_angina      | int   | 0=No, 1=Yes             | Exercise-induced angina         |
| oldpeak              | float | 0.0–10.0                | ST depression (exercise vs rest)|

**Response 200** (truncated)
```json
{
  "report_id": "RPT-20240101120000-PT-00123",
  "generated_at": "2024-01-01T12:00:00",
  "patient_id": "PT-00123",
  "model_version": "decision_tree_v2",
  "prediction": {
    "class": 1,
    "label": "Elevated Cardiac Risk",
    "confidence": 0.82,
    "confidence_pct": "82.0%",
    "ci_95": { "lower": 0.75, "upper": 0.89 }
  },
  "key_factors": [
    { "feature": "exercise_angina", "shap_value": 0.22, "direction": "increases_risk" },
    { "feature": "oldpeak",          "shap_value": 0.18, "direction": "increases_risk" }
  ],
  "lime_narrative": "Risk-increasing factors: exercise_angina, oldpeak, age. Protective factors: max_hr.",
  "guideline_check": {
    "guideline_adherent": true,
    "has_contraindication": false,
    "findings": [
      { "rule_id": "CV-002", "level": "recommendation",
        "message": "Cholesterol > 240 mg/dL — statin therapy strongly recommended.",
        "source": "ACC/AHA Cholesterol Guidelines 2018, Class I" }
    ],
    "total_flags": 1
  },
  "recommendations": [
    "Immediate cardiology consultation within 24–48 hours.",
    "Order resting 12-lead ECG and transthoracic echocardiogram."
  ],
  "disclaimer": "CLINICAL DECISION SUPPORT ONLY. ..."
}
```

**Error 400**
```json
{ "error": "Feature validation failed:\n  'age' = 999 is above maximum 120" }
```

---

### `GET /audit?limit=50`
Returns the most recent audit log entries.

---

### `GET /compliance`
Returns aggregate compliance statistics (total predictions, flag rate, avg confidence).

---

## cURL Example

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "patient_id": "PT-DEMO",
    "features": {
      "age": 62, "sex": 1, "chest_pain_type": 3,
      "resting_bp": 145, "cholesterol": 280,
      "fasting_blood_sugar": 1, "max_hr": 110,
      "exercise_angina": 1, "oldpeak": 2.5
    }
  }'
```
