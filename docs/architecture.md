# MedXAI — System Architecture

## Overview

MedXAI is a modular, production-grade Explainable AI system for clinical
decision support. Every prediction is accompanied by SHAP values, LIME
local explanations, guideline validation, and a 95% confidence interval.

---

## Directory Structure

```
medical-explainable-ai/
│
├── src/                          # Core Python package
│   ├── main.py                   # Pipeline entry point
│   │
│   ├── models/
│   │   ├── explainable_model.py  # Model wrapper (DT / RF / LR)
│   │   ├── train_model.py        # CLI training script
│   │   └── model_registry.py    # Version tracking
│   │
│   ├── explainability/
│   │   ├── shap_explainer.py     # SHAP values (global + local)
│   │   ├── lime_explainer.py     # LIME local approximations
│   │   ├── feature_importance.py # Intrinsic + permutation importance
│   │   └── counterfactuals.py   # "What would need to change?"
│   │
│   ├── clinical/
│   │   ├── clinical_rules.py     # Hard-coded evidence-based rules
│   │   ├── guidelines_validator.py  # JSON-driven guideline checker
│   │   └── terminology_mapper.py # Feature codes → clinical labels
│   │
│   ├── visualization/
│   │   ├── dashboard.py          # Plotly interactive charts
│   │   ├── plots.py              # Matplotlib static charts
│   │   └── reports.py           # Clinical report generator
│   │
│   ├── compliance/
│   │   ├── audit_logger.py       # SQLite audit trail (HIPAA-aligned)
│   │   ├── fairness_checker.py   # Demographic parity + equalized odds
│   │   └── regulatory_reports.py # FDA/CE compliance report output
│   │
│   └── utils/
│       ├── data_loader.py        # CSV / Parquet / synthetic loader
│       ├── preprocessor.py       # Scaling + imputation pipeline
│       └── validators.py        # Feature range + type validation
│
├── api/
│   ├── app.py                    # Flask REST API (POST /predict etc.)
│   ├── routes.py                 # Blueprint routes
│   └── schemas.py               # JSON schema documentation
│
├── frontend/
│   └── templates/
│       └── dashboard.html        # Self-contained clinical UI
│
├── data/
│   ├── raw/clinical_guidelines.json
│   └── processed/               # train_data.parquet, test_data.parquet
│
├── models/
│   ├── trained/                  # Serialised .pkl model files
│   └── checkpoints/
│
├── tests/
│   ├── test_models.py
│   ├── test_explainability.py
│   └── test_compliance.py
│
├── config/
│   ├── config.yaml
│   └── logging_config.yaml
│
├── docs/                         # This documentation
├── logs/                         # Rotating log files + audit trail
├── notebooks/                    # Jupyter exploration notebooks
├── requirements.txt
├── setup.py
└── README.md
```

---

## Data Flow

```
HTTP POST /predict
      │
      ▼
FeatureValidator.validate()
      │
      ▼
Preprocessor.transform()
      │
      ├──► ExplainableMedicalAI.predict()       → class (0 or 1)
      │                                           → confidence (0–1)
      │
      ├──► SHAPExplainer.explain()              → top features + values
      │
      ├──► LIMEExplainer.explain()              → local narrative
      │
      ├──► ClinicalDecisionSupport.validate()   → guideline findings
      │
      ├──► ClinicalReportGenerator.generate()   → structured report dict
      │
      └──► AuditLogger.log()                    → SQLite record
      │
      ▼
JSON response → clinician dashboard
```

---

## Key Design Principles

1. **Interpretability First** — shallow decision trees and L1 logistic
   regression are preferred; SHAP wraps random forests where needed.

2. **Multi-level Explanations** — SHAP (global + local), LIME (local),
   counterfactuals (actionable), and natural-language narratives.

3. **Uncertainty Quantification** — 95% bootstrap confidence intervals
   displayed on every prediction.

4. **Guideline Validation** — predictions are checked against ACC/AHA,
   ESC, JNC, and FDA rules before surfacing to the clinician.

5. **Regulatory Compliance** — full audit trail, fairness monitoring,
   and auto-generated compliance reports.

6. **Modular & Extensible** — swap models, add clinical rulesets, or
   plug in new explainability methods without touching the API layer.
