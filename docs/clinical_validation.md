# Clinical Validation Plan

## Purpose

This document describes the validation strategy for MedXAI prior to any
clinical deployment. All validation must be completed and reviewed by
qualified clinicians before the system is used in patient-facing workflows.

---

## Validation Stages

### Stage 1 — Technical Validation (Internal)
- [ ] Unit tests pass (`pytest tests/ -v`) with ≥ 90% code coverage
- [ ] Model accuracy ≥ 80% and ROC-AUC ≥ 0.85 on held-out test set
- [ ] SHAP explanations verified on known cases (sanity check)
- [ ] All clinical guideline rules reviewed by a physician
- [ ] Fairness: demographic parity difference < 0.10 across sex and age groups
- [ ] API load tested at 100 concurrent requests
- [ ] Audit log: all fields correctly populated and tamper-evident

### Stage 2 — Clinical Review (External)
- [ ] 10 attending cardiologists review 50 case explanations each
- [ ] Clinician trust score ≥ 4/5 on explanation clarity (Likert scale)
- [ ] Sensitivity ≥ 85%, specificity ≥ 75% on clinical validation dataset
- [ ] No missed contraindications in guideline validation module
- [ ] Counterfactual suggestions reviewed for clinical plausibility

### Stage 3 — Prospective Pilot
- [ ] 3-month silent deployment in one outpatient cardiology clinic
- [ ] Model predictions recorded but not shown to clinicians
- [ ] Compare model predictions vs final clinical diagnoses
- [ ] Monitor for data drift using PSI (Population Stability Index)
- [ ] Review audit log for unexpected patterns

### Stage 4 — Supervised Deployment
- [ ] Model predictions shown to clinicians as advisory overlay
- [ ] Clinicians must confirm or override every AI recommendation
- [ ] All overrides logged and reviewed monthly
- [ ] Performance metrics recalculated quarterly

---

## Minimum Performance Thresholds

| Metric                          | Threshold   |
|---------------------------------|-------------|
| Accuracy                        | ≥ 80%       |
| ROC-AUC                         | ≥ 0.85      |
| Sensitivity (recall, positive)  | ≥ 85%       |
| Specificity                     | ≥ 75%       |
| Demographic parity difference   | < 0.10      |
| Equalized odds difference (TPR) | < 0.10      |
| Explanation fidelity (LIME R²)  | ≥ 0.75      |

---

## Revalidation Triggers

The model must be retrained and revalidated if:
- Patient population demographics shift significantly (PSI > 0.2)
- Clinical guidelines used in validation are updated
- Model accuracy drops below threshold on a rolling 30-day window
- A patient safety incident is linked to a model prediction
