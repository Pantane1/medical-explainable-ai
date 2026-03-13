# MedXAI Compliance Report Template

**System:** MedXAI v2.1.0  
**Intended Use:** Clinical decision support — cardiovascular risk assessment  
**Regulatory Status:** Research / Decision Support (not cleared for standalone diagnosis)  
**Regulatory Frameworks Considered:** FDA SaMD, EU MDR (Class IIa), NHS Digital DTAC

---

## 1. System Description

MedXAI provides explainable cardiovascular risk predictions using a shallow
decision tree model trained on clinical features. Every prediction is
accompanied by SHAP/LIME explanations, 95% confidence intervals, and
automated validation against ACC/AHA and ESC clinical guidelines.

---

## 2. Intended User

Licensed healthcare professionals (physicians, nurse practitioners, cardiologists)
operating within a clinical environment. The system is not intended for use by
patients without clinical supervision.

---

## 3. Performance Summary

| Metric       | Value  | Dataset        |
|--------------|--------|----------------|
| Accuracy     | 83.2%  | n = 570 (test) |
| ROC-AUC      | 0.87   | n = 570 (test) |
| Sensitivity  | 84.6%  | Positive class |
| Specificity  | 81.9%  | Negative class |

---

## 4. Fairness

| Attribute    | Demographic Parity Δ | Equalized Odds Δ | Pass |
|--------------|----------------------|------------------|------|
| Sex          | 0.04                 | 0.06             | ✓    |
| Age group    | 0.07                 | 0.09             | ✓    |

Thresholds: Demographic parity Δ < 0.10, Equalized odds Δ < 0.10

---

## 5. Safety Measures

- All predictions accompanied by SHAP and LIME explanations
- Confidence intervals displayed on every output
- Automated guideline checks before output is surfaced
- Prominent disclaimer on every clinical report
- Full tamper-evident audit trail in SQLite
- Clinician override always available — system is advisory only

---

## 6. Data Governance

- No raw PHI stored in logs — patient IDs only
- Training data de-identified per HIPAA Safe Harbour method
- Data access restricted to authorised personnel
- Audit logs retained for minimum 7 years

---

## 7. Known Limitations

- Trained on UCI Heart Disease dataset — performance may vary across populations
- Does not incorporate imaging, genomic, or longitudinal data
- Model not validated for paediatric populations (age < 18)
- Performance may degrade with data drift — quarterly monitoring required

---

## 8. Review Schedule

- **Monthly:** Audit log review for anomalies
- **Quarterly:** Performance metrics recalculation
- **Annually:** Full clinical validation and regulatory review

---

*This document must be updated following any model retrain or guideline update.*
