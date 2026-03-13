"""
Tests for audit logging, fairness checking, and clinical rules.
Run with: pytest tests/test_compliance.py -v
"""
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ------------------------------------------------------------------
# Clinical Rules
# ------------------------------------------------------------------

def test_clinical_rules_adult_no_flags():
    from src.clinical.clinical_rules import ClinicalDecisionSupport
    cds = ClinicalDecisionSupport()
    features = {
        "age": 45, "sex": 1, "cholesterol": 200, "resting_bp": 120,
        "exercise_angina": 0, "max_hr": 155, "chest_pain_type": 1,
    }
    result = cds.validate(0, features)
    assert result["guideline_adherent"] is True
    assert result["has_contraindication"] is False


def test_clinical_rules_pediatric_aspirin():
    from src.clinical.clinical_rules import ClinicalDecisionSupport
    cds = ClinicalDecisionSupport()
    features = {"age": 12, "cholesterol": 160, "resting_bp": 110,
                "exercise_angina": 0, "max_hr": 185, "chest_pain_type": 0}
    result = cds.validate(0, features)
    assert result["has_contraindication"] is True


def test_clinical_rules_high_cholesterol():
    from src.clinical.clinical_rules import ClinicalDecisionSupport
    cds = ClinicalDecisionSupport()
    features = {"age": 55, "cholesterol": 280, "resting_bp": 130,
                "exercise_angina": 0, "max_hr": 130, "chest_pain_type": 1}
    result = cds.validate(1, features)
    flag_ids = [f["rule_id"] for f in result["findings"]]
    assert "CV-002" in flag_ids


# ------------------------------------------------------------------
# Validators
# ------------------------------------------------------------------

def test_feature_validator_passes():
    from src.utils.validators import FeatureValidator
    v = FeatureValidator()
    features = {
        "age": 50, "sex": 1, "chest_pain_type": 2, "resting_bp": 130,
        "cholesterol": 220, "fasting_blood_sugar": 0,
        "max_hr": 150, "exercise_angina": 0, "oldpeak": 1.5,
    }
    cleaned = v.validate(features)
    assert cleaned["age"] == 50.0


def test_feature_validator_missing_field():
    from src.utils.validators import FeatureValidator
    v = FeatureValidator()
    with pytest.raises(ValueError, match="Missing required feature"):
        v.validate({"age": 50})


def test_feature_validator_out_of_range():
    from src.utils.validators import FeatureValidator
    v = FeatureValidator()
    features = {
        "age": 999,  # out of range
        "sex": 1, "chest_pain_type": 0, "resting_bp": 120,
        "cholesterol": 200, "fasting_blood_sugar": 0,
        "max_hr": 150, "exercise_angina": 0, "oldpeak": 0.5,
    }
    with pytest.raises(ValueError, match="above maximum"):
        v.validate(features)


# ------------------------------------------------------------------
# Audit Logger
# ------------------------------------------------------------------

def test_audit_logger(tmp_path):
    from src.compliance.audit_logger import AuditLogger
    db_path = str(tmp_path / "test_audit.db")
    audit = AuditLogger(db_path=db_path)

    features = np.array([45, 1, 2, 130, 220, 0, 150, 0, 1.5])
    row_id = audit.log(
        patient_id="PT-TEST-001",
        features=features,
        prediction=1,
        confidence=0.76,
        explanation={"shap": {}, "lime": {}},
        guideline_check={"has_contraindication": False, "total_flags": 0},
    )
    assert row_id == 1

    records = audit.fetch_recent(limit=10)
    assert len(records) == 1
    assert records[0]["patient_id"] == "PT-TEST-001"

    summary = audit.compliance_summary()
    assert summary["total_predictions"] == 1
    assert summary["flagged_predictions"] == 0


# ------------------------------------------------------------------
# Fairness Checker
# ------------------------------------------------------------------

def test_fairness_checker():
    from src.compliance.fairness_checker import FairnessChecker
    rng = np.random.default_rng(42)
    n = 200
    y_true = rng.integers(0, 2, n)
    y_pred = (y_true + rng.integers(0, 2, n)) % 2   # noisy predictions
    y_proba = rng.random(n)
    groups = rng.choice(["M", "F"], n)

    fc = FairnessChecker()
    dpd = fc.demographic_parity_difference(y_pred, groups)
    assert 0 <= dpd <= 1

    report = fc.full_report(y_true, y_pred, y_proba, {"sex": groups})
    assert "sex" in report
    assert "_summary" in report
