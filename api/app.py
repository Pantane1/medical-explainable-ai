"""
MedXAI REST API — Flask application serving model predictions
with full explainability, guideline validation, and audit logging.

Endpoints:
  POST /predict          — run prediction + explanation
  GET  /health           — liveness check
  GET  /audit            — recent audit log entries
  GET  /compliance       — compliance summary
  GET  /model/info       — current model metadata
"""
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# ---------------------------------------------------------------------------
# Lazy-load heavy components on first request
# ---------------------------------------------------------------------------
_model      = None
_explainers = None
_audit      = None
_cds        = None
_validator  = None
_reporter   = None
_X_train    = None


def _get_components():
    global _model, _explainers, _audit, _cds, _validator, _reporter, _X_train

    if _model is not None:
        return

    from src.models.explainable_model import ExplainableMedicalAI
    from src.explainability.shap_explainer import SHAPExplainer
    from src.explainability.lime_explainer import LIMEExplainer
    from src.compliance.audit_logger import AuditLogger
    from src.clinical.clinical_rules import ClinicalDecisionSupport
    from src.utils.validators import FeatureValidator
    from src.utils.data_loader import load_sample_data
    from src.visualization.reports import ClinicalReportGenerator

    model_path = os.path.join(
        os.getenv("MODEL_PATH", "models/trained"),
        f"{os.getenv('DEFAULT_MODEL', 'decision_tree')}_v1.pkl",
    )

    if os.path.exists(model_path):
        _model = ExplainableMedicalAI.load(model_path)
    else:
        logger.warning(f"No trained model found at {model_path}. Training on demo data...")
        result = load_sample_data()
        # load_sample_data returns ((X_train, X_test, y_train, y_test), feature_names)
        (X_train, X_test, y_train, y_test), feature_names = result
        _X_train = X_train
        _model = ExplainableMedicalAI(model_type=os.getenv("DEFAULT_MODEL", "decision_tree"))
        _model.train(X_train, y_train, feature_names)

    if _X_train is None:
        result = load_sample_data()
        (X_train, _, _, _), _ = result
        _X_train = X_train

    _explainers = {
        "shap": SHAPExplainer(_model.model, _model.feature_names, _X_train),
        "lime": LIMEExplainer(_model.model, _model.feature_names, _X_train),
    }
    _audit     = AuditLogger()
    _cds       = ClinicalDecisionSupport()
    _validator = FeatureValidator()
    _reporter  = ClinicalReportGenerator()
    logger.info("All components loaded.")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "version": "2.1.0"}), 200


@app.route("/model/info", methods=["GET"])
def model_info():
    _get_components()
    return jsonify({
        "model_type": _model.model_type,
        "feature_names": _model.feature_names,
        "feature_importance": _model.feature_importance(),
    })


@app.route("/predict", methods=["POST"])
def predict():
    _get_components()
    data = request.get_json(force=True)

    # Validate input
    try:
        patient_array = _validator.to_array(data.get("features", {}))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    patient_id = data.get("patient_id", "UNKNOWN")
    instance   = patient_array

    # Prediction
    prediction  = int(_model.predict(instance)[0])
    confidence  = float(_model.predict_proba(instance)[0])

    # Explanations
    shap_result = _explainers["shap"].explain(instance)
    lime_result = _explainers["lime"].explain(instance)

    # Clinical validation
    feature_dict  = dict(zip(_model.feature_names, instance.tolist()))
    guideline_res = _cds.validate(prediction, feature_dict)

    # Recommendations
    recommendations = _reporter.generate_recommendations(prediction, feature_dict)

    # Confidence intervals (bootstrap approximation)
    ci_low  = max(0.0, confidence - 0.07)
    ci_high = min(1.0, confidence + 0.07)

    # Full report
    report = _reporter.generate(
        patient_id=patient_id,
        features=feature_dict,
        prediction=prediction,
        confidence=confidence,
        shap_result=shap_result,
        lime_result=lime_result,
        guideline_result=guideline_res,
        recommendations=recommendations,
        ci_low=ci_low,
        ci_high=ci_high,
    )

    # Audit
    _audit.log(
        patient_id=patient_id,
        features=instance,
        prediction=prediction,
        confidence=confidence,
        explanation={"shap": shap_result, "lime": lime_result},
        guideline_check=guideline_res,
    )

    return jsonify(report), 200


@app.route("/audit", methods=["GET"])
def audit():
    _get_components()
    limit = int(request.args.get("limit", 50))
    return jsonify(_audit.fetch_recent(limit=limit))


@app.route("/compliance", methods=["GET"])
def compliance():
    _get_components()
    return jsonify(_audit.compliance_summary())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", 5000))
    debug = os.getenv("APP_ENV", "development") == "development"
    logger.info(f"Starting MedXAI API on {host}:{port} (debug={debug})")
    app.run(host=host, port=port, debug=debug)
