"""
Main application entry point for Medical Explainable AI System.
Run this to launch the full pipeline interactively.
"""
import logging
import os
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    from src.models.explainable_model import ExplainableMedicalAI
    from src.explainability.shap_explainer import SHAPExplainer
    from src.explainability.lime_explainer import LIMEExplainer
    from src.compliance.audit_logger import AuditLogger
    from src.clinical.clinical_rules import ClinicalDecisionSupport
    from src.visualization.dashboard import MedicalVisualizer
    from src.utils.data_loader import load_sample_data

    logger.info("Initialising MedXAI pipeline...")

    # Load sample data
    X_train, X_test, y_train, y_test, feature_names = load_sample_data()

    # Train model
    model = ExplainableMedicalAI(model_type=os.getenv("DEFAULT_MODEL", "decision_tree"))
    model.train(X_train, y_train, feature_names)
    model.save(os.path.join(os.getenv("MODEL_PATH", "models/trained/"), "model_v1.pkl"))
    logger.info("Model trained and saved.")

    # Evaluate
    metrics = model.evaluate(X_test, y_test)
    logger.info(f"Evaluation metrics: {metrics}")

    # Explain first test instance
    shap_exp = SHAPExplainer(model.model, feature_names, X_train)
    lime_exp = LIMEExplainer(model.model, feature_names, X_train)

    instance = X_test[0]
    shap_vals = shap_exp.explain(instance)
    lime_vals = lime_exp.explain(instance)

    # Clinical validation
    cds = ClinicalDecisionSupport()
    validation = cds.validate(model.predict(instance), dict(zip(feature_names, instance)))

    # Audit
    audit = AuditLogger()
    audit.log(patient_id="PT-DEMO-001", features=instance,
              prediction=model.predict(instance),
              confidence=model.predict_proba(instance),
              explanation={"shap": shap_vals, "lime": lime_vals},
              guideline_check=validation)

    logger.info("Pipeline complete. Launch the API with: python api/app.py")


if __name__ == "__main__":
    main()
