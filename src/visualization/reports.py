"""
Clinical Report Generator — produces structured, doctor-friendly reports
from model predictions, SHAP values, and guideline validation results.
"""
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class ClinicalReportGenerator:

    def generate(self, patient_id: str, features: dict, prediction: int,
                 confidence: float, shap_result: dict, lime_result: dict,
                 guideline_result: dict, recommendations: list,
                 ci_low: float = None, ci_high: float = None) -> dict:
        """Build the full structured clinical report dict."""
        return {
            "report_id": f"RPT-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}-{patient_id}",
            "generated_at": datetime.utcnow().isoformat(),
            "patient_id": patient_id,
            "model_version": "decision_tree_v2",
            "prediction": {
                "class": prediction,
                "label": "Elevated Cardiac Risk" if prediction == 1 else "Low Cardiac Risk",
                "confidence": round(confidence, 4),
                "confidence_pct": f"{confidence:.1%}",
                "ci_95": {
                    "lower": round(ci_low, 4) if ci_low is not None else None,
                    "upper": round(ci_high, 4) if ci_high is not None else None,
                },
            },
            "key_factors": shap_result.get("top_features", []),
            "lime_narrative": lime_result.get("narrative", ""),
            "lime_fidelity": lime_result.get("score"),
            "guideline_check": guideline_result,
            "recommendations": recommendations,
            "input_features": features,
            "disclaimer": (
                "CLINICAL DECISION SUPPORT ONLY. This AI output must be reviewed "
                "and validated by a qualified healthcare professional before any "
                "clinical action is taken."
            ),
        }

    def to_text(self, report: dict) -> str:
        """Render the report as plain text for printing or EMR export."""
        lines = [
            "=" * 60,
            "  MEDXAI CLINICAL DECISION SUPPORT REPORT",
            "=" * 60,
            f"  Report ID   : {report['report_id']}",
            f"  Generated   : {report['generated_at']}",
            f"  Patient ID  : {report['patient_id']}",
            f"  Model       : {report['model_version']}",
            "-" * 60,
            "  PREDICTION",
            f"    {report['prediction']['label']}",
            f"    Confidence  : {report['prediction']['confidence_pct']}",
        ]
        ci = report["prediction"]["ci_95"]
        if ci["lower"] is not None:
            lines.append(f"    95% CI      : [{ci['lower']:.1%}, {ci['upper']:.1%}]")

        lines += ["-" * 60, "  KEY CONTRIBUTING FACTORS (SHAP)"]
        for f in report["key_factors"]:
            arrow = "▲" if f["direction"] == "increases_risk" else "▼"
            lines.append(f"    {arrow} {f['feature']:25s}  SHAP={f['shap_value']:+.4f}")

        lines += ["-" * 60, "  GUIDELINE FLAGS"]
        findings = report["guideline_check"].get("findings", [])
        if findings:
            for g in findings:
                lines.append(f"    [{g['level'].upper()}] {g['message']}")
                lines.append(f"             Source: {g['source']}")
        else:
            lines.append("    No guideline conflicts detected.")

        lines += ["-" * 60, "  RECOMMENDATIONS"]
        for i, rec in enumerate(report["recommendations"], 1):
            lines.append(f"    {i}. {rec}")

        lines += [
            "-" * 60,
            f"  ⚠  {report['disclaimer']}",
            "=" * 60,
        ]
        return "\n".join(lines)

    def to_json(self, report: dict, path: str = None) -> str:
        payload = json.dumps(report, indent=2, default=str)
        if path:
            with open(path, "w") as f:
                f.write(payload)
            logger.info(f"Report saved to {path}")
        return payload

    @staticmethod
    def generate_recommendations(prediction: int, features: dict) -> list:
        recs = []
        confidence_high = features.get("oldpeak", 0) > 2 or features.get("exercise_angina", 0) == 1

        if prediction == 1:
            recs.append("Immediate cardiology consultation within 24–48 hours.")
            recs.append("Order resting 12-lead ECG and transthoracic echocardiogram.")
            if confidence_high:
                recs.append("Consider coronary angiography based on clinical judgment.")
            recs.append("Review and initiate antiplatelet therapy if not contraindicated.")
            recs.append("Schedule follow-up appointment within 1 week.")
        else:
            recs.append("Continue routine preventive cardiovascular care.")
            recs.append("Annual blood pressure and cholesterol monitoring.")
            recs.append("Lifestyle counselling: diet, exercise, smoking cessation.")
            recs.append("Next scheduled review in 12 months unless symptoms develop.")
        return recs
