"""
Regulatory Reports — generates compliance summaries for regulatory bodies
(FDA, CE, NHS Digital) and internal quality assurance teams.
"""
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class RegulatoryReportGenerator:

    def generate(self, audit_summary: dict, fairness_report: dict,
                 model_metrics: dict, model_version: str = "decision_tree_v2") -> dict:
        """Compile a full regulatory compliance report."""
        now = datetime.utcnow().isoformat()
        return {
            "report_type": "Regulatory Compliance Report",
            "generated_at": now,
            "model_version": model_version,
            "intended_use": "Clinical decision support — cardiovascular risk assessment.",
            "regulatory_status": "Research / Decision Support (not cleared for standalone diagnosis)",
            "performance": {
                "accuracy":  model_metrics.get("accuracy"),
                "roc_auc":   model_metrics.get("roc_auc"),
                "validation_set_size": model_metrics.get("n_test"),
            },
            "usage_statistics": audit_summary,
            "fairness": {
                "overall_pass": fairness_report.get("_summary", {}).get("overall_fairness_pass"),
                "attributes_checked": fairness_report.get("_summary", {}).get("attributes_checked", []),
                "details": {k: v for k, v in fairness_report.items() if not k.startswith("_")},
            },
            "safety_measures": [
                "All predictions accompanied by SHAP and LIME explanations.",
                "Automated guideline validation before output is surfaced.",
                "95% confidence intervals shown for every prediction.",
                "Full audit trail stored in tamper-evident SQLite log.",
                "Disclaimer prominently displayed on every clinical report.",
                "Fairness monitoring across sex and age groups — auto-flagged if thresholds breached.",
            ],
            "known_limitations": [
                "Trained on UCI Heart Disease dataset — may not generalise to all populations.",
                "Does not incorporate imaging, genetic, or longitudinal data.",
                "Model performance may degrade with data drift — monitor in production.",
            ],
            "next_review_date": "Annual or upon significant model update.",
        }

    def to_json(self, report: dict, output_path: str = None) -> str:
        payload = json.dumps(report, indent=2, default=str)
        if output_path:
            with open(output_path, "w") as f:
                f.write(payload)
            logger.info(f"Regulatory report saved to {output_path}")
        return payload

    def to_markdown(self, report: dict, output_path: str = None) -> str:
        lines = [
            f"# {report['report_type']}",
            f"\n**Generated:** {report['generated_at']}  ",
            f"**Model:** {report['model_version']}  ",
            f"**Intended Use:** {report['intended_use']}  ",
            f"**Regulatory Status:** {report['regulatory_status']}",
            "\n## Performance",
            f"- Accuracy: {report['performance']['accuracy']}",
            f"- ROC-AUC:  {report['performance']['roc_auc']}",
            "\n## Usage Statistics",
        ]
        for k, v in report["usage_statistics"].items():
            lines.append(f"- {k}: {v}")
        lines += [
            f"\n## Fairness — Overall Pass: {report['fairness']['overall_pass']}",
            "\n## Safety Measures",
        ]
        for s in report["safety_measures"]:
            lines.append(f"- {s}")
        lines += ["\n## Known Limitations"]
        for l in report["known_limitations"]:
            lines.append(f"- {l}")
        md = "\n".join(lines)
        if output_path:
            with open(output_path, "w") as f:
                f.write(md)
            logger.info(f"Markdown report saved to {output_path}")
        return md
