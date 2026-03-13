"""
Fairness Checker — measures demographic parity, equalized odds,
and per-group performance metrics across sensitive attributes.
"""
import logging
import numpy as np

logger = logging.getLogger(__name__)


class FairnessChecker:
    THRESHOLD_DEMOGRAPHIC_PARITY = 0.10
    THRESHOLD_EQUALIZED_ODDS     = 0.10

    def __init__(self, sensitive_attributes: list = None):
        self.sensitive_attributes = sensitive_attributes or ["sex", "age_group"]

    # ------------------------------------------------------------------
    # Core metrics
    # ------------------------------------------------------------------

    def demographic_parity_difference(self, y_pred, groups) -> float:
        """Max positive prediction rate difference across groups."""
        rates = {}
        for g in np.unique(groups):
            mask = np.array(groups) == g
            rates[g] = np.mean(np.array(y_pred)[mask])
        vals = list(rates.values())
        return round(max(vals) - min(vals), 4) if vals else 0.0

    def equalized_odds_difference(self, y_true, y_pred, groups) -> dict:
        """TPR and FPR differences across groups."""
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        groups = np.array(groups)

        tpr, fpr = {}, {}
        for g in np.unique(groups):
            mask = groups == g
            pos = y_true[mask] == 1
            neg = y_true[mask] == 0
            tpr[g] = np.mean(y_pred[mask][pos]) if pos.sum() else 0.0
            fpr[g] = np.mean(y_pred[mask][neg]) if neg.sum() else 0.0

        tpr_vals = list(tpr.values())
        fpr_vals = list(fpr.values())
        return {
            "tpr_difference": round(max(tpr_vals) - min(tpr_vals), 4) if tpr_vals else 0.0,
            "fpr_difference": round(max(fpr_vals) - min(fpr_vals), 4) if fpr_vals else 0.0,
            "tpr_by_group": {str(k): round(v, 4) for k, v in tpr.items()},
            "fpr_by_group": {str(k): round(v, 4) for k, v in fpr.items()},
        }

    def group_performance(self, y_true, y_pred, y_proba, groups) -> dict:
        """Per-group accuracy, sensitivity, and specificity."""
        from sklearn.metrics import accuracy_score, roc_auc_score
        y_true  = np.array(y_true)
        y_pred  = np.array(y_pred)
        y_proba = np.array(y_proba)
        groups  = np.array(groups)

        result = {}
        for g in np.unique(groups):
            mask = groups == g
            yt, yp, ypr = y_true[mask], y_pred[mask], y_proba[mask]
            tp = int(((yt == 1) & (yp == 1)).sum())
            tn = int(((yt == 0) & (yp == 0)).sum())
            fp = int(((yt == 0) & (yp == 1)).sum())
            fn = int(((yt == 1) & (yp == 0)).sum())
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            try:
                auc = float(roc_auc_score(yt, ypr)) if len(np.unique(yt)) > 1 else None
            except Exception:
                auc = None
            result[str(g)] = {
                "n": int(mask.sum()),
                "accuracy": round(float(accuracy_score(yt, yp)), 4),
                "sensitivity": round(sensitivity, 4),
                "specificity": round(specificity, 4),
                "auc": round(auc, 4) if auc else None,
            }
        return result

    # ------------------------------------------------------------------
    # Full report
    # ------------------------------------------------------------------

    def full_report(self, y_true, y_pred, y_proba, groups_dict: dict) -> dict:
        """
        groups_dict: {attribute_name: array_of_group_labels}
        """
        report = {}
        for attr, groups in groups_dict.items():
            dpd = self.demographic_parity_difference(y_pred, groups)
            eod = self.equalized_odds_difference(y_true, y_pred, groups)
            gp  = self.group_performance(y_true, y_pred, y_proba, groups)
            report[attr] = {
                "demographic_parity_difference": dpd,
                "dp_pass": dpd <= self.THRESHOLD_DEMOGRAPHIC_PARITY,
                "equalized_odds": eod,
                "eo_pass": (eod["tpr_difference"] <= self.THRESHOLD_EQUALIZED_ODDS
                            and eod["fpr_difference"] <= self.THRESHOLD_EQUALIZED_ODDS),
                "group_performance": gp,
            }
        overall_pass = all(
            v["dp_pass"] and v["eo_pass"] for v in report.values()
        )
        report["_summary"] = {
            "overall_fairness_pass": overall_pass,
            "attributes_checked": list(groups_dict.keys()),
        }
        return report
