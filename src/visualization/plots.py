"""
Static plots — matplotlib-based charts for reports and notebooks.
"""
import logging
import numpy as np

logger = logging.getLogger(__name__)


def plot_decision_tree(model, feature_names: list, class_names: list = None,
                       output_path: str = "logs/decision_tree.png"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.tree import plot_tree
    except ImportError:
        logger.warning("matplotlib or sklearn not available for tree plot.")
        return

    fig, ax = plt.subplots(figsize=(20, 10))
    plot_tree(model, feature_names=feature_names,
              class_names=class_names or ["Negative", "Positive"],
              filled=True, rounded=True, ax=ax, fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Decision tree saved to {output_path}")


def plot_roc_curve(y_true, y_proba, output_path: str = "logs/roc_curve.png"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve, auc
    except ImportError:
        return

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#63B3ED", lw=2, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], color="#718096", lw=1, linestyle="--")
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver Operating Characteristic")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info(f"ROC curve saved to {output_path}")


def plot_confusion_matrix(y_true, y_pred, output_path: str = "logs/confusion_matrix.png"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
    except ImportError:
        return

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Negative", "Positive"])
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info(f"Confusion matrix saved to {output_path}")
