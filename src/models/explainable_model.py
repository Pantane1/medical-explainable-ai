"""
Explainable Medical AI — core model wrapper.
Supports decision_tree, random_forest, and logistic regression.
All models are chosen or constrained for interpretability.
"""
import joblib
import logging
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

logger = logging.getLogger(__name__)


class ExplainableMedicalAI:
    SUPPORTED_MODELS = ("decision_tree", "random_forest", "logistic")

    def __init__(self, model_type: str = "decision_tree"):
        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(f"model_type must be one of {self.SUPPORTED_MODELS}")
        self.model_type = model_type
        self.model = None
        self.feature_names: list = []
        self._build_model()

    def _build_model(self):
        if self.model_type == "decision_tree":
            self.model = DecisionTreeClassifier(
                max_depth=4,            # shallow → interpretable
                min_samples_split=20,
                random_state=42,
            )
        elif self.model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                min_samples_split=10,
                random_state=42,
            )
        elif self.model_type == "logistic":
            self.model = LogisticRegression(
                penalty="l1",           # L1 → sparse / interpretable
                solver="saga",
                max_iter=1000,
                random_state=42,
            )

    def train(self, X, y, feature_names: list):
        self.feature_names = feature_names
        self.model.fit(X, y)
        logger.info(f"Model '{self.model_type}' trained on {len(y)} samples.")
        return self

    def predict(self, X):
        return self.model.predict(np.atleast_2d(X))

    def predict_proba(self, X):
        return self.model.predict_proba(np.atleast_2d(X))[:, 1]

    def evaluate(self, X_test, y_test) -> dict:
        preds = self.predict(X_test)
        probas = self.predict_proba(X_test)
        return {
            "accuracy": round(accuracy_score(y_test, preds), 4),
            "roc_auc": round(roc_auc_score(y_test, probas), 4),
            "report": classification_report(y_test, preds, output_dict=True),
        }

    def feature_importance(self) -> dict:
        if hasattr(self.model, "feature_importances_"):
            return dict(zip(self.feature_names, self.model.feature_importances_))
        if hasattr(self.model, "coef_"):
            return dict(zip(self.feature_names, np.abs(self.model.coef_[0])))
        return {}

    def save(self, path: str):
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({"model": self.model, "feature_names": self.feature_names,
                     "model_type": self.model_type}, path)
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str) -> "ExplainableMedicalAI":
        data = joblib.load(path)
        obj = cls(model_type=data["model_type"])
        obj.model = data["model"]
        obj.feature_names = data["feature_names"]
        logger.info(f"Model loaded from {path}")
        return obj
