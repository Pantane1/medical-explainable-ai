"""
Preprocessor — handles scaling, imputation, and encoding
for the medical feature set. Saves/loads fitted transformers
so train and inference use identical transformations.
"""
import joblib
import logging
import numpy as np
import os

logger = logging.getLogger(__name__)

# Features that should be scaled to zero mean / unit variance
CONTINUOUS_FEATURES = [
    "age", "resting_bp", "cholesterol", "max_hr", "oldpeak"
]

# Features left as-is (binary / ordinal codes)
CATEGORICAL_FEATURES = [
    "sex", "chest_pain_type", "fasting_blood_sugar", "exercise_angina",
]


class Preprocessor:
    def __init__(self, feature_names: list):
        self.feature_names = feature_names
        self._scaler = None

    def fit_transform(self, X):
        from sklearn.preprocessing import StandardScaler
        from sklearn.impute import SimpleImputer

        X = np.array(X, dtype=float)

        # Impute missing values with median
        self._imputer = SimpleImputer(strategy="median")
        X = self._imputer.fit_transform(X)

        # Scale continuous features only
        self._scaler = StandardScaler()
        cont_idx = self._continuous_indices()
        if cont_idx:
            X[:, cont_idx] = self._scaler.fit_transform(X[:, cont_idx])

        logger.info("Preprocessor fitted and data transformed.")
        return X

    def transform(self, X):
        X = np.array(X, dtype=float)
        if self._imputer:
            X = self._imputer.transform(X)
        cont_idx = self._continuous_indices()
        if self._scaler and cont_idx:
            X[:, cont_idx] = self._scaler.transform(X[:, cont_idx])
        return X

    def _continuous_indices(self) -> list:
        return [i for i, n in enumerate(self.feature_names)
                if n in CONTINUOUS_FEATURES]

    def save(self, path: str = "models/trained/preprocessor.pkl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({"scaler": self._scaler,
                     "imputer": getattr(self, "_imputer", None),
                     "feature_names": self.feature_names}, path)
        logger.info(f"Preprocessor saved to {path}")

    @classmethod
    def load(cls, path: str) -> "Preprocessor":
        data = joblib.load(path)
        obj = cls(feature_names=data["feature_names"])
        obj._scaler  = data["scaler"]
        obj._imputer = data.get("imputer")
        logger.info(f"Preprocessor loaded from {path}")
        return obj

    def inverse_transform_feature(self, feature_name: str, value: float) -> float:
        """Convert a scaled value back to its original scale."""
        idx = self.feature_names.index(feature_name)
        cont_idx = self._continuous_indices()
        if idx not in cont_idx or self._scaler is None:
            return value
        pos = cont_idx.index(idx)
        mean = self._scaler.mean_[pos]
        std  = self._scaler.scale_[pos]
        return round(value * std + mean, 3)
