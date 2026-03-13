"""
Data Loader — loads medical datasets from CSV, Parquet, or
uses the UCI Heart Disease dataset as a built-in fallback for demos.
"""
import logging
import os
import numpy as np

logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    "age", "sex", "chest_pain_type", "resting_bp", "cholesterol",
    "fasting_blood_sugar", "max_hr", "exercise_angina", "oldpeak",
]


def load_sample_data(test_size: float = 0.2, random_state: int = 42):
    """
    Load the UCI Heart Disease dataset via sklearn or generate synthetic data.
    Returns: X_train, X_test, y_train, y_test, feature_names
    """
    from sklearn.model_selection import train_test_split

    try:
        # Try to load from processed cache first
        processed_train = "data/processed/train_data.parquet"
        processed_test  = "data/processed/test_data.parquet"
        if os.path.exists(processed_train):
            import pandas as pd
            train_df = pd.read_parquet(processed_train)
            test_df  = pd.read_parquet(processed_test)
            feature_cols = [c for c in train_df.columns if c != "target"]
            X_train = train_df[feature_cols].values
            y_train = train_df["target"].values
            X_test  = test_df[feature_cols].values
            y_test  = test_df["target"].values
            logger.info("Loaded processed data from parquet.")
            return X_train, X_test, y_train, y_test, feature_cols
    except Exception as e:
        logger.warning(f"Could not load parquet: {e}")

    # Fallback — generate synthetic cardiovascular-like data
    logger.info("Generating synthetic demo data...")
    rng = np.random.default_rng(random_state)
    n = 800

    age        = rng.integers(25, 80,  n).astype(float)
    sex        = rng.integers(0,  2,   n).astype(float)
    cp         = rng.integers(0,  4,   n).astype(float)
    bp         = rng.normal(130, 20,   n).clip(80, 200)
    chol       = rng.normal(220, 50,   n).clip(100, 500)
    fbs        = rng.integers(0, 2,    n).astype(float)
    max_hr     = (220 - age) - rng.normal(10, 20, n)
    max_hr     = max_hr.clip(60, 210)
    exang      = rng.integers(0, 2,    n).astype(float)
    oldpeak    = rng.exponential(1.2,  n).clip(0, 6)

    # Simple logistic rule for labels
    log_odds = (-3.5
                + 0.04 * age
                + 0.3  * sex
                + 0.25 * cp
                + 0.015 * (bp - 120)
                + 0.005 * (chol - 200)
                + 0.2  * fbs
                - 0.012 * (max_hr - 120)
                + 0.6  * exang
                + 0.25 * oldpeak)
    prob = 1 / (1 + np.exp(-log_odds))
    y = (rng.random(n) < prob).astype(int)

    X = np.column_stack([age, sex, cp, bp, chol, fbs, max_hr, exang, oldpeak])
    return train_test_split(X, y, test_size=test_size, random_state=random_state,
                             stratify=y), FEATURE_NAMES


def load_from_csv(path: str, target_col: str = "target",
                  test_size: float = 0.2, random_state: int = 42):
    """Load from a CSV file and split into train/test."""
    import pandas as pd
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(path)
    feature_cols = [c for c in df.columns if c != target_col]
    X = df[feature_cols].values
    y = df[target_col].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y)
    logger.info(f"Loaded {len(df)} rows from {path}. Features: {feature_cols}")
    return X_train, X_test, y_train, y_test, feature_cols


def save_processed(X_train, X_test, y_train, y_test, feature_names: list):
    """Persist processed splits to parquet for reproducibility."""
    import pandas as pd
    os.makedirs("data/processed", exist_ok=True)
    pd.DataFrame(X_train, columns=feature_names).assign(
        target=y_train).to_parquet("data/processed/train_data.parquet", index=False)
    pd.DataFrame(X_test, columns=feature_names).assign(
        target=y_test).to_parquet("data/processed/test_data.parquet", index=False)
    logger.info("Processed data saved to data/processed/")
