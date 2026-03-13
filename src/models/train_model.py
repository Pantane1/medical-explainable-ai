"""
Standalone training script.
Usage: python src/models/train_model.py --model decision_tree
"""
import argparse
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.models.explainable_model import ExplainableMedicalAI
from src.utils.data_loader import load_sample_data
from src.utils.preprocessor import Preprocessor


def train(model_type: str = "decision_tree", output_dir: str = "models/trained"):
    os.makedirs(output_dir, exist_ok=True)

    print("Loading data...")
    X_train, X_test, y_train, y_test, feature_names = load_sample_data()

    print("Preprocessing...")
    prep = Preprocessor(feature_names)
    X_train = prep.fit_transform(X_train)
    X_test  = prep.transform(X_test)

    print(f"Training {model_type}...")
    model = ExplainableMedicalAI(model_type=model_type)
    model.train(X_train, y_train, feature_names)

    metrics = model.evaluate(X_test, y_test)
    print(f"  Accuracy : {metrics['accuracy']}")
    print(f"  ROC-AUC  : {metrics['roc_auc']}")

    save_path = os.path.join(output_dir, f"{model_type}_v1.pkl")
    model.save(save_path)
    print(f"  Saved → {save_path}")
    return model, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an ExplainableMedicalAI model")
    parser.add_argument("--model", default="decision_tree",
                        choices=["decision_tree", "random_forest", "logistic"])
    parser.add_argument("--output", default="models/trained")
    args = parser.parse_args()
    train(args.model, args.output)
