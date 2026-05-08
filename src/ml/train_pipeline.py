"""
Full ML training pipeline.
Run: python src/ml/train_pipeline.py
"""

import os
import sys

ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
sys.path.insert(0, ROOT_DIR)

from src.config.settings import ROOT_DIR
from src.ml.classifier import train

TRAIN_CSV = os.path.join(
    ROOT_DIR, "data", "annotations",
    "classification_labels", "train_data.csv"
)
TEST_CSV = os.path.join(
    ROOT_DIR, "data", "annotations",
    "classification_labels", "test_data.csv"
)


def run_training():
    print("=" * 55)
    print("  ML Document Classifier — Training Pipeline")
    print("  Dataset: 20 Newsgroups + AG News (merged)")
    print("=" * 55)

    if not os.path.exists(TRAIN_CSV):
        print(f"\n❌ train_data.csv not found.")
        print("Run this first:")
        print("  python data/annotations/classification_labels/"
              "download_dataset.py")
        return

    results = train(TRAIN_CSV, TEST_CSV)

    print("\n" + "=" * 55)
    print("  Training Complete")
    print("=" * 55)
    
    print(f"  Best Model    : {results['best_model']}")
    print(f"  CV Accuracy   : {results['best_score']:.2%}")
    
    if "test_accuracy" in results:
        print(f"  Test Accuracy : {results['test_accuracy']:.2%}")
    
    print(f"  Classes       : {results['classes']}")
    
    print("=" * 55)
    print("\nNext: python src/ml/evaluate.py")


if __name__ == "__main__":
    run_training()