import os
import sys
import pickle
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.feature_extraction.text import TfidfVectorizer 

import re 
import string
import nltk 
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

#Dowenload corpus of stopword and wordnet 
nltk.download('stopwords' , download_dir='../../data/nltk/stopwords') 
nltk.download('wordnet' , download_dir='../../data/nltk/wordnet')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


# Models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier
)

# Path setup
ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
sys.path.insert(0, ROOT_DIR)

from src.config.settings import ML_MODEL_PATH


# ────────────────────────────────────────────────
# Aggressive Text Preprocessing
# ────────────────────────────────────────────────
def preprocess_text(text: str) -> str:
    
    # 1. Lowercase
    text = text.lower()

    # 2. Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # 3. Remove emails
    text = re.sub(r"\S+@\S+", "", text)

    # 4. Remove numbers
    text = re.sub(r"\d+", "", text)

    # 5. Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # 6. Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # 7. Tokenization
    tokens = text.split()

    # 8. Remove stopwords + short tokens
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]

    # 9. Lemmatization
    tokens = [lemmatizer.lemmatize(t) for t in tokens]

    return " ".join(tokens)



# ────────────────────────────────────────────────
# Build Pipeline
# ────────────────────────────────────────────────
def build_pipeline(model):
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            preprocessor=preprocess_text,   
            max_features=50000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=2,
            strip_accents="unicode"
        )),
        ("clf", model)
    ])


# ────────────────────────────────────────────────
# Models + Hyperparameters
# ────────────────────────────────────────────────
def get_models_and_params():
    return {

        "logistic_regression": {
            "model": LogisticRegression(max_iter=1000),
            "params": {
                "clf__C": [0.1, 1, 5],
                "clf__class_weight": [None, "balanced"]
            }
        },

        "svm": {
           "model": SVC(probability=True),
           "params": {
                "clf__kernel": ["rbf"],
                "clf__C": [0.01, 0.1, 1, 10, 50, 100],
                "clf__gamma": ["scale", "auto", 0.001, 0.01, 0.1, 1],
                "clf__class_weight": [None, "balanced"]
            }
        },

        "naive_bayes": {
            "model": MultinomialNB(),
            "params": {
                "clf__alpha": [0.5, 1.0, 2.0]
            }
        },

        "decision_tree": {
            "model": DecisionTreeClassifier(random_state=42),
            "params": {
                "clf__max_depth": [None, 10, 20],
                "clf__min_samples_split": [2, 5, 10]
            }
        },

        "random_forest": {
            "model": RandomForestClassifier(random_state=42),
            "params": {
                "clf__n_estimators": [100, 200],
                "clf__max_depth": [None, 10],
                "clf__min_samples_split": [2, 5]
            }
        },

        "gradient_boosting": {
            "model": GradientBoostingClassifier(),
            "params": {
                "clf__n_estimators": [100, 200],
                "clf__learning_rate": [0.05, 0.1],
                "clf__max_depth": [3, 5]
            }
        }
    }


# ────────────────────────────────────────────────
# Train Function
# ────────────────────────────────────────────────
def train(train_csv: str, test_csv: str = None):

    print(f"\nLoading: {train_csv}")
    df = pd.read_csv(train_csv).dropna(subset=["text", "label"])

    X = df["text"].tolist()
    y = df["label"].tolist()

    print(f"Samples: {len(X)}")
    print(f"Classes: {set(y)}")
    print(df["label"].value_counts())

    models_dict = get_models_and_params()

    best_model = None
    best_score = 0
    best_name  = ""

    results = {}

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # ────────────────────────────────────────────────
    # Train & Tune Each Model
    # ────────────────────────────────────────────────
    for name, mp in models_dict.items():
        print(f"\n🔍 Training: {name}")

        pipeline = build_pipeline(mp["model"])

        grid = GridSearchCV(
            pipeline,
            param_grid=mp["params"],
            cv=cv,
            scoring="accuracy",
            n_jobs=-1,
            verbose=1
        )

        grid.fit(X, y)

        print(f"Best Score ({name}): {grid.best_score_:.4f}")
        print(f"Best Params: {grid.best_params_}")

        results[name] = {
            "score": grid.best_score_,
            "params": grid.best_params_
        }

        if grid.best_score_ > best_score:
            best_score = grid.best_score_
            best_model = grid.best_estimator_
            best_name  = name

    # ────────────────────────────────────────────────
    # Voting Ensemble
    # ────────────────────────────────────────────────
    print("\n🔗 Building Voting Ensemble...")

    sorted_models = sorted(results.items(), key=lambda x: x[1]["score"], reverse=True)

    top_models = []
    for name, _ in sorted_models[:3]:
        model_pipeline = build_pipeline(models_dict[name]["model"])
        model_pipeline.fit(X, y)
        top_models.append((name, model_pipeline))

    voting_clf = VotingClassifier(
        estimators=top_models,
        voting='hard',
        n_jobs=-1
    )

    voting_score = np.mean(
        cross_val_score(voting_clf, X, y, cv=cv, scoring="accuracy")
    )

    print(f"Voting Score: {voting_score:.4f}")

    if voting_score > best_score:
        print("🏆 Voting Ensemble is BEST")
        best_model = voting_clf
        best_name = "voting_ensemble"
        best_score = voting_score

    print(f"\n🏆 FINAL BEST MODEL: {best_name} ({best_score:.4f})")

    # ────────────────────────────────────────────────
    # Test Evaluation
    # ────────────────────────────────────────────────
    if test_csv and os.path.exists(test_csv):
        print("\nEvaluating on test set...")

        test_df = pd.read_csv(test_csv).dropna(subset=["text", "label"])

        X_test = test_df["text"].tolist()
        y_test = test_df["label"].tolist()

        y_pred = best_model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)

        print(f"\nTest Accuracy: {acc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

        results["best_model"] = best_name
        results["best_score"] = best_score
        results["classes"] = list(set(y))

    save_model(best_model)

    return results


# ────────────────────────────────────────────────
# Save / Load
# ────────────────────────────────────────────────
def save_model(model):
    os.makedirs(os.path.dirname(ML_MODEL_PATH), exist_ok=True)

    with open(ML_MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    print(f"\n✅ Best model saved → {ML_MODEL_PATH}")


def load_model():
    if not os.path.exists(ML_MODEL_PATH):
        raise FileNotFoundError("Model not found. Train first.")

    with open(ML_MODEL_PATH, "rb") as f:
        return pickle.load(f)