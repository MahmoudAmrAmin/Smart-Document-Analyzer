"""
Merges 20 Newsgroups (sklearn) + AG News (Kaggle CSV)
into a single balanced dataset of 15,000 samples.

Run: python data/annotations/classification_labels/download_dataset.py
"""

import os
import sys
import re
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from collections import Counter

ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
sys.path.insert(0, ROOT_DIR)

# ── Output paths ───────────────────────────────────────
OUTPUT_DIR = os.path.join(ROOT_DIR, "data", "annotations",
                          "classification_labels")
TRAIN_CSV  = os.path.join(OUTPUT_DIR, "train_data.csv")
TEST_CSV   = os.path.join(OUTPUT_DIR, "test_data.csv")

# ── AG News paths ──────────────────────────────────────
AGNEWS_TRAIN = os.path.join(ROOT_DIR, "data", "external",
                            "agnews", "train.csv")
AGNEWS_TEST  = os.path.join(ROOT_DIR, "data", "external",
                            "agnews", "test.csv")

# ── Samples per class in final balanced dataset ────────
SAMPLES_PER_CLASS_TRAIN = 4000   # 4000 × 3 = 12,000 train
SAMPLES_PER_CLASS_TEST  = 1000   # 1000 × 3 =  3,000 test


# ══════════════════════════════════════════════════════
#  MAPPING TABLES
# ══════════════════════════════════════════════════════

# 20 Newsgroups category → our label
NEWSGROUPS_MAP = {
    # CONTRACT — legal, political, policy
    "talk.politics.guns":     "contract",
    "talk.politics.mideast":  "contract",
    "talk.politics.misc":     "contract",
    "talk.religion.misc":     "contract",
    "soc.religion.christian": "contract",
    "alt.atheism":            "contract",
    # INVOICE — technical, commercial, transactional
    "sci.crypt":                  "invoice",
    "sci.electronics":            "invoice",
    "comp.sys.ibm.pc.hardware":   "invoice",
    "comp.sys.mac.hardware":      "invoice",
    "comp.os.ms-windows.misc":    "invoice",
    "misc.forsale":               "invoice",
    # RESEARCH — scientific, analytical, academic
    "sci.med":             "research",
    "sci.space":           "research",
    "rec.autos":           "research",
    "rec.motorcycles":     "research",
    "rec.sport.baseball":  "research",
    "rec.sport.hockey":    "research",
    "comp.graphics":       "research",
}

# AG News class_index → our label
# AG News classes: 1=World, 2=Sports, 3=Business, 4=Sci/Tech
AGNEWS_MAP = {
    1: "contract",   # World  → political/legal/international
    2: "research",   # Sports → analytical/statistical
    3: "invoice",    # Business → commercial/financial
    4: "invoice",    # Sci/Tech → technical/transactional
}


# ══════════════════════════════════════════════════════
#  CLEANERS
# ══════════════════════════════════════════════════════

def clean_newsgroup(text: str) -> str:
    """Strip headers, emails, URLs from newsgroup posts."""
    # Remove header block
    lines = text.split("\n")
    body, past_header = [], False
    for line in lines:
        if not past_header:
            if line.strip() == "":
                past_header = True
            continue
        body.append(line)
    text = " ".join(body)
    text = re.sub(r"[\w\.-]+@[\w\.-]+", " ", text)   # emails
    text = re.sub(r"http\S+", " ", text)              # URLs
    text = re.sub(r"[^a-zA-Z\s]", " ", text)          # non-alpha
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_agnews(text: str) -> str:
    """Clean AG News title+description text."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\\", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ══════════════════════════════════════════════════════
#  LOADERS
# ══════════════════════════════════════════════════════

def load_newsgroups(subset: str) -> pd.DataFrame:
    print(f"  Loading 20 Newsgroups ({subset})...")
    categories = list(NEWSGROUPS_MAP.keys())
    raw = fetch_20newsgroups(
        subset=subset,
        categories=categories,
        remove=("headers", "footers", "quotes"),
        random_state=42
    )
    rows = []
    for text, cat_idx in zip(raw.data, raw.target):
        category = raw.target_names[cat_idx]
        label    = NEWSGROUPS_MAP.get(category)
        if not label:
            continue
        cleaned = clean_newsgroup(text)
        if len(cleaned.split()) < 20:
            continue
        rows.append({"text": cleaned, "label": label, "source": "newsgroups"})

    df = pd.DataFrame(rows)
    print(f"    → {len(df)} usable samples")
    return df


def load_agnews(csv_path: str) -> pd.DataFrame:
    print(f"  Loading AG News: {csv_path}")

    if not os.path.exists(csv_path):
        print(f"    ❌ File not found: {csv_path}")
        return pd.DataFrame(columns=["text", "label", "source"])

    # AG News CSV columns: class_index, title, description
    df_raw = pd.read_csv(
        csv_path,
        header=None,
        names=["Class Index", "Title", "Description"]
    )

    rows = []
    for _, row in df_raw.iterrows():
        label = AGNEWS_MAP.get(row["Class Index"])
        if not label:
            continue
        # Combine title + description for richer text
        combined = f"{row['Title']} {row['Description']}"
        cleaned  = clean_agnews(combined)
        if len(cleaned.split()) < 10:
            continue
        rows.append({"text": cleaned, "label": label, "source": "agnews"})

    df = pd.DataFrame(rows)
    print(f"    → {len(df)} usable samples")
    return df


# ══════════════════════════════════════════════════════
#  MERGE + BALANCE
# ══════════════════════════════════════════════════════

def merge_and_balance(df: pd.DataFrame,
                      samples_per_class: int,
                      split_name: str) -> pd.DataFrame:
    print(f"\n  Balancing {split_name} set ({samples_per_class}/class)...")

    parts = []
    for label in ["contract", "invoice", "research"]:
        subset = df[df["label"] == label]
        n      = min(len(subset), samples_per_class)
        if n < samples_per_class:
            print(f"    ⚠️  {label}: only {n} available "
                  f"(wanted {samples_per_class})")
        sampled = subset.sample(n, random_state=42)
        parts.append(sampled)
        print(f"    {label:10s}: {n} samples "
              f"(newsgroups={len(subset[subset.source=='newsgroups'])}, "
              f"agnews={len(subset[subset.source=='agnews'])})")

    balanced = pd.concat(parts).sample(
        frac=1, random_state=42
    ).reset_index(drop=True)

    return balanced[["text", "label"]]   # drop source column for training


# ══════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════

def create_merged_dataset():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 55)
    print("  Building Merged Dataset")
    print("  Sources: 20 Newsgroups + AG News")
    print("=" * 55)

    # ── Load all data ──────────────────────────────────
    print("\n[1/4] Loading datasets...")
    ng_train = load_newsgroups("train")
    ng_test  = load_newsgroups("test")
    ag_train = load_agnews(AGNEWS_TRAIN)
    ag_test  = load_agnews(AGNEWS_TEST)

    # ── Combine ────────────────────────────────────────
    print("\n[2/4] Combining sources...")
    train_combined = pd.concat([ng_train, ag_train], ignore_index=True)
    test_combined  = pd.concat([ng_test,  ag_test],  ignore_index=True)

    print(f"\n  Combined train: {len(train_combined)} samples")
    print(f"  Distribution:\n"
          f"{train_combined['label'].value_counts().to_string()}")
    print(f"\n  Combined test: {len(test_combined)} samples")
    print(f"  Distribution:\n"
          f"{test_combined['label'].value_counts().to_string()}")

    # ── Balance ────────────────────────────────────────
    print("\n[3/4] Balancing classes...")
    train_final = merge_and_balance(
        train_combined, SAMPLES_PER_CLASS_TRAIN, "train"
    )
    test_final = merge_and_balance(
        test_combined, SAMPLES_PER_CLASS_TEST, "test"
    )

    # ── Save ───────────────────────────────────────────
    print("\n[4/4] Saving CSVs...")
    train_final.to_csv(TRAIN_CSV, index=False, encoding="utf-8")
    test_final.to_csv(TEST_CSV,   index=False, encoding="utf-8")

    print(f"\n{'='*55}")
    print(f"✅ Dataset ready!")
    print(f"   Train : {len(train_final):,} samples → {TRAIN_CSV}")
    print(f"   Test  : {len(test_final):,} samples  → {TEST_CSV}")
    print(f"\n   Final train distribution:")
    print(f"{train_final['label'].value_counts().to_string()}")
    print(f"\n   Final test distribution:")
    print(f"{test_final['label'].value_counts().to_string()}")
    print(f"{'='*55}")


if __name__ == "__main__":
    create_merged_dataset()