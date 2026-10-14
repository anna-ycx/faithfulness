"""
Count counterfactual pairs in which each feature changes (concealment plan, Test 1).

The pair files only store text descriptions, plus the row ids
(original_question_idx / counterfactual_question_idx) of the two cases in the
dataset's binned feature table. This script rebuilds that table with the
dataset's own loader (src/templates/<dataset>.py: load_dataset), looks up each
case's features by row id, and compares them.

Safety check: every description is regenerated from its table row with the
template's description_generator and must match the stored text exactly.

Works for: attrition, income, heart_disease, pima_diabetes,
breast_cancer_recurrence. (bank_marketing row ids no longer line up with its
loader; moral_machines has no feature table.)

Usage:
    python -m concealment.pairs                 # attrition + income
    python -m concealment.pairs --dataset heart_disease
"""
import argparse
from pathlib import Path

import pandas as pd

from src.schema import CounterfactualDatabase

DATA = "data/natural_counterfactuals/{}_counterfactual_dataset_balanced.parquet"
SUPPORTED = ["attrition", "income", "heart_disease", "pima_diabetes", "breast_cancer_recurrence"]
OUT_DIR = Path("concealment/results/pair_counts")


def load_pairs_with_features(dataset: str):
    """
    Returns (pairs, orig, cf):
        pairs: the pair file, one row per stored pair
        orig, cf: feature tables aligned row-for-row with pairs
    Raises ValueError if any regenerated description differs from the stored text.
    """
    cls = CounterfactualDatabase.dataset_class_map[dataset]
    table = cls.load_dataset()
    features = [c for c in table.columns if c != "target"]
    pairs = pd.read_parquet(DATA.format(dataset))

    sides = {}
    for side in ["original", "counterfactual"]:
        idx = pairs[f"{side}_question_idx"].astype(int)
        rows = table.loc[idx, features].reset_index(drop=True)
        regenerated = [cls.description_generator(i, table.loc[i], features) for i in idx]
        mismatches = (pd.Series(regenerated) != pairs[f"{side}_description"].values).sum()
        if mismatches:
            raise ValueError(f"{dataset}: {mismatches} {side} descriptions don't match their table row; "
                             "the source data or loader has probably changed.")
        sides[side] = rows
    return pairs, sides["original"], sides["counterfactual"]


def distinct_pairs(pairs: pd.DataFrame) -> pd.Index:
    """Index of one row per unordered pair of descriptions. The files repeat each pair
    once per answer order, and often in both directions (A->B and B->A). Matched on
    text, not row id: some different rows bin to the same description, and the
    models only ever see the text."""
    key = [frozenset(p) for p in zip(pairs.original_description, pairs.counterfactual_description)]
    return pairs.assign(_key=key).drop_duplicates("_key").index


def summarise(dataset: str, min_pairs: int = 150) -> pd.DataFrame:
    pairs, orig, cf = load_pairs_with_features(dataset)
    keep = distinct_pairs(pairs)
    pairs, orig, cf = pairs.loc[keep], orig.loc[keep], cf.loc[keep]

    diffs = orig != cf
    n_changed = diffs.sum(axis=1)
    hamming = pairs.counterfactual_hamming_distance.fillna(-1).astype(int)
    label_flip = pairs.original_ground_truth != pairs.counterfactual_ground_truth

    rows = []
    for feat in diffs.columns:
        changes = diffs[feat]
        rows.append({
            "feature": feat,
            "pairs_changed": int(changes.sum()),
            "only_change": int((changes & (n_changed == 1)).sum()),
            "true_label_flips": f"{label_flip[changes].mean():.0%}" if changes.any() else "-",
            f"pass_>={min_pairs}": "yes" if changes.sum() >= min_pairs else "no",
        })
    table = pd.DataFrame(rows).sort_values("pairs_changed", ascending=False)

    summary = "\n".join([
        f"=== {dataset}: {len(pairs)} distinct pairs ===",
        "Descriptions match their table rows: all",
        f"Changed features match stored Hamming distance: {(n_changed == hamming).sum()}/{len(pairs)}",
        f"Pairs with no feature change: {(n_changed == 0).sum()}",
        table.to_string(index=False),
    ])
    print("\n" + summary)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_DIR / f"{dataset}.csv", index=False)
    (OUT_DIR / f"{dataset}_summary.txt").write_text(summary + "\n")
    print(f"Saved to {OUT_DIR}/{dataset}.csv and {dataset}_summary.txt")
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=SUPPORTED, nargs="+", default=["attrition", "income"])
    parser.add_argument("--min-pairs", type=int, default=150)
    args = parser.parse_args()
    for dataset in args.dataset:
        summarise(dataset, args.min_pairs)


if __name__ == "__main__":
    main()
