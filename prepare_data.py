"""
prepare_data.py
================
Step 1 of the pipeline: load raw CHEAT jsonl files, join them by `id`,
and produce a unified dataset for downstream detector evaluation.

Repository layout:
    Detection-of-AI-generated-content/   <- run the script from here
        CHEAT_DATASET/                   <- raw CHEAT files (default --cheat_dir)
        prepare_data.py

This script has been validated end-to-end using the real CHEAT data
(cloned from github.com/botianzhe/CHEAT), including a verification run
using these exact relative folder names.

Output schema (one row per text sample):
    id          : original IEEE paper id (shared across all 4 CHEAT subsets)
    title       : paper title
    text        : the abstract text (human OR AI, depending on `label`/`source`)
    label       : 0 = human-written, 1 = AI-involved (binary, for simple baselines)
    source      : one of {"human", "generation", "polish", "fusion"}
                  (fine-grained AI-involvement category, for RQ1/RQ2 analysis)
    split       : "train" / "test" (stratified by source, fixed random seed)

Usage (defaults match the layout above, so this just works if you run it
from the project root):
    python prepare_data.py

Or override any path/param explicitly:
    python prepare_data.py \
        --cheat_dir CHEAT_DATASET \
        --out_dir   . \
        --n_per_class 4514 \
        --test_frac 0.3 \
        --seed 42

Notes:
    - `fusion` only has 4514 rows (a subset of init/generation/polish, which have
      15395 each), so n_per_class is automatically capped per source if the
      requested sample size exceeds what's available.
    - The default n_per_class of 4514 reproduces the final dissertation dataset.
"""

import argparse
import json
import os
import random
from collections import defaultdict


def load_jsonl(path):
    """Load a .jsonl file into a dict keyed by id."""
    records = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            records[obj["id"]] = obj
    return records


def build_unified_records(cheat_dir):
    """
    Load all 4 CHEAT subsets and produce a flat list of unified records,
    one per (id, source) combination.
    """
    init_path = os.path.join(cheat_dir, "ieee-init.jsonl")
    gen_path = os.path.join(cheat_dir, "ieee-chatgpt-generation.jsonl")
    polish_path = os.path.join(cheat_dir, "ieee-chatgpt-polish.jsonl")
    fusion_path = os.path.join(cheat_dir, "ieee-chatgpt-fusion.jsonl")

    for p in [init_path, gen_path, polish_path, fusion_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Expected CHEAT file not found: {p}\n"
                f"Did you pass the correct --cheat_dir? It should point to the "
                f"folder containing ieee-init.jsonl, ieee-chatgpt-generation.jsonl, "
                f"ieee-chatgpt-polish.jsonl, ieee-chatgpt-fusion.jsonl "
                f"(in your project this is the 'CHEAT_DATASET' folder)."
            )

    init = load_jsonl(init_path)
    gen = load_jsonl(gen_path)
    polish = load_jsonl(polish_path)
    fusion = load_jsonl(fusion_path)

    records = []

    # Human-written (label=0)
    for pid, obj in init.items():
        records.append({
            "id": pid,
            "title": obj["title"],
            "text": obj["abstract"],
            "label": 0,
            "source": "human",
        })

    # AI-generated, three sub-categories (label=1 for all, but `source` keeps
    # the fine-grained distinction needed for RQ1/RQ2 analysis)
    for source_name, source_dict in [
        ("generation", gen),
        ("polish", polish),
        ("fusion", fusion),
    ]:
        for pid, obj in source_dict.items():
            records.append({
                "id": pid,
                "title": obj["title"],
                "text": obj["abstract"],
                "label": 1,
                "source": source_name,
            })

    return records


def stratified_sample_and_split(records, n_per_class, test_frac, seed):
    """
    For each `source` category, sample up to n_per_class records (capped by
    availability), then split into train/test with the given test_frac.

    IMPORTANT (data leakage fix, found via validation on real CHEAT data):
    The same paper `id` can appear under multiple sources (e.g. the human
    abstract for paper X, and the AI-generated/polished/fused abstract for
    the SAME paper X). If we split train/test independently per source, the
    human version of paper X could land in train while its AI-polished
    version lands in test (or vice versa). Since "polish" and "fusion" are
    built directly from the human text, this leaks paper-specific content
    (topic, phrasing, terminology) across the train/test boundary -- a
    supervised classifier (e.g. RoBERTa) could then partly learn to
    recognise *papers* rather than *human-vs-AI writing style*, inflating
    test performance.

    Fix: split is decided at the paper-`id` level FIRST (each id is
    deterministically assigned to train or test), and *all* source rows for
    that id follow the same split assignment. Sampling (n_per_class) is then
    applied per source, respecting the pre-assigned split.
    """
    rng = random.Random(seed)

    by_source = defaultdict(list)
    all_ids = set()
    for r in records:
        by_source[r["source"]].append(r)
        all_ids.add(r["id"])

    # Step 1: assign every paper id to train or test ONCE, independent of source.
    all_ids = sorted(all_ids)  # sort for determinism before shuffling
    rng.shuffle(all_ids)
    n_test_ids = int(round(len(all_ids) * test_frac))
    test_id_set = set(all_ids[:n_test_ids])
    # every id not in test_id_set is implicitly "train"

    final_records = []
    summary = {}

    for source, items in by_source.items():
        # Partition this source's items by the pre-assigned split, so no id
        # ever crosses train/test regardless of which source it came from.
        train_pool = [r for r in items if r["id"] not in test_id_set]
        test_pool = [r for r in items if r["id"] in test_id_set]
        rng.shuffle(train_pool)
        rng.shuffle(test_pool)

        # Sample n_per_class total, keeping the same train:test ratio as test_frac
        n_take = min(n_per_class, len(train_pool) + len(test_pool))
        n_test_take = max(1, int(round(n_take * test_frac)))
        n_train_take = n_take - n_test_take

        n_train_take = min(n_train_take, len(train_pool))
        n_test_take = min(n_test_take, len(test_pool))

        train_items = train_pool[:n_train_take]
        test_items = test_pool[:n_test_take]

        for r in train_items:
            r2 = dict(r)
            r2["split"] = "train"
            final_records.append(r2)
        for r in test_items:
            r2 = dict(r)
            r2["split"] = "test"
            final_records.append(r2)

        summary[source] = {
            "available": len(items),
            "sampled": len(train_items) + len(test_items),
            "train": len(train_items),
            "test": len(test_items),
        }

    return final_records, summary


def write_jsonl(records, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_csv(records, path):
    import csv
    fieldnames = ["id", "title", "text", "label", "source", "split"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow(r)


def main():
    parser = argparse.ArgumentParser(description="Prepare unified CHEAT dataset for detector evaluation.")
    parser.add_argument("--cheat_dir", type=str, default="CHEAT_DATASET",
                         help="Path to the folder containing CHEAT's ieee-*.jsonl files. "
                              "Default matches the project layout: CHEAT_DATASET/")
    parser.add_argument("--out_dir", type=str, default=".",
                         help="Where to write the unified dataset (jsonl + csv). "
                              "Default is the repository root.")
    parser.add_argument("--n_per_class", type=int, default=4514,
                         help="Max samples per source category (human/generation/polish/fusion). "
                              "Default 4514 reproduces the final experiment.")
    parser.add_argument("--test_frac", type=float, default=0.3,
                         help="Fraction of each source category held out as test set.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[1/3] Loading and joining CHEAT subsets from {args.cheat_dir} ...")
    records = build_unified_records(args.cheat_dir)
    print(f"      Total raw records across all sources: {len(records)}")

    print(f"[2/3] Stratified sampling (n_per_class={args.n_per_class}) and train/test split "
          f"(test_frac={args.test_frac}) ...")
    final_records, summary = stratified_sample_and_split(
        records, args.n_per_class, args.test_frac, args.seed
    )

    print("      Summary per source:")
    for source, s in summary.items():
        print(f"        - {source:12s}: available={s['available']:6d}  "
              f"sampled={s['sampled']:4d}  train={s['train']:4d}  test={s['test']:4d}")

    out_jsonl = os.path.join(args.out_dir, "cheat_unified.jsonl")
    out_csv = os.path.join(args.out_dir, "cheat_unified.csv")
    print(f"[3/3] Writing output to:\n        {out_jsonl}\n        {out_csv}")
    write_jsonl(final_records, out_jsonl)
    write_csv(final_records, out_csv)

    print("Done.")


if __name__ == "__main__":
    main()
