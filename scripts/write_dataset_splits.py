#!/usr/bin/env python3
"""
Write the publishable per-split CSVs (train/dev/test/test_unseen.csv).

Joins three intermediate artefacts:
  - data/cache/splits.csv     (name|video|start|end|split — from build_annotations)
  - data/cache/annotations.csv (name|text|text_norm|annotator — from build_annotations)
  - data/cache/signers.json    (per-clip signer ids — from identify_signers)

and writes one CSV per split to the dataset directory with schema
`name|text_norm|signer_id`. Runs after identify_signers.py so the
signer_id column is populated.

Usage:
    python scripts/write_dataset_splits.py
"""

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_signer_mapping(signers_path):
    """Load {clip_name -> signer_id} from identify_signers.py output.

    clip_name uses the same `{video_id}/{clip_idx:04d}` convention as splits.csv.
    Returns empty dict if the file is missing.
    """
    signers_path = Path(signers_path)
    if not signers_path.exists():
        print(f"  [WARN] signers file missing ({signers_path}); signer_id will be blank")
        return {}
    with open(signers_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    out = {}
    for vid, info in raw.items():
        for clip_file, signer_id in info.get("clips", {}).items():
            # clip_file is e.g. "0042.mp4"; strip extension to match splits.csv "name"
            stem = Path(clip_file).stem
            out[f"{vid}/{stem}"] = signer_id
    return out


def write_dataset_splits(splits_csv, annotations_csv, signers_path, dataset_dir):
    """Materialize the four published split CSVs.

    Args:
        splits_csv: Path to splits.csv (name|video|start|end|split).
        annotations_csv: Path to annotations.csv (name|text|text_norm|annotator).
        signers_path: Path to signers.json from identify_signers.py.
        dataset_dir: Directory to write {train,dev,test,test_unseen}.csv into.

    Returns dict {"counts": {split: n}}.
    """
    splits_csv = Path(splits_csv)
    annotations_csv = Path(annotations_csv)
    dataset_dir = Path(dataset_dir)

    with open(annotations_csv, "r", encoding="utf-8") as f:
        text_by_name = {row["name"]: row["text_norm"] for row in csv.DictReader(f, delimiter="|")}

    signer_by_name = _load_signer_mapping(signers_path)

    by_split = {"train": [], "dev": [], "test": [], "test_unseen": []}
    with open(splits_csv, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="|"):
            split = row["split"]
            if split not in by_split:
                raise ValueError(f"unknown split {split!r} in {splits_csv}")
            name = row["name"]
            signer_id = signer_by_name.get(name)
            by_split[split].append({
                "name": name,
                "text_norm": text_by_name.get(name, ""),
                "signer_id": signer_id if signer_id is not None else "",
            })

    fields = ["name", "text_norm", "signer_id"]
    dataset_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split_name, items in by_split.items():
        out_path = dataset_dir / f"{split_name}.csv"
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields, delimiter="|")
            writer.writeheader()
            writer.writerows(items)
        counts[split_name] = len(items)
        print(f"  {split_name}: {len(items)} clips -> {out_path}")

    return {"counts": counts}


def main():
    default_splits = ROOT / "data/cache/splits.csv"
    default_annotations = ROOT / "data/cache/annotations.csv"
    default_signers = ROOT / "data/cache/signers.json"
    default_dataset = ROOT / "data/usl-suspilne"

    parser = argparse.ArgumentParser(description="Write per-split CSVs from intermediates")
    parser.add_argument("--splits", type=Path, default=default_splits,
                        help=f"Path to splits.csv (default: {default_splits})")
    parser.add_argument("--annotations", type=Path, default=default_annotations,
                        help=f"Path to annotations.csv (default: {default_annotations})")
    parser.add_argument("--signers", type=Path, default=default_signers,
                        help=f"Path to signers.json (default: {default_signers}; "
                             f"missing file → blank signer_id)")
    parser.add_argument("--dataset-dir", type=Path, default=default_dataset,
                        help=f"Where to write per-split CSVs (default: {default_dataset})")
    args = parser.parse_args()

    write_dataset_splits(
        splits_csv=args.splits,
        annotations_csv=args.annotations,
        signers_path=args.signers,
        dataset_dir=args.dataset_dir,
    )


if __name__ == "__main__":
    main()