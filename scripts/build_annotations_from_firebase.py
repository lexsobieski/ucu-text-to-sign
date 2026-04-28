#!/usr/bin/env python3
"""
Build annotation CSV from the Firebase RTDB export.

Reads aligned captions from the export JSON and produces:
  - data/cache/annotations.csv  (name|text|text_norm|annotator)
  - data/cache/splits.csv       (name|video|start|end|split — clip manifest with split assignment)

The published per-split CSVs (train/dev/test/test_unseen.csv) are written later
by write_dataset_splits.py, after identify_signers.py has produced signers.json.

Usage:
    python scripts/build_annotations_from_firebase.py
    python scripts/build_annotations_from_firebase.py --export data/firebase/2025-03-11.json
"""

import argparse
import csv
import json
import random
import re
from pathlib import Path

import num2words
import pymorphy3

ROOT = Path(__file__).resolve().parent.parent

_morph = pymorphy3.MorphAnalyzer(lang='uk')

# Patterns for number detection
_RE_CARDINAL = re.compile(r'^\d+$')
_RE_ORDINAL = re.compile(r'^(\d+)-([а-яіїєґ]+)$')
_RE_TIME = re.compile(r'^(\d{1,2})\.(\d{2})$')
_RE_RANGE = re.compile(r'^(\d+)-(\d+)$')
_RE_DECIMAL = re.compile(r'^(\d+),(\d+)$')
_RE_MILITARY = re.compile(r'^[А-ЯІЇЄҐA-Z][а-яіїєґa-z]*-\d+$')
_UNIT_SUFFIXES = {"мм", "см", "км", "кг", "мг"}


def _inflect_ordinal(number: int, suffix: str) -> str:
    """Convert number to ordinal and inflect last word to match the suffix."""
    ordinal = num2words.num2words(number, lang='uk', to='ordinal')
    words = ordinal.split()
    last = words[-1]
    parsed = _morph.parse(last)[0]
    for form in parsed.lexeme:
        if form.word.endswith(suffix):
            words[-1] = form.word
            return " ".join(words)
    return " ".join(words)


def _convert_number_token(token: str) -> str:
    """Convert a single token containing digits to words.

    Returns the original token if no conversion rule matches.
    """
    # Time: 21.20 → двадцять одна двадцять
    m = _RE_TIME.match(token)
    if m:
        h, mi = int(m.group(1)), int(m.group(2))
        if 0 <= h <= 23 and 0 <= mi <= 59:
            parts = [num2words.num2words(h, lang='uk')]
            if mi > 0:
                parts.append(num2words.num2words(mi, lang='uk'))
            return " ".join(parts)

    # Decimal: 4,5 → чотири цілих п'ять
    m = _RE_DECIMAL.match(token)
    if m:
        whole, frac = m.group(1), m.group(2)
        return (num2words.num2words(int(whole), lang='uk')
                + " цілих "
                + num2words.num2words(int(frac), lang='uk'))

    # Range: 30-40 → тридцять сорок
    m = _RE_RANGE.match(token)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        return (num2words.num2words(a, lang='uk')
                + " "
                + num2words.num2words(b, lang='uk'))

    # Ordinal with suffix: 26-го → двадцять шостого
    m = _RE_ORDINAL.match(token)
    if m:
        number, suffix = int(m.group(1)), m.group(2)
        # True ordinal endings are 1-2 chars (го, ї, й, му, та, те, ю, х, ти)
        # Anything longer or unit suffixes — compound modifier, use cardinal + suffix
        if len(suffix) > 2 or suffix in _UNIT_SUFFIXES:
            return num2words.num2words(number, lang='uk') + " " + suffix
        return _inflect_ordinal(number, suffix)

    # Bare cardinal: 40 → сорок
    if _RE_CARDINAL.match(token):
        return num2words.num2words(int(token), lang='uk')

    return token


def normalize_text(text: str) -> str:
    """Lowercase, convert numbers to words, strip edge punctuation."""
    # Expand % before tokenizing (otherwise edge-strip removes it)
    text = re.sub(r'(\d)%', r'\1 відсотків', text)
    raw_tokens = text.split()
    cleaned = []
    for raw_t in raw_tokens:
        # Strip edge punctuation on original casing (for military detection)
        stripped = re.sub(r'^[^\w]+|[^\w]+$', '', raw_t, flags=re.UNICODE)
        if not stripped:
            continue
        # Detect military/model designations before lowercasing
        if re.search(r'\d', stripped) and _RE_MILITARY.match(stripped):
            cleaned.append(stripped.lower())
            continue
        t = stripped.lower()
        if re.search(r'\d', t):
            t = _convert_number_token(t)
        cleaned.append(t)
    return " ".join(cleaned)


# =====================================================================
# Split strategies
#
# Each strategy is a callable that takes the full list of row dicts and
# returns a dict {"train", "dev", "test", "test_unseen"} of row lists.
# Add a new strategy by writing a function and registering it in
# SPLIT_STRATEGIES below.
# =====================================================================

# Hardcoded video-level partition (the historical 25-video split):
DEV_VIDEOS = {"6O0ZiSgKJNc", "cNT6ajjEwVU"}
TEST_VIDEOS = {"0ULOz5HM4pA", "SG9xYYOLBNI"}
# Held-out signer (s3): used for signer-independent evaluation.
TEST_UNSEEN_VIDEOS = {"82dy0zC6X_8"}


def videolevel_split(rows, **kwargs):
    """Assign each row to train/dev/test/test_unseen by its video ID,
    using the hardcoded DEV_VIDEOS / TEST_VIDEOS / TEST_UNSEEN_VIDEOS sets.
    Anything not matched falls into train."""
    out = {"train": [], "dev": [], "test": [], "test_unseen": []}
    for r in rows:
        vid = r["name"].split("/")[0]
        if vid in DEV_VIDEOS:
            out["dev"].append(r)
        elif vid in TEST_VIDEOS:
            out["test"].append(r)
        elif vid in TEST_UNSEEN_VIDEOS:
            out["test_unseen"].append(r)
        else:
            out["train"].append(r)
    return out


def cliplevel_random_split(rows, seed: int = 42,
                           ratios=(0.8, 0.1, 0.1), **kwargs):
    """Random clip-level 80/10/10 split. test_unseen is left empty."""
    assert abs(sum(ratios) - 1.0) < 1e-6, f"ratios must sum to 1, got {ratios}"
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * ratios[0])
    n_dev = int(n * ratios[1])
    return {
        "train": shuffled[:n_train],
        "dev": shuffled[n_train:n_train + n_dev],
        "test": shuffled[n_train + n_dev:],
        "test_unseen": [],
    }


SPLIT_STRATEGIES = {
    "video": videolevel_split,
    "random": cliplevel_random_split,
}


def extract_video_id(url):
    """YouTube URL -> video ID."""
    m = re.search(r'(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})', url)
    return m.group(1) if m else None


def build_annotations(export_path, output_csv, splits_csv,
                      include_all=False,
                      min_duration=0.3, max_duration=60.0,
                      split_strategy="video", split_seed=42):
    """Build annotations from Firebase export.

    Args:
        export_path: Path to Firebase RTDB export JSON.
        output_csv: Output path for annotations.csv (name|text|text_norm|annotator).
        splits_csv: Output path for splits.csv — clip manifest with the split
            assignment recorded as an extra column (name|video|start|end|split).
            Consumed by download_videos.py, split_videos.py, and later by
            write_dataset_splits.py.
        include_all: Include non-aligned captions.
        min_duration: Minimum clip duration in seconds.
        max_duration: Maximum clip duration in seconds (filters unsegmented captions).
        split_strategy: Name of a registered split strategy (see SPLIT_STRATEGIES).
        split_seed: RNG seed for split strategies that need one.

    Returns dict with stats: {"rows": int, "videos": set}.
    """
    export_path = Path(export_path)
    output_csv = Path(output_csv)
    splits_csv = Path(splits_csv)

    with open(export_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    videos = data.get("videos", {})
    if isinstance(videos, list):
        videos = {str(i): v for i, v in enumerate(videos) if v is not None}

    video_captions = data.get("video_captions", {})

    # Build URL -> metadata mapping
    yt_id_to_annotator = {}
    completed_ids = set()
    for v in videos.values():
        if not isinstance(v, dict):
            continue
        yt_id = extract_video_id(v.get("url", ""))
        if yt_id:
            yt_id_to_annotator[yt_id] = v.get("assigned_to", "unknown")
            if v.get("complete"):
                completed_ids.add(yt_id)

    print(f"Videos: {len(completed_ids)} complete, {len(yt_id_to_annotator) - len(completed_ids)} incomplete")

    # Build rows (only from completed videos by default)
    rows = []
    skipped_incomplete = 0
    for yt_id, cap_data in sorted(video_captions.items()):
        if not include_all and yt_id not in completed_ids:
            skipped_incomplete += 1
            continue
        captions = cap_data.get("captions", cap_data) if isinstance(cap_data, dict) else cap_data

        clip_idx = 0
        for c in captions:
            if not isinstance(c, dict):
                continue
            if not include_all and not c.get("aligned"):
                continue

            start = c.get("start_time", 0)
            end = c.get("end_time", 0)
            dur = end - start
            if dur < min_duration or dur > max_duration:
                clip_idx += 1
                continue

            name = f"{yt_id}/{clip_idx:04d}"
            raw_text = c.get("text", "").strip()
            rows.append({
                "name": name,
                "video": f"{name}.mp4",
                "start": round(start, 3),
                "end": round(end, 3),
                "annotator": yt_id_to_annotator.get(yt_id, "unknown"),
                "text": raw_text,
                "text_norm": normalize_text(raw_text),
            })
            clip_idx += 1

    # Assign each row to a split using the chosen strategy.
    # The split is recorded in splits.csv so write_dataset_splits.py can route
    # rows later (after signer identification) without re-running the strategy.
    if split_strategy not in SPLIT_STRATEGIES:
        raise ValueError(f"unknown split_strategy {split_strategy!r}; "
                         f"available: {sorted(SPLIT_STRATEGIES)}")
    print(f"  split_strategy={split_strategy}")
    split_rows = SPLIT_STRATEGIES[split_strategy](rows, seed=split_seed)
    name_to_split = {r["name"]: split for split, items in split_rows.items() for r in items}

    # Write annotations CSV (clip-level metadata from Firebase, no signer info yet)
    annotations_fields = ["name", "text", "text_norm", "annotator"]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=annotations_fields, delimiter="|")
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r[k] for k in annotations_fields})

    # Write splits CSV (clip manifest + split assignment)
    splits_fields = ["name", "video", "start", "end", "split"]
    splits_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(splits_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=splits_fields, delimiter="|")
        writer.writeheader()
        for r in rows:
            writer.writerow({
                "name": r["name"],
                "video": r["video"],
                "start": r["start"],
                "end": r["end"],
                "split": name_to_split[r["name"]],
            })
    for split_name, items in split_rows.items():
        print(f"  {split_name}: {len(items)} clips")

    # Stats
    videos_seen = set(r["name"].split("/")[0] for r in rows)
    print(f"Wrote {len(rows)} annotations from {len(videos_seen)} videos to {output_csv}")
    if skipped_incomplete:
        print(f"Skipped {skipped_incomplete} incomplete videos (use --all to include)")
    for vid in sorted(videos_seen):
        n = sum(1 for r in rows if r["name"].startswith(vid + "/"))
        print(f"  {vid}: {n} clips")

    return {"rows": len(rows), "videos": videos_seen}


def main():
    default_export = ROOT / "data/firebase/latest.json"
    default_output = ROOT / "data/cache/annotations.csv"
    default_splits = ROOT / "data/cache/splits.csv"

    parser = argparse.ArgumentParser(description="Build annotations from Firebase export")
    parser.add_argument("--export", type=Path, default=default_export,
                        help=f"Path to Firebase RTDB export JSON (default: {default_export})")
    parser.add_argument("--output", type=Path, default=default_output,
                        help=f"Output annotations CSV (default: {default_output})")
    parser.add_argument("--splits", type=Path, default=default_splits,
                        help=f"Output splits CSV (default: {default_splits})")
    parser.add_argument("--all", action="store_true",
                        help="Include incomplete videos and non-aligned captions")
    parser.add_argument("--min-duration", type=float, default=0.3,
                        help="Minimum clip duration in seconds (default: 0.3)")
    parser.add_argument("--max-duration", type=float, default=60.0,
                        help="Maximum clip duration in seconds (default: 60.0)")
    parser.add_argument("--split-strategy", choices=sorted(SPLIT_STRATEGIES),
                        default="video",
                        help=f"How to assign clips to splits "
                             f"(default: video; choices: {sorted(SPLIT_STRATEGIES)})")
    parser.add_argument("--split-seed", type=int, default=42,
                        help="RNG seed for split strategies that need one (e.g. random)")
    args = parser.parse_args()

    build_annotations(
        export_path=args.export,
        output_csv=args.output,
        splits_csv=args.splits,
        include_all=args.all,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        split_strategy=args.split_strategy,
        split_seed=args.split_seed,
    )


if __name__ == "__main__":
    main()
