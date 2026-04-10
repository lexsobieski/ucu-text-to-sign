#!/usr/bin/env python3
"""
Build annotation CSV from the Firebase RTDB export.

Reads aligned captions from the export JSON and produces:
  - data/usl-suspilne/annotations.csv  (name|text|annotator — for training)
  - data/cache/splits.csv              (name|video|start|end — for download_and_split.py)

Usage:
    python scripts/build_annotations_from_firebase.py
    python scripts/build_annotations_from_firebase.py --export data/firebase/2025-03-11.json
"""

import argparse
import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def extract_video_id(url):
    """YouTube URL -> video ID."""
    m = re.search(r'(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})', url)
    return m.group(1) if m else None


def build_annotations(export_path, output_csv, splits_csv,
                      include_all=False, min_duration=0.3):
    """Build annotations from Firebase export.

    Args:
        export_path: Path to Firebase RTDB export JSON.
        output_csv: Output path for annotations.csv.
        splits_csv: Output path for splits.csv.
        include_all: Include non-aligned captions.
        min_duration: Minimum clip duration in seconds.

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

    # Build URL -> annotator mapping
    yt_id_to_annotator = {}
    for v in videos.values():
        if not isinstance(v, dict):
            continue
        yt_id = extract_video_id(v.get("url", ""))
        if yt_id:
            yt_id_to_annotator[yt_id] = v.get("assigned_to", "unknown")

    # Build rows
    rows = []
    for yt_id, cap_data in sorted(video_captions.items()):
        captions = cap_data.get("captions", cap_data) if isinstance(cap_data, dict) else cap_data

        clip_idx = 0
        for c in captions:
            if not isinstance(c, dict):
                continue
            if not include_all and not c.get("aligned"):
                continue

            start = c.get("start_time", 0)
            end = c.get("end_time", 0)
            if end - start < min_duration:
                clip_idx += 1
                continue

            name = f"{yt_id}/{clip_idx:04d}"
            rows.append({
                "name": name,
                "video": f"{name}.mp4",
                "start": round(start, 3),
                "end": round(end, 3),
                "annotator": yt_id_to_annotator.get(yt_id, "unknown"),
                "text": c.get("text", "").strip(),
            })
            clip_idx += 1

    # Write annotations CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "text", "annotator"], delimiter="|")
        writer.writeheader()
        for r in rows:
            writer.writerow({"name": r["name"], "text": r["text"], "annotator": r["annotator"]})

    # Write splits CSV
    splits_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(splits_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "video", "start", "end"], delimiter="|")
        writer.writeheader()
        for r in rows:
            writer.writerow({"name": r["name"], "video": r["video"], "start": r["start"], "end": r["end"]})

    # Stats
    videos_seen = set(r["name"].split("/")[0] for r in rows)
    print(f"Wrote {len(rows)} annotations from {len(videos_seen)} videos to {output_csv}")
    for vid in sorted(videos_seen):
        n = sum(1 for r in rows if r["name"].startswith(vid + "/"))
        print(f"  {vid}: {n} clips")

    return {"rows": len(rows), "videos": videos_seen}


def main():
    default_export = ROOT / "data/firebase/latest.json"
    default_output = ROOT / "data/usl-suspilne/annotations.csv"
    default_splits = ROOT / "data/cache/splits.csv"

    parser = argparse.ArgumentParser(description="Build annotations from Firebase export")
    parser.add_argument("--export", type=Path, default=default_export,
                        help=f"Path to Firebase RTDB export JSON (default: {default_export})")
    parser.add_argument("--output", type=Path, default=default_output,
                        help=f"Output annotations CSV (default: {default_output})")
    parser.add_argument("--splits", type=Path, default=default_splits,
                        help=f"Output splits CSV (default: {default_splits})")
    parser.add_argument("--all", action="store_true",
                        help="Include non-aligned captions too")
    parser.add_argument("--min-duration", type=float, default=0.3,
                        help="Minimum clip duration in seconds (default: 0.3)")
    args = parser.parse_args()

    build_annotations(
        export_path=args.export,
        output_csv=args.output,
        splits_csv=args.splits,
        include_all=args.all,
        min_duration=args.min_duration,
    )


if __name__ == "__main__":
    main()
