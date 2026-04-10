#!/usr/bin/env python3
"""
Split downloaded YouTube videos into sentence-level clips based on timestamps.

Usage:
    python3 scripts/split_videos.py
    python3 scripts/split_videos.py --splits data/cache/splits.csv --raw data/cache/raw_videos --dst data/cache/uncropped
    python3 scripts/split_videos.py --force eYEK-n2alOA
"""

import argparse
import csv
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


def read_splits(splits_csv):
    """Read split timestamps from splits.csv, grouped by video ID."""
    splits_csv = Path(splits_csv)
    videos = defaultdict(list)

    if not splits_csv.exists():
        print(f"ERROR: {splits_csv} not found. Run build_annotations_from_firebase.py first.", file=sys.stderr)
        sys.exit(1)

    with open(splits_csv) as f:
        reader = csv.DictReader(f, delimiter="|")
        for row in reader:
            name = row["name"]
            vid = name.split("/")[0]
            videos[vid].append({
                "name": name.split("/")[1],
                "video_id": vid,
                "start": float(row["start"]),
                "end": float(row["end"]),
            })

    return videos


def extract_clip(source: Path, clip_dir: Path, name: str, start: float, duration: float, force: bool = False):
    """Extract a clip from source video using ffmpeg with input seeking."""
    clip_dir.mkdir(parents=True, exist_ok=True)
    clip_path = clip_dir / f"{name}.mp4"

    if clip_path.exists() and not force:
        return True

    cmd = [
        "ffmpeg",
        "-y",
        "-ss", f"{start:.3f}",
        "-i", str(source),
        "-t", f"{duration:.3f}",
        "-c:v", "libx264",
        "-c:a", "aac",
        "-preset", "fast",
        "-loglevel", "error",
        str(clip_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"    [ERROR] ffmpeg failed for {name}: {result.stderr[:200]}", file=sys.stderr)
        return False
    return True


def split_all(splits_csv, raw_dir, dst, force_ids=None):
    """Split all videos into clips.

    Args:
        splits_csv: Path to splits.csv with timestamps.
        raw_dir: Directory with downloaded raw videos.
        dst: Output directory for clips.
        force_ids: Set of video IDs to re-split.

    Returns:
        dict with 'total', 'failed', 'skipped_videos'.
    """
    raw_dir, dst = Path(raw_dir), Path(dst)
    videos = read_splits(splits_csv)
    print(f"Found {sum(len(v) for v in videos.values())} annotations across {len(videos)} videos\n")

    total_clips = 0
    failed_clips = 0
    skipped_videos = []

    for video_id in sorted(videos.keys()):
        rows = videos[video_id]
        force = force_ids is not None and video_id in force_ids

        source = raw_dir / f"{video_id}.mp4"
        if not source.exists():
            skipped_videos.append(video_id)
            print(f"[{video_id}] raw video not found, skipping {len(rows)} clips")
            continue

        print(f"[{video_id}] {len(rows)} clips{' (force)' if force else ''}")

        for row in rows:
            start = row["start"]
            end = row["end"]
            duration = end - start
            if duration <= 0:
                print(f"    [WARN] Invalid duration for {row['name']}: {start}-{end}")
                failed_clips += 1
                continue

            ok = extract_clip(source, dst / video_id, row["name"], start, duration, force=force)
            if ok:
                total_clips += 1
            else:
                failed_clips += 1

    print(f"\nDone. {total_clips} clips, {failed_clips} failed, {len(skipped_videos)} videos skipped.")
    if skipped_videos:
        print(f"Skipped: {', '.join(skipped_videos)}")
    return {"total": total_clips, "failed": failed_clips, "skipped_videos": skipped_videos}


def main():
    ROOT = Path(__file__).resolve().parent.parent
    default_splits = ROOT / "data/cache/splits.csv"
    default_raw = ROOT / "data/cache/raw_videos"
    default_dst = ROOT / "data/cache/uncropped"

    parser = argparse.ArgumentParser(description="Split videos into sentence-level clips")
    parser.add_argument("--splits", type=Path, default=default_splits,
                        help=f"Path to splits.csv (default: {default_splits})")
    parser.add_argument("--raw", type=Path, default=default_raw,
                        help=f"Raw videos directory (default: {default_raw})")
    parser.add_argument("--dst", type=Path, default=default_dst,
                        help=f"Output directory (default: {default_dst})")
    parser.add_argument("--force", nargs="*", default=None,
                        help="Re-split specific video IDs (or all if no IDs given)")
    args = parser.parse_args()

    force_ids = None
    if args.force is not None:
        force_ids = set(args.force) if args.force else set(read_splits(args.splits).keys())

    split_all(splits_csv=args.splits, raw_dir=args.raw, dst=args.dst, force_ids=force_ids)


if __name__ == "__main__":
    main()
