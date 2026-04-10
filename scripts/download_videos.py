#!/usr/bin/env python3
"""
Download YouTube videos listed in splits.csv.

Usage:
    python3 scripts/download_videos.py
    python3 scripts/download_videos.py --splits data/cache/splits.csv --dst data/cache/raw_videos
    python3 scripts/download_videos.py --force eYEK-n2alOA

Requirements:
    pip install yt-dlp
"""

import argparse
import csv
import subprocess
import sys
from pathlib import Path


def get_video_ids(splits_csv):
    """Get unique video IDs from splits.csv."""
    splits_csv = Path(splits_csv)
    if not splits_csv.exists():
        print(f"ERROR: {splits_csv} not found. Run build_annotations_from_firebase.py first.", file=sys.stderr)
        sys.exit(1)

    video_ids = set()
    with open(splits_csv) as f:
        reader = csv.DictReader(f, delimiter="|")
        for row in reader:
            video_ids.add(row["name"].split("/")[0])
    return sorted(video_ids)


def download_video(video_id, dst, force=False):
    """Download a YouTube video. Returns path on success, None on failure."""
    dst = Path(dst)
    out_path = dst / f"{video_id}.mp4"
    if out_path.exists() and not force:
        print(f"  [skip] {out_path} already exists")
        return out_path

    dst.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    url = f"https://www.youtube.com/watch?v={video_id}"
    cmd = [
        "yt-dlp",
        "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--merge-output-format", "mp4",
        "-o", str(out_path),
        url,
    ]
    print(f"  Downloading {url} ...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [ERROR] yt-dlp failed for {video_id}:", file=sys.stderr)
        print(result.stderr[:500], file=sys.stderr)
        return None
    return out_path


def download_all(splits_csv, dst, force_ids=None):
    """Download all videos.

    Args:
        splits_csv: Path to splits.csv with clip names.
        dst: Output directory for downloaded videos.
        force_ids: Set of video IDs to re-download.

    Returns:
        dict with 'succeeded' and 'failed' video ID lists.
    """
    video_ids = get_video_ids(splits_csv)
    print(f"Found {len(video_ids)} videos in {splits_csv}\n")

    succeeded = []
    failed = []

    for vid in video_ids:
        force = force_ids is not None and vid in force_ids
        print(f"[{vid}]")
        path = download_video(vid, dst=dst, force=force)
        if path:
            succeeded.append(vid)
        else:
            failed.append(vid)

    print(f"\nDone. {len(succeeded)} downloaded, {len(failed)} failed.")
    if failed:
        print(f"Failed: {', '.join(failed)}")
    return {"succeeded": succeeded, "failed": failed}


def main():
    ROOT = Path(__file__).resolve().parent.parent
    default_splits = ROOT / "data/cache/splits.csv"
    default_dst = ROOT / "data/cache/raw_videos"

    parser = argparse.ArgumentParser(description="Download YouTube videos from splits.csv")
    parser.add_argument("--splits", type=Path, default=default_splits,
                        help=f"Path to splits.csv (default: {default_splits})")
    parser.add_argument("--dst", type=Path, default=default_dst,
                        help=f"Output directory (default: {default_dst})")
    parser.add_argument("--force", nargs="*", default=None,
                        help="Re-download specific video IDs (or all if no IDs given)")
    args = parser.parse_args()

    force_ids = None
    if args.force is not None:
        force_ids = set(args.force) if args.force else set(get_video_ids(args.splits))

    download_all(splits_csv=args.splits, dst=args.dst, force_ids=force_ids)


if __name__ == "__main__":
    main()
