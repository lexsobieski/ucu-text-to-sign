#!/usr/bin/env python3
"""
Download Firebase RTDB export and save with today's date.

Usage:
    python scripts/download_firebase.py
    python scripts/download_firebase.py --date 2025-03-13
    python scripts/download_firebase.py --key path/to/serviceAccountKey.json

Saves to: data/firebase/YYYY-MM-DD.json
Updates symlink: data/firebase/latest.json
"""

import argparse
import json
import os
from datetime import date
from pathlib import Path

FIREBASE_DB_URL = "https://usleditordatabase-default-rtdb.europe-west1.firebasedatabase.app/"


def find_key_path(explicit_path=None, project_root=None):
    """Find Firebase service account key file."""
    if explicit_path:
        return Path(explicit_path)
    env_path = os.environ.get("FIREBASE_SERVICE_ACCOUNT_KEY")
    if env_path:
        return Path(env_path)
    if project_root:
        default = Path(project_root) / "serviceAccountKey.json"
        if default.exists():
            return default
    raise FileNotFoundError(
        "No service account key found. Provide --key, set FIREBASE_SERVICE_ACCOUNT_KEY, "
        "or place serviceAccountKey.json in the repo root."
    )


def summarize_export(data):
    """Print summary stats for a Firebase export."""
    videos = data.get("videos", {})
    if isinstance(videos, list):
        videos = {str(i): v for i, v in enumerate(videos) if v is not None}

    completed_videos = sum(
        1 for v in videos.values()
        if isinstance(v, dict) and v.get("complete")
    )

    video_captions = data.get("video_captions", {})
    total_captions = 0
    aligned_captions = 0
    for yt_id, cap_data in video_captions.items():
        captions = cap_data.get("captions", cap_data) if isinstance(cap_data, dict) else cap_data
        if not isinstance(captions, list):
            continue
        for c in captions:
            if isinstance(c, dict):
                total_captions += 1
                if c.get("aligned"):
                    aligned_captions += 1

    print(f"  Videos: {len(videos)} ({completed_videos} marked complete)")
    print(f"  Video captions entries: {len(video_captions)}")
    print(f"  Total captions: {total_captions}")
    print(f"  Aligned captions: {aligned_captions}")
    return {
        "videos": len(videos),
        "completed_videos": completed_videos,
        "captions": total_captions,
        "aligned": aligned_captions,
    }


def download_and_save(output_dir, key_path=None, export_date=None):
    """Download Firebase RTDB export and save to a dated JSON file.

    Args:
        output_dir: Directory to save the export JSON.
        key_path: Path to Firebase service account key (auto-detected if None).
        export_date: Date string for filename (default: today).

    Returns the path to the saved file.
    """
    import firebase_admin
    from firebase_admin import credentials, db

    key_path = find_key_path(key_path, project_root=Path(output_dir).parent.parent)
    output_dir = Path(output_dir)
    export_date = export_date or date.today().isoformat()

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{export_date}.json"

    # Initialize Firebase
    if not firebase_admin._apps:
        cred = credentials.Certificate(str(key_path))
        firebase_admin.initialize_app(cred, {"databaseURL": FIREBASE_DB_URL})

    # Download
    print(f"Downloading Firebase RTDB...")
    data = db.reference("/").get()

    # Save
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Saved to {output_path}")

    # Update latest symlink
    latest = output_dir / "latest.json"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(output_path.name)
    print(f"Updated {latest} -> {output_path.name}")

    # Summary
    summarize_export(data)

    return output_path


def main():
    ROOT = Path(__file__).resolve().parent.parent
    default_output = ROOT / "data/firebase"

    parser = argparse.ArgumentParser(description="Download Firebase RTDB export")
    parser.add_argument("--key", type=str, default=None,
                        help="Path to Firebase service account key JSON")
    parser.add_argument("--date", type=str, default=None,
                        help="Date string for filename (default: today, YYYY-MM-DD)")
    parser.add_argument("--output-dir", type=Path, default=default_output,
                        help=f"Output directory (default: {default_output})")
    args = parser.parse_args()

    download_and_save(
        output_dir=args.output_dir,
        key_path=args.key,
        export_date=args.date,
    )


if __name__ == "__main__":
    main()
