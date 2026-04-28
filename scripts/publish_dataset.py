#!/usr/bin/env python3
"""
Publish the working tree (data/cache/) as the released dataset (data/usl-suspilne/)
with sequential v-IDs replacing YouTube IDs.

The build pipeline writes everything to data/cache/ keyed by YouTube ID
(the natural ID coming out of Firebase + downloads). This script publishes
that working tree as the immutable release artefact:

  data/cache/features/{yt_id}/{clip}.mp4
                                     ──────►  data/usl-suspilne/features/v00/{clip}.mp4
  data/cache/poses/{repr}/{yt_id}/{clip}.npy
                                     ──────►  data/usl-suspilne/poses/{repr}/v00/{clip}.npy
  data/cache/{train,dev,test}.csv (YT IDs in `name`)
                                     ──────►  data/usl-suspilne/{train,dev,test}.csv (v-IDs)

The YT_ID ↔ v_ID mapping is persisted in data/cache/videos.json. It is
append-only across runs — once a video is assigned v05, it stays v05.

By default the script uses **hardlinks** between cache and the published tree
so you don't pay 2× disk cost. Hardlinks fall back to a normal copy if the
two paths sit on different filesystems.

Idempotent: a fully-published tree is a no-op on re-run; new clips in cache
get linked across; clips that disappear from cache do *not* get pruned from
the published tree (caller's responsibility — usually `rm -rf` the published
features/poses dirs before re-publishing).
"""

import argparse
import csv
import json
import os
import re
import shutil
from pathlib import Path

V_ID_RE = re.compile(r"^v\d{2,}$")
DEFAULT_SPLITS = ("train", "dev", "test")
DEFAULT_TREES = ("features", "poses/mediapipe_holistic", "poses/mediapipe_3d")


def _is_v_id(name):
    return V_ID_RE.match(name) is not None


def _load_or_init_mapping(videos_json):
    if videos_json.exists():
        return json.loads(videos_json.read_text())
    return {"videos": []}


def _save_mapping(videos_json, data):
    videos_json.parent.mkdir(parents=True, exist_ok=True)
    videos_json.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def _next_v_id(used):
    n = 0
    while True:
        candidate = f"v{n:02d}"
        if candidate not in used:
            return candidate
        n += 1


def _resolve_mapping(youtube_ids, mapping_data):
    yid_to_vid = {e["youtube_id"]: e["v_id"] for e in mapping_data["videos"]}
    used = set(yid_to_vid.values())
    for yid in sorted(youtube_ids):
        if yid in yid_to_vid:
            continue
        v = _next_v_id(used)
        yid_to_vid[yid] = v
        used.add(v)
        mapping_data["videos"].append({"v_id": v, "youtube_id": yid})
    return yid_to_vid


def _link_or_copy(src, dst, prefer_hardlink=True):
    """Create dst as a link/copy of src. Returns True if it actually wrote."""
    if dst.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    if prefer_hardlink:
        try:
            os.link(src, dst)
            return True
        except OSError:
            pass  # cross-device or unsupported; fall through to copy
    shutil.copy2(src, dst)
    return True


def _publish_tree(cache_root, dst_root, yid_to_vid, prefer_hardlink):
    """Walk cache_root/{yt_id}/* → dst_root/{v_id}/*. Returns # files added."""
    if not cache_root.exists():
        return 0
    n = 0
    for yt_dir in sorted(cache_root.iterdir()):
        if not yt_dir.is_dir() or yt_dir.name not in yid_to_vid:
            continue
        v_id = yid_to_vid[yt_dir.name]
        for src in sorted(yt_dir.iterdir()):
            if not src.is_file() or src.name.startswith("."):
                continue
            dst = dst_root / v_id / src.name
            if _link_or_copy(src, dst, prefer_hardlink=prefer_hardlink):
                n += 1
    return n


def _publish_csv(src_csv, dst_csv, yid_to_vid):
    """Read src_csv (YT IDs in `name`), write dst_csv with v-IDs."""
    if not src_csv.exists():
        return 0
    with open(src_csv, encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="|"))
    if not rows:
        return 0
    n = 0
    for r in rows:
        vid, clip = r["name"].split("/")
        if vid in yid_to_vid:
            r["name"] = f"{yid_to_vid[vid]}/{clip}"
        n += 1
    dst_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(dst_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="|")
        w.writeheader()
        w.writerows(rows)
    return n


def publish_dataset(
    cache_dir,
    dataset_dir,
    videos_json=None,
    splits=DEFAULT_SPLITS,
    trees=DEFAULT_TREES,
    docs=("README.md",),
    prefer_hardlink=True,
):
    """Publish the working tree under cache_dir to dataset_dir with v-IDs.

    Args:
        cache_dir: Path to data/cache/. Must contain features/, poses/...,
            per-split CSVs, and any docs (README.md) to ship.
        dataset_dir: Path to data/usl-suspilne/. Will be populated with v-ID
            directories, v-ID-keyed CSVs, and copied docs.
        videos_json: Path where the YT_ID ↔ v_ID mapping is persisted.
            Defaults to cache_dir / "videos.json".
        splits: Iterable of split names to publish (CSV files).
        trees: Iterable of subtree paths under cache/ and dataset/ to copy.
        docs: Iterable of files (relative to cache_dir) to copy verbatim into
            dataset_dir — e.g. README.md.
        prefer_hardlink: Use os.link when possible (fast, no disk overhead).
            Falls back to shutil.copy2 if hardlink fails (cross-device, etc.).

    Returns: dict with summary stats and the resolved mapping.
    """
    cache_dir = Path(cache_dir)
    dataset_dir = Path(dataset_dir)
    videos_json = Path(videos_json) if videos_json else cache_dir / "videos.json"

    # Discover all YT IDs in cache subtrees
    found = set()
    for tree in trees:
        t = cache_dir / tree
        if not t.exists():
            continue
        for d in t.iterdir():
            if d.is_dir():
                found.add(d.name)
    youtube_ids = {x for x in found if not _is_v_id(x)}

    mapping_data = _load_or_init_mapping(videos_json)
    yid_to_vid = _resolve_mapping(youtube_ids, mapping_data)
    _save_mapping(videos_json, mapping_data)

    summary = {
        "discovered_videos": len(found),
        "trees": [],
        "files_added": 0,
        "csvs_written": 0,
        "rows_published": 0,
        "docs_copied": 0,
    }

    for tree in trees:
        cache_root = cache_dir / tree
        dst_root = dataset_dir / tree
        n = _publish_tree(cache_root, dst_root, yid_to_vid, prefer_hardlink)
        summary["trees"].append({"path": tree, "files_added": n})
        summary["files_added"] += n

    for sp in splits:
        src = cache_dir / f"{sp}.csv"
        dst = dataset_dir / f"{sp}.csv"
        n = _publish_csv(src, dst, yid_to_vid)
        if n:
            summary["csvs_written"] += 1
            summary["rows_published"] += n

    # Docs (README.md, etc.) — overwrite each time so AUTO-STATS updates land.
    dataset_dir.mkdir(parents=True, exist_ok=True)
    for doc in docs:
        src = cache_dir / doc
        if not src.exists():
            continue
        dst = dataset_dir / doc
        if dst.exists():
            dst.unlink()
        shutil.copy2(src, dst)
        summary["docs_copied"] += 1

    summary["mapping"] = {e["youtube_id"]: e["v_id"] for e in mapping_data["videos"]}
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-dir", type=Path, default=Path("data/cache"))
    p.add_argument("--dataset-dir", type=Path, default=Path("data/usl-suspilne"))
    p.add_argument("--videos-json", type=Path, default=None,
                   help="Default: <cache-dir>/videos.json")
    p.add_argument("--splits", nargs="+", default=list(DEFAULT_SPLITS))
    p.add_argument("--no-hardlinks", action="store_true",
                   help="Always copy, even when hardlinks would work.")
    args = p.parse_args()
    summary = publish_dataset(
        cache_dir=args.cache_dir,
        dataset_dir=args.dataset_dir,
        videos_json=args.videos_json,
        splits=args.splits,
        prefer_hardlink=not args.no_hardlinks,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
