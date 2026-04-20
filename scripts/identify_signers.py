#!/usr/bin/env python3
"""
Identify which signer is in each clip by clustering face embeddings.

Two-phase pipeline:
 1. Sample frames across clips to get one embedding per video. Also detect
    multi-signer videos by clustering each video's frame embeddings
    against themselves.
 2. For flagged multi-signer videos, fall back to per-clip embeddings so we
    can assign a signer id to each clip individually.

Global agglomerative clustering over (video-means + per-clip-means)
produces stable signer ids shared across videos.

Outputs:
    signers.json       — {video_id: {"n_signers": int, "signers": {id: count},
                                     "clips": {clip_name: signer_id}}}
    signer_previews/signers.png — contact sheet grouped by signer id
    signer_previews/{video_id}.jpg — representative face per video
    signer_previews/{video_id}__clip_{name}.jpg — per-clip face for multi-signer videos

Usage:
    python3 scripts/identify_signers.py
    python3 scripts/identify_signers.py --threshold 0.45 --intra-threshold 0.35
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def _pick_clips(video_dir, n):
    """Pick up to n clip paths evenly spaced across the video's clip list."""
    clips = sorted(p for p in video_dir.glob("*.mp4") if not p.name.startswith("._"))
    if not clips:
        return []
    if len(clips) <= n:
        return clips
    idx = np.linspace(0, len(clips) - 1, n).round().astype(int)
    return [clips[i] for i in idx]


def _all_clips(video_dir):
    return sorted(p for p in video_dir.glob("*.mp4") if not p.name.startswith("._"))


def _sample_frames(clip_path, n):
    """Read up to n frames evenly spaced across the clip."""
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        cap.release()
        return []
    if n == 1:
        positions = [total // 2]
    else:
        positions = np.linspace(total * 0.1, total * 0.9, n).round().astype(int).tolist()
    frames = []
    for pos in positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(pos))
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
    cap.release()
    return frames


def _largest_face(app, frame):
    """Run face analysis and return the largest detection (or None)."""
    faces = app.get(frame)
    if not faces:
        return None
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))


def _crop_face(frame, bbox, size=128):
    """Return a square face crop resized to `size`."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    side = int(max(x2 - x1, y2 - y1) * 1.4)
    x1 = max(0, cx - side // 2)
    y1 = max(0, cy - side // 2)
    x2 = min(w, cx + side // 2)
    y2 = min(h, cy + side // 2)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (size, size))


def _collect_embeddings(app, clips, frames_per_clip, min_det_score):
    """Return lists (embeddings, scores, best_crop) across sampled frames."""
    embs = []
    scores = []
    best_crop = None
    best_score = 0.0
    n_candidates = 0
    for clip in clips:
        for frame in _sample_frames(clip, frames_per_clip):
            n_candidates += 1
            face = _largest_face(app, frame)
            if face is None or float(face.det_score) < min_det_score:
                continue
            embs.append(face.normed_embedding)
            scores.append(float(face.det_score))
            if face.det_score > best_score:
                best_score = float(face.det_score)
                best_crop = _crop_face(frame, face.bbox)
    return embs, scores, best_crop, n_candidates


def _weighted_mean(embs, scores):
    E = np.vstack(embs)
    w = np.asarray(scores, dtype=np.float32)
    mean = (E * w[:, None]).sum(axis=0) / w.sum()
    return mean / (np.linalg.norm(mean) + 1e-9)


def _smooth_tiny_clusters(labels, samples, min_cluster_size, window):
    """Reassign clips in tiny clusters to the majority label among temporal neighbors.

    Only operates on 'clip' samples (multi-signer videos). 'video' samples
    are preserved as-is. Works iteratively: after each pass, recompute which
    clusters are tiny and repeat until stable or 5 passes hit.
    """
    from collections import Counter

    labels = list(labels)
    # Build a per-video list of (index_in_samples, clip_name) sorted by clip_name
    per_video = {}
    for i, (kind, key, *_rest) in enumerate(samples):
        if kind != "clip":
            continue
        video_id, clip_name = key
        per_video.setdefault(video_id, []).append((clip_name, i))
    for v in per_video:
        per_video[v].sort(key=lambda t: t[0])

    for _ in range(5):
        counts = Counter(labels)
        tiny = {lab for lab, c in counts.items() if c < min_cluster_size}
        if not tiny:
            break
        changed = 0
        for video_id, seq in per_video.items():
            idx_list = [i for _, i in seq]
            for pos, (_, i) in enumerate(seq):
                if labels[i] not in tiny:
                    continue
                lo = max(0, pos - window)
                hi = min(len(seq), pos + window + 1)
                neighbor_labels = [labels[idx_list[p]] for p in range(lo, hi)
                                   if p != pos and labels[idx_list[p]] not in tiny]
                if not neighbor_labels:
                    continue
                new_lab = Counter(neighbor_labels).most_common(1)[0][0]
                if new_lab != labels[i]:
                    labels[i] = new_lab
                    changed += 1
        if changed == 0:
            break
    return labels


def _build_contact_sheet(crops, labels, keys, out_path, tile=128, cols=8):
    """Arrange face crops into a grid grouped by cluster label."""
    order = sorted(range(len(labels)), key=lambda i: (labels[i], i))
    n = len(order)
    rows = (n + cols - 1) // cols
    sheet = np.full((rows * tile, cols * tile, 3), 255, dtype=np.uint8)
    for i, idx in enumerate(order):
        crop = crops[idx]
        if crop is None:
            continue
        r, c = divmod(i, cols)
        y, x = r * tile, c * tile
        sheet[y:y + tile, x:x + tile] = crop
        cv2.putText(sheet, f"s{labels[idx]}", (x + 4, y + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imwrite(str(out_path), sheet)


def identify_signers(features_dir, output_path, preview_dir,
                     clips_per_video=8, frames_per_clip=3,
                     frames_per_clip_fallback=2,
                     min_det_score=0.6,
                     distance_threshold=0.4,
                     intra_video_threshold=0.4,
                     intra_min_cluster_size=3,
                     min_signer_cluster_size=5,
                     temporal_window=5):
    """Cluster signers via a video-level pass + per-clip fallback.

    Args:
        features_dir: Directory with cropped clips ({video_id}/*.mp4).
        output_path: Where to write the JSON mapping.
        preview_dir: Directory for preview images.
        clips_per_video: Clips sampled in the video-level pass.
        frames_per_clip: Frames per clip in the video-level pass.
        frames_per_clip_fallback: Frames per clip in the per-clip pass.
        min_det_score: Drop face detections below this score.
        distance_threshold: Global agglomerative cosine threshold.
        intra_video_threshold: Cosine threshold for detecting multi-signer videos.
        intra_min_cluster_size: Ignore intra-video sub-clusters smaller than this
            (suppresses false multi-signer flags from 1-2 bad frames).
        min_signer_cluster_size: Any global cluster with fewer than this many
            samples is treated as noise and its clips get reassigned to the
            majority label among their temporal neighbors.
        temporal_window: ± neighbor radius (in clip-index distance) for the
            temporal smoothing vote.

    Returns:
        dict with summary stats.
    """
    from insightface.app import FaceAnalysis
    from sklearn.cluster import AgglomerativeClustering

    features_dir = Path(features_dir)
    output_path = Path(output_path)
    preview_dir = Path(preview_dir)
    preview_dir.mkdir(parents=True, exist_ok=True)

    video_dirs = sorted(p for p in features_dir.iterdir() if p.is_dir())
    print(f"Found {len(video_dirs)} videos in {features_dir}\n")

    app = FaceAnalysis(name="buffalo_s", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))

    # ------------------------------------------------------------------
    # Phase 1: video-level pass
    # ------------------------------------------------------------------
    print("== Phase 1: video-level pass ==")
    video_frames = {}   # video_id -> (embs, scores)
    video_crop = {}     # video_id -> preview crop
    failed = []

    for video_dir in video_dirs:
        video_id = video_dir.name
        clips = _pick_clips(video_dir, clips_per_video)
        if not clips:
            print(f"  [SKIP] {video_id}: no clips")
            failed.append(video_id)
            continue
        embs, scores, crop, n_cands = _collect_embeddings(
            app, clips, frames_per_clip, min_det_score
        )
        if not embs:
            print(f"  [SKIP] {video_id}: 0/{n_cands} faces above {min_det_score}")
            failed.append(video_id)
            continue
        video_frames[video_id] = (embs, scores)
        video_crop[video_id] = crop
        if crop is not None:
            cv2.imwrite(str(preview_dir / f"{video_id}.jpg"), crop)
        print(f"  [OK]   {video_id}: {len(embs)}/{n_cands} frames")

    # ------------------------------------------------------------------
    # Phase 1b: multi-signer detection via intra-video clustering
    # ------------------------------------------------------------------
    print("\n== Phase 1b: intra-video clustering to flag multi-signer videos ==")
    multi_signer = []
    for video_id, (embs, _) in video_frames.items():
        if len(embs) < max(2, intra_min_cluster_size):
            continue
        sub = AgglomerativeClustering(
            n_clusters=None, metric="cosine", linkage="average",
            distance_threshold=intra_video_threshold,
        ).fit_predict(np.vstack(embs))
        sizes = np.bincount(sub)
        big = [i for i, s in enumerate(sizes) if s >= intra_min_cluster_size]
        if len(big) > 1:
            multi_signer.append(video_id)
            print(f"  [MULTI] {video_id}: sub-cluster sizes {sorted(sizes.tolist(), reverse=True)}")
    if not multi_signer:
        print("  (none flagged)")

    # ------------------------------------------------------------------
    # Phase 2: per-clip pass for flagged videos
    # ------------------------------------------------------------------
    print(f"\n== Phase 2: per-clip pass for {len(multi_signer)} flagged videos ==")
    clip_embs = {}  # (video_id, clip_name) -> (emb, crop)
    clip_missing = {}  # video_id -> list of clip_names without detections

    for video_id in multi_signer:
        video_dir = features_dir / video_id
        clips = _all_clips(video_dir)
        n_ok = 0
        n_missing = 0
        for clip in clips:
            embs, scores, crop, _ = _collect_embeddings(
                app, [clip], frames_per_clip_fallback, min_det_score
            )
            if not embs:
                clip_missing.setdefault(video_id, []).append(clip.name)
                n_missing += 1
                continue
            mean = _weighted_mean(embs, scores)
            clip_embs[(video_id, clip.name)] = (mean, crop)
            n_ok += 1
        print(f"  {video_id}: {n_ok}/{len(clips)} clips embedded "
              f"({n_missing} without detections)")

    # ------------------------------------------------------------------
    # Phase 3: global clustering over video-means + clip-means
    # ------------------------------------------------------------------
    print("\n== Phase 3: global clustering ==")
    samples = []  # list of (kind, key, emb, crop)
    for video_id, (embs, scores) in video_frames.items():
        if video_id in multi_signer:
            continue
        samples.append(("video", video_id, _weighted_mean(embs, scores), video_crop[video_id]))
    for (video_id, clip_name), (emb, crop) in clip_embs.items():
        samples.append(("clip", (video_id, clip_name), emb, crop))

    if len(samples) < 2:
        raise RuntimeError(f"Need >=2 samples to cluster, got {len(samples)}")

    X = np.vstack([s[2] for s in samples])
    labels = AgglomerativeClustering(
        n_clusters=None, metric="cosine", linkage="average",
        distance_threshold=distance_threshold,
    ).fit_predict(X)

    labels = [int(l) for l in labels]
    print(f"  samples={len(samples)} → raw n_clusters={len(set(labels))} "
          f"(threshold={distance_threshold})")

    # ------------------------------------------------------------------
    # Phase 3b: temporal smoothing — reassign tiny clusters by neighbor vote
    # ------------------------------------------------------------------
    labels = _smooth_tiny_clusters(
        labels, samples,
        min_cluster_size=min_signer_cluster_size,
        window=temporal_window,
    )

    # Renumber so signer ids are consecutive, stable by first appearance.
    remap = {}
    for lab in labels:
        if lab not in remap:
            remap[lab] = len(remap)
    labels = [remap[l] for l in labels]

    n_clusters = len(set(labels))
    print(f"  after temporal smoothing: n_clusters={n_clusters}")

    # ------------------------------------------------------------------
    # Build per-video output mapping
    # ------------------------------------------------------------------
    mapping = {}
    for video_id in sorted({v.name for v in video_dirs}):
        mapping[video_id] = {"n_signers": 0, "signers": {}, "clips": {}}

    for (kind, key, _, crop), lab in zip(samples, labels):
        if kind == "video":
            video_id = key
            for clip_path in _all_clips(features_dir / video_id):
                mapping[video_id]["clips"][clip_path.name] = lab
        else:
            video_id, clip_name = key
            mapping[video_id]["clips"][clip_name] = lab
            if crop is not None:
                cv2.imwrite(str(preview_dir / f"{video_id}__clip_{clip_name}.jpg"), crop)

    # Fill in clips without detections using the dominant signer id of their video
    for video_id, missing_clips in clip_missing.items():
        known = [v for v in mapping[video_id]["clips"].values() if v is not None]
        fallback = max(set(known), key=known.count) if known else None
        for clip_name in missing_clips:
            mapping[video_id]["clips"][clip_name] = fallback

    # Summarise signer distribution per video
    for video_id, info in mapping.items():
        counts = {}
        for lab in info["clips"].values():
            if lab is None:
                continue
            counts[lab] = counts.get(lab, 0) + 1
        info["signers"] = {str(k): v for k, v in sorted(counts.items())}
        info["n_signers"] = len(counts)

    # ------------------------------------------------------------------
    # Write outputs
    # ------------------------------------------------------------------
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(mapping, f, indent=2, sort_keys=True)

    # Contact sheet: one tile per *sample* (video-mean or per-clip)
    sheet_crops = [s[3] for s in samples]
    sheet_keys = [s[1] for s in samples]
    _build_contact_sheet(sheet_crops, labels, sheet_keys, preview_dir / "signers.png")

    print(f"\nDone.")
    print(f"  multi-signer videos: {multi_signer or 'none'}")
    print(f"  mapping: {output_path}")
    print(f"  preview: {preview_dir / 'signers.png'}")
    if failed:
        print(f"  [WARN] {len(failed)} videos without any detections: {failed}")

    return {
        "videos": len(video_dirs),
        "multi_signer": multi_signer,
        "n_clusters": n_clusters,
        "failed": failed,
    }


def main():
    ROOT = Path(__file__).resolve().parent.parent
    default_src = ROOT / "data/usl-suspilne/features"
    default_out = ROOT / "data/cache/signers.json"
    default_preview = ROOT / "data/cache/signer_previews"

    parser = argparse.ArgumentParser(description="Cluster signers via face embeddings")
    parser.add_argument("--src", type=Path, default=default_src)
    parser.add_argument("--out", type=Path, default=default_out)
    parser.add_argument("--preview", type=Path, default=default_preview)
    parser.add_argument("--clips", type=int, default=8,
                        help="Clips sampled per video in phase 1 (default: 8)")
    parser.add_argument("--frames-per-clip", type=int, default=3,
                        help="Frames per clip in phase 1 (default: 3)")
    parser.add_argument("--frames-per-clip-fallback", type=int, default=2,
                        help="Frames per clip in phase 2 (default: 2)")
    parser.add_argument("--min-score", type=float, default=0.6,
                        help="Drop detections below this score (default: 0.6)")
    parser.add_argument("--threshold", type=float, default=0.4,
                        help="Global cosine distance threshold (default: 0.4)")
    parser.add_argument("--intra-threshold", type=float, default=0.4,
                        help="Cosine threshold for intra-video sub-clustering (default: 0.4)")
    parser.add_argument("--intra-min-cluster-size", type=int, default=3,
                        help="Min sub-cluster size to flag multi-signer (default: 3)")
    parser.add_argument("--min-cluster-size", type=int, default=5,
                        help="Clusters smaller than this are treated as noise "
                             "and reassigned by temporal neighbors (default: 5)")
    parser.add_argument("--temporal-window", type=int, default=5,
                        help="± clip-index radius for temporal smoothing vote (default: 5)")
    args = parser.parse_args()

    identify_signers(
        features_dir=args.src,
        output_path=args.out,
        preview_dir=args.preview,
        clips_per_video=args.clips,
        frames_per_clip=args.frames_per_clip,
        frames_per_clip_fallback=args.frames_per_clip_fallback,
        min_det_score=args.min_score,
        distance_threshold=args.threshold,
        intra_video_threshold=args.intra_threshold,
        intra_min_cluster_size=args.intra_min_cluster_size,
        min_signer_cluster_size=args.min_cluster_size,
        temporal_window=args.temporal_window,
    )


if __name__ == "__main__":
    main()