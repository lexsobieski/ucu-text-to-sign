#!/usr/bin/env python3
"""
Prepare data files for Progressive Transformers training.

Reads:
  - data/usl-suspilne/annotations.csv  (pipe-delimited, with text column)
  - data/usl-suspilne/poses/3d/{videoId}/{clipIdx}.npy  (3D lifted poses, shape (T, 150))

Writes (for each split: train, dev, test):
  - experiments/progressive_transformers/model/Data/usl/{split}.text/.skels/.files

Also writes:
  - experiments/progressive_transformers/model/Configs/usl_src_vocab.txt

Usage:
    python experiments/progressive_transformers/prepare_data.py
    python experiments/progressive_transformers/prepare_data.py --bpe 2000

Format:
  .text  — one line per sequence, Ukrainian text (source for Text->Pose)
  .skels — one line per sequence, all frames concatenated:
           each frame is 150 joint values + 1 counter value (= 151 per frame)
  .files — one line per sequence, the sequence name (videoId/clipIdx)

Counter: frame i of N total frames => i / (N - 1), appended as 151st value.
"""

import argparse
import csv
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parent.parent
SP_MODEL_PATH = EXPERIMENT_DIR / "model/Configs/usl_sp.model"

# Train/dev/test split by video ID.
# ~80% train, ~10% dev, ~10% test (by clip count).
# Videos with aligned data (after min-duration filter):
#   d37lwXaSjs4: ~498, IOflFDS2biE: ~507, 6O0ZiSgKJNc: ~245,
#   S0o1oJ6G5qw: ~114, w_LdfLKP_0o: ~71, Nyykyn4FpNo: 18
# Total: ~1453.  Dev ~114, Test ~89, Train ~1250.
DEV_VIDEOS = {"S0o1oJ6G5qw"}                          # ~114
TEST_VIDEOS = {"w_LdfLKP_0o", "Nyykyn4FpNo"}          # ~71 + 18 = ~89
# Train: d37lwXaSjs4 + IOflFDS2biE + 6O0ZiSgKJNc = ~498 + ~507 + ~245 = ~1250


def normalize_text(text: str) -> str:
    """Lowercase and strip punctuation from token edges."""
    tokens = text.lower().split()
    cleaned = []
    for t in tokens:
        t = re.sub(r'^[^\w]+|[^\w]+$', '', t, flags=re.UNICODE)
        if t:
            cleaned.append(t)
    return " ".join(cleaned)


def build_bpe_tokenizer(train_texts, vocab_size):
    """Train a SentencePiece BPE model on training texts, return the processor."""
    import sentencepiece as spm

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        for t in train_texts:
            f.write(t + "\n")
        train_file = f.name

    SP_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    prefix = str(SP_MODEL_PATH).removesuffix(".model")

    spm.SentencePieceTrainer.train(
        input=train_file,
        model_prefix=prefix,
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=1.0,
        pad_id=3,
    )
    Path(train_file).unlink()

    sp = spm.SentencePieceProcessor(model_file=str(SP_MODEL_PATH))
    print(f"Trained BPE model: vocab_size={sp.get_piece_size()}, saved to {SP_MODEL_PATH}")
    return sp


def load_annotations(annotation_path):
    """Load annotations from usl-suspilne.csv."""
    rows = []
    with open(annotation_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="|")
        for row in reader:
            rows.append(row)
    return rows


def get_split(video_id):
    if video_id in DEV_VIDEOS:
        return "dev"
    elif video_id in TEST_VIDEOS:
        return "test"
    else:
        return "train"


def format_skels_line(poses_3d: np.ndarray) -> str:
    """Convert a (T, 150) 3D pose array to a single-line skeleton string.

    Each frame becomes 151 values: 150 joint coordinates + 1 counter.
    Counter = i / (N-1) for frame i of N total frames.
    """
    T = poses_3d.shape[0]
    parts = []
    for i in range(T):
        frame_vals = poses_3d[i].tolist()
        # Append counter
        counter = i / max(T - 1, 1)
        frame_vals.append(counter)
        parts.extend(frame_vals)
    return " ".join(f"{v:.6f}" for v in parts)


def prepare_all(annotations, poses_dir, out_dir, vocab_path, norm_stats_path, bpe=None):
    """Prepare Progressive Transformers training data.

    Args:
        annotations: Path to annotations.csv.
        poses_dir: Directory with 3D pose npy files.
        out_dir: Output directory for .text/.skels/.files.
        vocab_path: Output path for vocabulary file.
        norm_stats_path: Output path for normalization stats.
        bpe: If set, use BPE subword tokenization with this vocab size.
    """
    annotations, poses_dir = Path(annotations), Path(poses_dir)
    out_dir, vocab_path = Path(out_dir), Path(vocab_path)
    norm_stats_path = Path(norm_stats_path)

    rows = load_annotations(annotations)
    print(f"Loaded {len(rows)} annotations from {annotations}")

    # Pass 1: load raw poses and texts, grouped by split
    splits = {"train": [], "dev": [], "test": []}
    skipped = 0

    for row in rows:
        name = row["name"]  # e.g. "d37lwXaSjs4/0000"
        video_id = name.split("/")[0]
        clip_idx = name.split("/")[1]

        pose_path = poses_dir / video_id / f"{clip_idx}.npy"
        if not pose_path.exists():
            skipped += 1
            continue

        poses_3d = np.load(pose_path)
        if poses_3d.shape[0] == 0 or poses_3d.shape[1] != 150:
            print(f"  WARNING: {name} has unexpected shape {poses_3d.shape}, "
                  f"skipping", file=sys.stderr)
            skipped += 1
            continue

        text = normalize_text(row["text"])
        split = get_split(video_id)
        splits[split].append({
            "name": name,
            "text": text,
            "poses": poses_3d,
        })

    if skipped > 0:
        print(f"Skipped {skipped} clips (missing 3D poses)")

    for split_name, items in splits.items():
        print(f"  {split_name}: {len(items)} sequences")

    # Pass 2: compute normalization stats from training set
    train_frames = np.concatenate([item["poses"] for item in splits["train"]], axis=0)
    mean = train_frames.mean(axis=0)  # (150,)
    std = train_frames.std(axis=0)    # (150,)
    std[std < 1e-6] = 1.0  # avoid division by zero for constant joints

    norm_stats_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(norm_stats_path, mean=mean, std=std)
    print(f"\nPose normalization stats (from {train_frames.shape[0]} train frames):")
    print(f"  mean range: [{mean.min():.4f}, {mean.max():.4f}]")
    print(f"  std range:  [{std.min():.4f}, {std.max():.4f}]")
    print(f"  Saved to {norm_stats_path}")

    # Pass 3: normalize poses and format skels
    for split_items in splits.values():
        for item in split_items:
            item["poses"] = (item["poses"] - mean) / std
            item["skels"] = format_skels_line(item["poses"])
            del item["poses"]

    # BPE tokenization (optional)
    sp = None
    if bpe:
        train_texts = [item["text"] for item in splits["train"]]
        sp = build_bpe_tokenizer(train_texts, bpe)
        for split_items in splits.values():
            for item in split_items:
                pieces = sp.encode(item["text"], out_type=str)
                item["text"] = " ".join(pieces)

    # Write output files
    out_dir.mkdir(parents=True, exist_ok=True)

    for split_name, items in splits.items():
        prefix = out_dir / split_name

        with open(str(prefix) + ".text", "w", encoding="utf-8") as ft, \
             open(str(prefix) + ".skels", "w", encoding="utf-8") as fs, \
             open(str(prefix) + ".files", "w", encoding="utf-8") as ff:

            for item in items:
                ft.write(item["text"] + "\n")
                fs.write(item["skels"] + "\n")
                ff.write(item["name"] + "\n")

    # Build vocabulary
    vocab_path.parent.mkdir(parents=True, exist_ok=True)

    if sp:
        # Write SentencePiece vocab (excluding special tokens)
        with open(vocab_path, "w", encoding="utf-8") as f:
            for i in range(4, sp.get_piece_size()):  # skip <unk>, <s>, </s>, <pad>
                f.write(sp.id_to_piece(i) + "\n")
        vocab_size = sp.get_piece_size() - 4
    else:
        # Word-level vocab from training data
        train_tokens = []
        for item in splits["train"]:
            train_tokens.extend(item["text"].split())

        counter = Counter(train_tokens)
        vocab_tokens = sorted(counter.keys(), key=lambda t: (-counter[t], t))

        with open(vocab_path, "w", encoding="utf-8") as f:
            for token in vocab_tokens:
                f.write(token + "\n")
        vocab_size = len(vocab_tokens)

    print(f"\nWrote data files to {out_dir}")
    print(f"Wrote vocabulary ({vocab_size} tokens) to {vocab_path}")

    # Verification
    for split_name in ["train", "dev", "test"]:
        prefix = out_dir / split_name
        for ext in [".text", ".skels", ".files"]:
            path = str(prefix) + ext
            with open(path) as f:
                n = sum(1 for _ in f)
            print(f"  {split_name}{ext}: {n} lines")


def main():
    default_annotations = PROJECT_ROOT / "data/usl-suspilne/annotations.csv"
    default_poses = PROJECT_ROOT / "data/usl-suspilne/poses/3d"
    default_out = EXPERIMENT_DIR / "model/Data/usl"
    default_vocab = EXPERIMENT_DIR / "model/Configs/usl_src_vocab.txt"
    default_norm = EXPERIMENT_DIR / "model/Configs/usl_pose_norm.npz"

    parser = argparse.ArgumentParser(description="Prepare Progressive Transformers data")
    parser.add_argument("--annotations", type=Path, default=default_annotations,
                        help=f"Path to annotations.csv (default: {default_annotations})")
    parser.add_argument("--poses", type=Path, default=default_poses,
                        help=f"3D poses directory (default: {default_poses})")
    parser.add_argument("--out", type=Path, default=default_out,
                        help=f"Output directory (default: {default_out})")
    parser.add_argument("--bpe", type=int, default=None,
                        help="Use BPE subword tokenization with this vocab size (e.g. 2000)")
    args = parser.parse_args()

    prepare_all(annotations=args.annotations, poses_dir=args.poses,
                out_dir=args.out, vocab_path=default_vocab,
                norm_stats_path=default_norm, bpe=args.bpe)


if __name__ == "__main__":
    main()
