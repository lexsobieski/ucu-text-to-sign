#!/usr/bin/env python3
"""
Crop all video clips to the signer region (bottom-right corner by default).

Usage:
    python3 scripts/crop_signer.py
    python3 scripts/crop_signer.py --src data/cache/uncropped --dst data/usl-suspilne/features
    python3 scripts/crop_signer.py --width 510 --height 510 --output-size 256
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


# Per-video crop overrides: {video_id: (crop_w, crop_h)}
VIDEO_OVERRIDES = {}


def get_resolution(path: Path):
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", str(path)],
        capture_output=True, text=True,
    )
    streams = json.loads(result.stdout)["streams"]
    for s in streams:
        if s.get("codec_type") == "video":
            return int(s["width"]), int(s["height"])
    return None, None


def crop_all(src, dst, width=510, height=510, output_size=None, force=False):
    """Crop video clips to the signer region (bottom-right corner).

    Args:
        src: Directory with uncropped clips ({videoId}/{clip}.mp4).
        dst: Output directory for cropped clips.
        width: Crop width in pixels.
        height: Crop height in pixels.
        output_size: If set, rescale cropped region to this square size.
        force: Re-crop even if output exists.

    Returns:
        dict with 'total' and 'failed' counts.
    """
    src, dst = Path(src), Path(dst)
    clips = sorted(p for p in src.glob("*/*.mp4") if not p.name.startswith("._"))
    print(f"Found {len(clips)} clips in {src}")

    total = 0
    failed = 0

    for clip in clips:
        video_id = clip.parent.name

        # Determine crop size: per-video override or function defaults
        crop_w, crop_h = VIDEO_OVERRIDES.get(video_id, (width, height))

        out_dir = dst / video_id
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / clip.name

        if out_path.exists() and not force:
            total += 1
            continue

        # Get source resolution to compute bottom-right offset
        w, h = get_resolution(clip)
        if w is None:
            print(f"  [ERROR] Can't read resolution: {clip}", file=sys.stderr)
            failed += 1
            continue

        x = max(0, w - crop_w)
        y = max(0, h - crop_h)

        # Build filter chain: crop, then optionally scale
        vf = f"crop={crop_w}:{crop_h}:{x}:{y}"
        if output_size:
            vf += f",scale={output_size}:{output_size}"

        cmd = [
            "ffmpeg", "-y",
            "-i", str(clip),
            "-vf", vf,
            "-c:v", "libx264",
            "-preset", "fast",
            "-an",
            "-loglevel", "error",
            str(out_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  [ERROR] {clip.name}: {result.stderr[:200]}", file=sys.stderr)
            failed += 1
        else:
            total += 1

    print(f"\nDone. {total} cropped, {failed} failed.")
    if VIDEO_OVERRIDES:
        print(f"Per-video overrides: {', '.join(f'{k} ({v[0]}x{v[1]})' for k, v in VIDEO_OVERRIDES.items())}")

    return {"total": total, "failed": failed}


def main():
    ROOT = Path(__file__).resolve().parent.parent
    default_src = ROOT / "data/cache/uncropped"
    default_dst = ROOT / "data/usl-suspilne/features"

    parser = argparse.ArgumentParser(description="Crop video clips to signer region")
    parser.add_argument("--src", type=Path, default=default_src,
                        help=f"Source directory (default: {default_src})")
    parser.add_argument("--dst", type=Path, default=default_dst,
                        help=f"Destination directory (default: {default_dst})")
    parser.add_argument("--width", type=int, default=510, help="Default crop width (default: 510)")
    parser.add_argument("--height", type=int, default=510, help="Default crop height (default: 510)")
    parser.add_argument("--output-size", type=int, default=None,
                        help="If set, rescale cropped region to this square size (e.g. 256)")
    parser.add_argument("--force", action="store_true", help="Re-crop even if output exists")
    args = parser.parse_args()

    crop_all(src=args.src, dst=args.dst, width=args.width, height=args.height,
             output_size=args.output_size, force=args.force)


if __name__ == "__main__":
    main()