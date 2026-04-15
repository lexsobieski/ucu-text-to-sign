#!/usr/bin/env python3
"""
Extract full MediaPipe Holistic poses from cropped sign language videos.

Joint layout (75 joints total):
  0-32:  Body (33 MediaPipe Pose landmarks)
  33-53: Left hand (21 MediaPipe hand landmarks)
  54-74: Right hand (21 MediaPipe hand landmarks)

Output per frame: 75 joints × (x, y, visibility) = 225 values.
Coordinates normalized to [0, 1].

Usage:
    python3 scripts/extract_poses.py --src data/usl-suspilne/features --dst data/usl-suspilne/poses/mediapipe_holistic
"""

import sys
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

_HAS_SOLUTIONS = hasattr(mp, "solutions") and hasattr(getattr(mp, "solutions", None), "holistic")
_HAS_TASKS = hasattr(mp, "tasks")

NUM_BODY = 33
NUM_HAND = 21
NUM_JOINTS = NUM_BODY + 2 * NUM_HAND  # 75
VALS_PER_FRAME = NUM_JOINTS * 3  # 225

# ---------------------------------------------------------------------------
# Model download for mp.tasks API
# ---------------------------------------------------------------------------
_POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_heavy/float16/latest/"
    "pose_landmarker_heavy.task"
)
_HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/"
    "hand_landmarker.task"
)


def _ensure_task_models():
    """Download mp.tasks model files if not already present."""
    model_dir = Path(__file__).resolve().parent.parent / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    pose_path = model_dir / "pose_landmarker_heavy.task"
    hand_path = model_dir / "hand_landmarker.task"

    for url, path in [(_POSE_MODEL_URL, pose_path), (_HAND_MODEL_URL, hand_path)]:
        if not path.exists():
            print(f"Downloading {path.name} ...")
            urllib.request.urlretrieve(url, path)
            print(f"  saved to {path}")

    return str(pose_path), str(hand_path)


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def process_video(video_path):
    """Extract 75-joint Holistic poses from all frames of a video.

    Returns:
        numpy array of shape (T, 225) where 225 = 75 joints × (x, y, visibility).
        Returns None if the video cannot be opened.
    """
    if _HAS_TASKS:
        return _process_video_tasks(video_path)
    elif _HAS_SOLUTIONS:
        return _process_video_legacy(video_path)
    else:
        raise RuntimeError(
            "MediaPipe has neither 'tasks' nor 'solutions' API. "
            "Install mediapipe >= 0.10.14: pip install -U mediapipe"
        )


def _extract_body_landmarks(landmarks):
    """Extract 33 body joints from pose landmarks."""
    kps = []
    for lm in landmarks:
        kps.extend([lm.x, lm.y, lm.visibility])
    return kps


def _extract_hand_landmarks(landmarks):
    """Extract 21 hand joints from hand landmarks."""
    kps = []
    for lm in landmarks:
        kps.extend([lm.x, lm.y, 1.0])
    return kps


def _process_video_tasks(video_path):
    """Extraction using mp.tasks API (PoseLandmarker + HandLandmarker)."""
    pose_model_path, hand_model_path = _ensure_task_models()

    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode

    delegate = (BaseOptions.Delegate.GPU
                if "google.colab" in sys.modules
                else BaseOptions.Delegate.CPU)

    pose_options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=pose_model_path, delegate=delegate),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
    )
    hand_options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=hand_model_path, delegate=delegate),
        running_mode=RunningMode.VIDEO,
        num_hands=2,
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [ERROR] Cannot open {video_path}", file=sys.stderr)
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    with PoseLandmarker.create_from_options(pose_options) as pose_lm, \
         HandLandmarker.create_from_options(hand_options) as hand_lm:

        all_frames = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(frame_idx * 1000 / fps)

            pose_result = pose_lm.detect_for_video(mp_image, timestamp_ms)
            hand_result = hand_lm.detect_for_video(mp_image, timestamp_ms)

            # Body (33 joints)
            if pose_result.pose_landmarks:
                body_kps = _extract_body_landmarks(pose_result.pose_landmarks[0])
            else:
                body_kps = [0.0] * (NUM_BODY * 3)

            # Hands (21 joints each)
            left_kps = [0.0] * (NUM_HAND * 3)
            right_kps = [0.0] * (NUM_HAND * 3)
            if hand_result.hand_landmarks:
                for h_idx, hand_landmarks in enumerate(hand_result.hand_landmarks):
                    handedness = hand_result.handedness[h_idx][0].category_name
                    kps = _extract_hand_landmarks(hand_landmarks)
                    if handedness == "Left":
                        left_kps = kps
                    else:
                        right_kps = kps

            all_frames.append(body_kps + left_kps + right_kps)
            frame_idx += 1

    cap.release()

    if not all_frames:
        return np.zeros((0, VALS_PER_FRAME), dtype=np.float32)
    return np.array(all_frames, dtype=np.float32)


def _process_video_legacy(video_path):
    """Extraction using legacy mp.solutions.holistic API."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [ERROR] Cannot open {video_path}", file=sys.stderr)
        return None

    holistic = mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    all_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)

        # Body (33 joints)
        if results.pose_landmarks:
            body_kps = _extract_body_landmarks(results.pose_landmarks.landmark)
        else:
            body_kps = [0.0] * (NUM_BODY * 3)

        # Left hand
        if results.left_hand_landmarks:
            left_kps = _extract_hand_landmarks(results.left_hand_landmarks.landmark)
        else:
            left_kps = [0.0] * (NUM_HAND * 3)

        # Right hand
        if results.right_hand_landmarks:
            right_kps = _extract_hand_landmarks(results.right_hand_landmarks.landmark)
        else:
            right_kps = [0.0] * (NUM_HAND * 3)

        all_frames.append(body_kps + left_kps + right_kps)

    cap.release()
    holistic.close()

    if not all_frames:
        return np.zeros((0, VALS_PER_FRAME), dtype=np.float32)
    return np.array(all_frames, dtype=np.float32)


# ---------------------------------------------------------------------------
# Batch extraction
# ---------------------------------------------------------------------------

class _TimeoutError(Exception):
    pass


def _sigalrm_handler(signum, frame):
    raise _TimeoutError()


def extract_all(src, dst):
    """Extract full Holistic poses for all video clips.

    Args:
        src: Directory with video clips ({videoId}/{clip}.mp4).
        dst: Output directory for pose npy files.

    Returns:
        dict with 'total' and 'failed' counts.
    """
    import signal

    src, dst = Path(src), Path(dst)
    clips = sorted(p for p in src.glob("*/*.mp4") if not p.name.startswith("._"))
    print(f"Found {len(clips)} clips in {src}\n")

    dst.mkdir(parents=True, exist_ok=True)
    timeout_sec = 120
    use_alarm = hasattr(signal, "SIGALRM")

    failed = []
    for i, clip in enumerate(clips):
        video_id = clip.parent.name
        clip_name = clip.stem
        out_path = dst / video_id / f"{clip_name}.npy"

        if out_path.exists():
            arr = np.load(out_path)
            print(f"  [{i+1}/{len(clips)}] {video_id}/{clip_name}: "
                  f"{arr.shape[0]} frames (cached)")
            continue

        out_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            if use_alarm:
                old_handler = signal.signal(signal.SIGALRM, _sigalrm_handler)
                signal.alarm(timeout_sec)
            arr = process_video(clip)
            if use_alarm:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
        except _TimeoutError:
            failed.append(f"{video_id}/{clip_name}")
            print(f"  [{i+1}/{len(clips)}] {video_id}/{clip_name}: TIMEOUT ({timeout_sec}s)")
            continue
        except Exception as e:
            failed.append(f"{video_id}/{clip_name}")
            print(f"  [{i+1}/{len(clips)}] {video_id}/{clip_name}: ERROR ({e})")
            continue

        if arr is not None and arr.shape[0] > 0:
            np.save(out_path, arr)
            print(f"  [{i+1}/{len(clips)}] {video_id}/{clip_name}: "
                  f"{arr.shape[0]} frames")
        else:
            failed.append(f"{video_id}/{clip_name}")
            print(f"  [{i+1}/{len(clips)}] {video_id}/{clip_name}: FAILED")

    total = len(list(dst.glob("*/*.npy")))
    print(f"\nDone. {total} poses extracted, {len(failed)} failed.")
    if failed:
        print(f"Failed clips: {failed[:20]}{'...' if len(failed) > 20 else ''}")
    return {"total": total, "failed": len(failed)}


if __name__ == "__main__":
    import argparse

    _ROOT = Path(__file__).resolve().parent.parent
    default_src = _ROOT / "data/usl-suspilne/features"
    default_dst = _ROOT / "data/usl-suspilne/poses/mediapipe_holistic"

    parser = argparse.ArgumentParser(description="Extract MediaPipe Holistic poses from videos")
    parser.add_argument("--src", type=Path, default=default_src,
                        help=f"Source video directory (default: {default_src})")
    parser.add_argument("--dst", type=Path, default=default_dst,
                        help=f"Output directory (default: {default_dst})")
    args = parser.parse_args()

    extract_all(src=args.src, dst=args.dst)
