#!/usr/bin/env python3
"""
Extract skeletal poses from cropped sign language videos.

Two modes:
  --mode openpose : Per-frame JSON (BODY_25 + hands + face) — requires legacy mp.solutions API
  --mode 50joint  : Per-clip numpy (50 joints for 3DposeEstimator) — uses mp.tasks API

Usage:
    pip install mediapipe opencv-python
    python3 scripts/extract_poses.py --mode 50joint

Reads:  data/usl-suspilne/features/{videoId}/{clip}.mp4
Writes: data/usl-suspilne/poses/2d/{videoId}/{clip}.npy   (50joint mode)
    or: data/poses_openpose/{videoId}/{clip}/{frame}_keypoints.json  (openpose mode)
"""

import json
import sys
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

# Detect which MediaPipe API is available
_HAS_SOLUTIONS = hasattr(mp, "solutions") and hasattr(getattr(mp, "solutions", None), "holistic")
_HAS_TASKS = hasattr(mp, "tasks")

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
# 50-joint extraction — mp.tasks API  (mediapipe >= 0.10.18)
# ---------------------------------------------------------------------------
# Joint layout (50 joints total):
#   0-7:   Body (Nose, Neck, RShoulder, RElbow, RWrist, LShoulder, LElbow, LWrist)
#   8-28:  Left hand (21 MediaPipe hand landmarks)
#   29-49: Right hand (21 MediaPipe hand landmarks)
# Output per frame: 50 joints x (x, y, confidence) = 150 values
# Coordinates normalised to [0, 1].

# MediaPipe Pose landmark indices for the 8 upper-body joints we need.
# Index 1 (Neck) is computed as midpoint of landmarks 11 & 12.
_BODY8_POSE_INDICES = [0, None, 12, 14, 16, 11, 13, 15]


def _extract_body8_from_landmarks(landmarks):
    """Extract 8 upper-body joints from a list of NormalizedLandmark."""
    kps = []
    for i, mp_idx in enumerate(_BODY8_POSE_INDICES):
        if i == 1:  # Neck = midpoint(LShoulder=11, RShoulder=12)
            x = (landmarks[11].x + landmarks[12].x) / 2
            y = (landmarks[11].y + landmarks[12].y) / 2
            c = (landmarks[11].visibility + landmarks[12].visibility) / 2
        else:
            x = landmarks[mp_idx].x
            y = landmarks[mp_idx].y
            c = landmarks[mp_idx].visibility
        kps.extend([x, y, c])
    return kps


def _extract_hand_from_landmarks(landmarks):
    """Extract 21 hand joints from a list of NormalizedLandmark."""
    kps = []
    for lm in landmarks:
        kps.extend([lm.x, lm.y, 1.0])
    return kps


def process_video_50joint(video_path: Path) -> np.ndarray:
    """Extract 50-joint poses from all frames of a video.

    Uses mp.tasks API (PoseLandmarker + HandLandmarker).
    Falls back to mp.solutions.holistic if available.

    Returns:
        numpy array of shape (T, 150) where T = number of frames,
        150 = 50 joints * (x, y, confidence).
        Returns None if the video cannot be opened.
    """
    if _HAS_TASKS:
        return _process_video_50joint_tasks(video_path)
    elif _HAS_SOLUTIONS:
        return _process_video_50joint_legacy(video_path)
    else:
        raise RuntimeError(
            "MediaPipe has neither 'tasks' nor 'solutions' API. "
            "Install mediapipe >= 0.10.14: pip install -U mediapipe"
        )


def _process_video_50joint_tasks(video_path: Path) -> np.ndarray:
    """50-joint extraction using mp.tasks API (PoseLandmarker + HandLandmarker)."""
    pose_model_path, hand_model_path = _ensure_task_models()

    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode

    # Use GPU delegate on Colab, CPU elsewhere
    delegate = (BaseOptions.Delegate.GPU
                if "google.colab" in sys.modules
                else BaseOptions.Delegate.CPU)

    pose_options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=pose_model_path,
                                 delegate=delegate),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
    )
    hand_options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=hand_model_path,
                                 delegate=delegate),
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

            # Body (8 joints)
            if pose_result.pose_landmarks:
                body_kps = _extract_body8_from_landmarks(
                    pose_result.pose_landmarks[0])
            else:
                body_kps = [0.0] * 24

            # Hands (21 joints each)
            left_kps = [0.0] * 63
            right_kps = [0.0] * 63
            if hand_result.hand_landmarks:
                for h_idx, hand_landmarks in enumerate(hand_result.hand_landmarks):
                    handedness = hand_result.handedness[h_idx][0].category_name
                    kps = _extract_hand_from_landmarks(hand_landmarks)
                    if handedness == "Left":
                        left_kps = kps
                    else:
                        right_kps = kps

            all_frames.append(body_kps + left_kps + right_kps)
            frame_idx += 1

    cap.release()

    if not all_frames:
        return np.zeros((0, 150), dtype=np.float32)
    return np.array(all_frames, dtype=np.float32)


# ---------------------------------------------------------------------------
# 50-joint extraction — legacy mp.solutions API  (mediapipe < 0.10.18)
# ---------------------------------------------------------------------------

def _process_video_50joint_legacy(video_path: Path) -> np.ndarray:
    """50-joint extraction using legacy mp.solutions.holistic API."""
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

        # Body (8 joints)
        if results.pose_landmarks:
            body_kps = _extract_body8_from_landmarks(
                results.pose_landmarks.landmark)
        else:
            body_kps = [0.0] * 24

        # Left hand (21 joints)
        if results.left_hand_landmarks:
            left_kps = _extract_hand_from_landmarks(
                results.left_hand_landmarks.landmark)
        else:
            left_kps = [0.0] * 63

        # Right hand (21 joints)
        if results.right_hand_landmarks:
            right_kps = _extract_hand_from_landmarks(
                results.right_hand_landmarks.landmark)
        else:
            right_kps = [0.0] * 63

        all_frames.append(body_kps + left_kps + right_kps)

    cap.release()
    holistic.close()

    if not all_frames:
        return np.zeros((0, 150), dtype=np.float32)
    return np.array(all_frames, dtype=np.float32)


# ---------------------------------------------------------------------------
# OpenPose-format extraction (legacy mp.solutions only)
# ---------------------------------------------------------------------------

# MediaPipe body landmark indices → OpenPose BODY_25 format.
MP_TO_BODY25 = [
    0, None, 12, 14, 16, 11, 13, 15, None,
    24, 26, 28, 23, 25, 27, 5, 2, 8, 7,
    31, 31, 29, 32, 32, 30,
]

MP_FACE_TO_OPENPOSE_70 = [
    234, 93, 132, 58, 172, 136, 150, 176, 152, 400, 379, 365, 397, 288, 361, 323, 454,
    70, 63, 105, 66, 107,
    336, 296, 334, 293, 300,
    168, 6, 197, 195, 5, 48, 115, 220, 45,
    33, 160, 158, 133, 153, 144,
    362, 385, 387, 263, 373, 380,
    61, 40, 37, 0, 267, 270, 291, 321, 314, 17, 84, 91,
    78, 82, 13, 312, 308, 317, 14, 87,
    468, 473,
]


def midpoint(lm, i, j, w, h):
    x = (lm[i].x + lm[j].x) / 2 * w
    y = (lm[i].y + lm[j].y) / 2 * h
    c = (lm[i].visibility + lm[j].visibility) / 2
    return x, y, c


def extract_body25(pose_landmarks, w, h):
    if pose_landmarks is None:
        return [0.0] * 75
    lm = pose_landmarks.landmark
    kps = []
    for i, mp_idx in enumerate(MP_TO_BODY25):
        if i == 1:
            x, y, c = midpoint(lm, 11, 12, w, h)
        elif i == 8:
            x, y, c = midpoint(lm, 23, 24, w, h)
        else:
            x = lm[mp_idx].x * w
            y = lm[mp_idx].y * h
            c = lm[mp_idx].visibility
        kps.extend([x, y, c])
    return kps


def extract_hand(hand_landmarks, w, h):
    if hand_landmarks is None:
        return [0.0] * 63
    kps = []
    for lm in hand_landmarks.landmark:
        kps.extend([lm.x * w, lm.y * h, 1.0])
    return kps


def extract_face70(face_landmarks, w, h):
    if face_landmarks is None:
        return [0.0] * 210
    lm = face_landmarks.landmark
    kps = []
    for mp_idx in MP_FACE_TO_OPENPOSE_70:
        if mp_idx < len(lm):
            kps.extend([lm[mp_idx].x * w, lm[mp_idx].y * h, 1.0])
        else:
            kps.extend([0.0, 0.0, 0.0])
    return kps


def process_video(video_path: Path, out_dir: Path):
    """Extract OpenPose-format poses. Requires legacy mp.solutions API."""
    if not _HAS_SOLUTIONS:
        raise RuntimeError(
            "OpenPose-format extraction requires mp.solutions (mediapipe < 0.10.18). "
            "Use --mode 50joint instead, or: pip install mediapipe==0.10.14"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [ERROR] Cannot open {video_path}", file=sys.stderr)
        return 0

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    holistic = mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)

        keypoints = {
            "people": [{
                "pose_keypoints_2d": extract_body25(results.pose_landmarks, w, h),
                "hand_left_keypoints_2d": extract_hand(results.left_hand_landmarks, w, h),
                "hand_right_keypoints_2d": extract_hand(results.right_hand_landmarks, w, h),
                "face_keypoints_2d": extract_face70(results.face_landmarks, w, h),
            }]
        }

        out_path = out_dir / f"{frame_idx:012d}_keypoints.json"
        with open(out_path, "w") as f:
            json.dump(keypoints, f)
        frame_idx += 1

    cap.release()
    holistic.close()
    return frame_idx


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------

def extract_all_openpose(src, dst):
    """OpenPose-format extraction (legacy API).

    Args:
        src: Directory with cropped video clips.
        dst: Output directory for keypoint JSONs.

    Returns:
        dict with 'total_frames' and 'clips' counts.
    """
    src, dst = Path(src), Path(dst)
    clips = sorted(p for p in src.glob("*/*.mp4") if not p.name.startswith("._"))
    print(f"Found {len(clips)} clips in {src}\n")

    total_frames = 0
    for i, clip in enumerate(clips):
        video_id = clip.parent.name
        clip_name = clip.stem
        out_dir = dst / video_id / clip_name

        if out_dir.exists() and any(out_dir.glob("*_keypoints.json")):
            existing = len(list(out_dir.glob("*_keypoints.json")))
            total_frames += existing
            continue

        n = process_video(clip, out_dir)
        total_frames += n
        print(f"  [{i+1}/{len(clips)}] {video_id}/{clip_name}: {n} frames")

    print(f"\nDone. {total_frames} total frames across {len(clips)} clips.")
    return {"total_frames": total_frames, "clips": len(clips)}


class _TimeoutError(Exception):
    pass


def _sigalrm_handler(signum, frame):
    raise _TimeoutError()


def extract_all_50joint(src, dst):
    """Extract 50-joint numpy arrays for all clips.

    Args:
        src: Directory with cropped video clips.
        dst: Output directory for pose npy files.

    Returns:
        dict with 'total' and 'failed' counts.
    """
    import signal

    src, dst = Path(src), Path(dst)
    clips = sorted(p for p in src.glob("*/*.mp4") if not p.name.startswith("._"))
    print(f"Found {len(clips)} clips in {src}\n")

    dst.mkdir(parents=True, exist_ok=True)
    timeout_sec = 120  # skip clips that take longer than 2 min
    use_alarm = hasattr(signal, "SIGALRM")  # Unix only

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
            arr = process_video_50joint(clip)
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
    default_openpose = _ROOT / "data/poses_openpose"
    default_50j = _ROOT / "data/usl-suspilne/poses/2d"

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["openpose", "50joint"], default="openpose",
                        help="openpose: per-frame JSON (BODY_25+hands+face); "
                             "50joint: per-clip numpy (50 joints)")
    parser.add_argument("--src", type=Path, default=default_src,
                        help=f"Source video directory (default: {default_src})")
    parser.add_argument("--dst", type=Path, default=None,
                        help="Output directory (default depends on mode)")
    args = parser.parse_args()
    if args.mode == "50joint":
        extract_all_50joint(src=args.src, dst=args.dst or default_50j)
    else:
        extract_all_openpose(src=args.src, dst=args.dst or default_openpose)
