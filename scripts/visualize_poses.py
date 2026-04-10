#!/usr/bin/env python3
"""
Visualize extracted 2D poses overlaid on the original video frames.

Usage:
    python scripts/visualize_poses.py                          # first clip
    python scripts/visualize_poses.py -_4_XPQCVwE/0003        # specific clip
    python scripts/visualize_poses.py -_4_XPQCVwE/0003 --3d   # visualize 3D poses

Opens a window showing the video with skeleton overlay.
Press Q to quit, any other key for next frame.
Pass --save to write an output video instead of displaying.
"""

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
VIDEO_DIR = ROOT / "data/usl-suspilne/features"
POSES_2D_DIR = ROOT / "data/usl-suspilne/poses/2d"
POSES_3D_DIR = ROOT / "data/usl-suspilne/poses/3d"
OUTPUT_DIR = ROOT / "data/pose_visualizations"

# 50-joint skeleton connections for drawing
# Body: Nose(0) - Neck(1), Neck - RShoulder(2), RShoulder - RElbow(3),
#        RElbow - RWrist(4), Neck - LShoulder(5), LShoulder - LElbow(6),
#        LElbow - LWrist(7)
BODY_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7),
]

# Hand connections (21 landmarks each):
# Wrist(0) -> Thumb: 1-2-3-4
# Wrist(0) -> Index: 5-6-7-8
# Wrist(0) -> Middle: 9-10-11-12
# Wrist(0) -> Ring: 13-14-15-16
# Wrist(0) -> Pinky: 17-18-19-20
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    # Palm connections
    (5, 9), (9, 13), (13, 17),
]

BODY_COLOR = (0, 255, 0)       # Green
LEFT_HAND_COLOR = (255, 0, 0)  # Blue
RIGHT_HAND_COLOR = (0, 0, 255) # Red
MISSING_COLOR = (128, 128, 128) # Gray


def draw_skeleton_2d(frame, pose_row, w, h):
    """Draw 2D skeleton on a frame.

    pose_row: 150 values = 50 joints x (x, y, confidence)
    x, y are normalized [0, 1]
    """
    joints = []
    for j in range(50):
        x = pose_row[j * 3 + 0]
        y = pose_row[j * 3 + 1]
        c = pose_row[j * 3 + 2]
        joints.append((int(x * w), int(y * h), c))

    # Draw body
    for a, b in BODY_CONNECTIONS:
        if joints[a][2] > 0.1 and joints[b][2] > 0.1:
            cv2.line(frame, joints[a][:2], joints[b][:2], BODY_COLOR, 2)
    # Body-to-hand wrist connections
    if joints[7][2] > 0.1 and joints[8][2] > 0.1:  # LWrist to left hand wrist
        cv2.line(frame, joints[7][:2], joints[8][:2], LEFT_HAND_COLOR, 1)
    if joints[4][2] > 0.1 and joints[29][2] > 0.1:  # RWrist to right hand wrist
        cv2.line(frame, joints[4][:2], joints[29][:2], RIGHT_HAND_COLOR, 1)

    # Draw left hand (joints 8-28)
    for a, b in HAND_CONNECTIONS:
        ja, jb = a + 8, b + 8
        if joints[ja][2] > 0.1 and joints[jb][2] > 0.1:
            cv2.line(frame, joints[ja][:2], joints[jb][:2], LEFT_HAND_COLOR, 1)

    # Draw right hand (joints 29-49)
    for a, b in HAND_CONNECTIONS:
        ja, jb = a + 29, b + 29
        if joints[ja][2] > 0.1 and joints[jb][2] > 0.1:
            cv2.line(frame, joints[ja][:2], joints[jb][:2], RIGHT_HAND_COLOR, 1)

    # Draw joint points
    for j_idx, (jx, jy, jc) in enumerate(joints):
        if jc > 0.1:
            if j_idx < 8:
                color = BODY_COLOR
            elif j_idx < 29:
                color = LEFT_HAND_COLOR
            else:
                color = RIGHT_HAND_COLOR
            cv2.circle(frame, (jx, jy), 3, color, -1)
        else:
            cv2.circle(frame, (jx, jy), 2, MISSING_COLOR, -1)

    return frame


def draw_skeleton_3d(frame, pose_row, w, h):
    """Draw 3D skeleton on a frame (projects x, y; ignores z).

    pose_row: 150 values = 50 joints x (x, y, z)
    After 3D lifting, coordinates are normalized (zero-mean, unit-var),
    so we need to rescale to frame dimensions.
    """
    xs = pose_row[0::3]
    ys = pose_row[1::3]

    # Rescale from normalized coords to pixel coords
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    x_range = max(x_max - x_min, 1e-6)
    y_range = max(y_max - y_min, 1e-6)
    # Maintain aspect ratio, add padding
    scale = max(x_range, y_range)
    margin = 0.1

    joints = []
    for j in range(50):
        x = pose_row[j * 3 + 0]
        y = pose_row[j * 3 + 1]
        px = int(((x - x_min) / scale * (1 - 2 * margin) + margin) * w)
        py = int(((y - y_min) / scale * (1 - 2 * margin) + margin) * h)
        joints.append((px, py, 1.0))  # All visible for 3D

    # Draw body
    for a, b in BODY_CONNECTIONS:
        cv2.line(frame, joints[a][:2], joints[b][:2], BODY_COLOR, 2)
    # Wrist-to-hand
    cv2.line(frame, joints[7][:2], joints[8][:2], LEFT_HAND_COLOR, 1)
    cv2.line(frame, joints[4][:2], joints[29][:2], RIGHT_HAND_COLOR, 1)

    # Left hand
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, joints[a + 8][:2], joints[b + 8][:2], LEFT_HAND_COLOR, 1)

    # Right hand
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, joints[a + 29][:2], joints[b + 29][:2], RIGHT_HAND_COLOR, 1)

    # Joint points
    for j_idx, (jx, jy, _) in enumerate(joints):
        color = BODY_COLOR if j_idx < 8 else (LEFT_HAND_COLOR if j_idx < 29 else RIGHT_HAND_COLOR)
        cv2.circle(frame, (jx, jy), 3, color, -1)

    return frame


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("clip", nargs="?", default=None,
                        help="Clip name like '-_4_XPQCVwE/0003'")
    parser.add_argument("--3d", dest="use_3d", action="store_true",
                        help="Visualize 3D poses instead of 2D")
    parser.add_argument("--save", action="store_true",
                        help="Save output video instead of displaying")
    args = parser.parse_args()

    # Pick a clip
    if args.clip:
        video_id, clip_idx = args.clip.split("/")
    else:
        # Use first available clip
        first_dir = sorted(POSES_2D_DIR.iterdir())[0]
        first_file = sorted(first_dir.glob("*.npy"))[0]
        video_id = first_dir.name
        clip_idx = first_file.stem
        print(f"No clip specified, using: {video_id}/{clip_idx}")

    # Load poses
    if args.use_3d:
        pose_path = POSES_3D_DIR / video_id / f"{clip_idx}.npy"
        if not pose_path.exists():
            print(f"3D poses not found: {pose_path}")
            sys.exit(1)
        draw_fn = draw_skeleton_3d
        label = "3D"
    else:
        pose_path = POSES_2D_DIR / video_id / f"{clip_idx}.npy"
        if not pose_path.exists():
            print(f"2D poses not found: {pose_path}")
            sys.exit(1)
        draw_fn = draw_skeleton_2d
        label = "2D"

    poses = np.load(pose_path)
    print(f"Loaded {label} poses: {poses.shape} from {pose_path}")

    # Load video
    video_path = VIDEO_DIR / video_id / f"{clip_idx}.mp4"
    if not video_path.exists():
        print(f"Video not found: {video_path}")
        print("Drawing skeleton on blank canvas instead.")
        use_video = False
        w, h = 640, 480
    else:
        use_video = True

    if use_video:
        cap = cv2.VideoCapture(str(video_path))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    else:
        fps = 25.0

    writer = None
    if args.save:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUTPUT_DIR / f"{video_id}_{clip_idx}_{label}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
        print(f"Writing to {out_path}")

    for i in range(poses.shape[0]):
        if use_video:
            ret, frame = cap.read()
            if not ret:
                break
        else:
            frame = np.zeros((h, w, 3), dtype=np.uint8)

        frame = draw_fn(frame, poses[i], w, h)

        # Add frame info
        cv2.putText(frame, f"{label} | {video_id}/{clip_idx} | frame {i}/{poses.shape[0]}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        if writer:
            writer.write(frame)
        else:
            cv2.imshow(f"Pose Visualization - {label}", frame)
            key = cv2.waitKey(int(1000 / fps)) & 0xFF
            if key == ord("q"):
                break

    if use_video:
        cap.release()
    if writer:
        writer.release()
        print(f"Saved: {out_path}")
    else:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
