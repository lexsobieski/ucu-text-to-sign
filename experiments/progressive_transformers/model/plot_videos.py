import sys
import math
import numpy as np
import cv2
import os
import _pickle as cPickle
import gzip
import subprocess
import torch

from pathlib import Path

from dtw import dtw
from constants import PAD_TOKEN

_NORM_PATH = Path(__file__).resolve().parent / "Configs" / "usl_pose_norm.npz"
_NORM_CACHE = None


def _load_norm_stats():
    """Return (mean, std) arrays of shape (150,) or (None, None) if unavailable."""
    global _NORM_CACHE
    if _NORM_CACHE is None:
        if _NORM_PATH.exists():
            data = np.load(_NORM_PATH)
            _NORM_CACHE = (data["mean"].astype(np.float32), data["std"].astype(np.float32))
        else:
            _NORM_CACHE = (None, None)
    return _NORM_CACHE


def _denormalize_sequence(seq):
    """Denormalize a (T, 151) sequence in-place (150 pose dims + 1 counter)."""
    mean, std = _load_norm_stats()
    if mean is None:
        return seq
    seq = np.asarray(seq, dtype=np.float32).copy()
    seq[:, :150] = seq[:, :150] * std + mean
    return seq

# Compute global coordinate bounds across all frames (and optionally references)
# so that every frame uses the same scale and the skeleton doesn't jump around.
def _compute_global_bounds(joints, references=None):
    all_xy = []
    for frame_joints in joints:
        if PAD_TOKEN in frame_joints:
            continue
        fj = np.array(frame_joints[:-1])  # cut counter
        xy = np.reshape(fj, (50, 3))[:, :2]
        valid = xy[np.any(xy != 0, axis=1)]
        if len(valid) > 0:
            all_xy.append(valid)
    if references is not None:
        for ref_joints in references:
            if PAD_TOKEN in ref_joints:
                continue
            rj = np.array(ref_joints[:-1])
            xy = np.reshape(rj, (50, 3))[:, :2]
            valid = xy[np.any(xy != 0, axis=1)]
            if len(valid) > 0:
                all_xy.append(valid)
    if all_xy:
        all_xy = np.concatenate(all_xy, axis=0)
        return all_xy.min(axis=0), all_xy.max(axis=0)
    return np.array([0, 0]), np.array([1, 1])


# Plot a video given a tensor of joints, a file path, video name and references/sequence ID
def plot_video(joints,
               file_path,
               video_name,
               references=None,
               skip_frames=1,
               sequence_ID=None):
    # Create video template
    FPS = (25 // skip_frames)
    video_file = file_path + "/{}.mp4".format(video_name.split(".")[0])
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    if references is None:
        video = cv2.VideoWriter(video_file, fourcc, float(FPS), (650, 650), True)
    elif references is not None:
        video = cv2.VideoWriter(video_file, fourcc, float(FPS), (1300, 650), True)  # Long

    # Denormalize predictions and references into raw pose space for correct spatial layout
    joints = _denormalize_sequence(joints)
    if references is not None:
        references = _denormalize_sequence(references)

    # Compute global bounds once so all frames share the same scale
    global_min, global_max = _compute_global_bounds(joints, references)

    num_frames = 0

    for (j, frame_joints) in enumerate(joints):

        # Reached padding
        if PAD_TOKEN in frame_joints:
            continue

        # Initialise frame of white
        frame = np.ones((650, 650, 3), np.uint8) * 255

        # Cut off the counter token
        frame_joints = frame_joints[:-1]

        # Reduce the frame joints down to 2D for visualisation
        frame_joints_2d = np.reshape(frame_joints, (50, 3))[:, :2]
        # Draw the frame given 2D joints
        draw_frame_2D(frame, frame_joints_2d, global_min, global_max)

        cv2.putText(frame, "Predicted Sign Pose", (180, 600), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 255), 2)

        # If reference is provided, create and concatenate on the end
        if references is not None:
            # Extract the reference joints
            ref_joints = references[j]
            # Initialise frame of white
            ref_frame = np.ones((650, 650, 3), np.uint8) * 255

            # Cut off the counter token
            ref_joints = ref_joints[:-1]

            # Reduce the frame joints down to 2D
            ref_joints_2d = np.reshape(ref_joints, (50, 3))[:, :2]

            # Draw these joints on the frame
            draw_frame_2D(ref_frame, ref_joints_2d, global_min, global_max)

            cv2.putText(ref_frame, "Ground Truth Pose", (190, 600), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 0), 2)

            frame = np.concatenate((frame, ref_frame), axis=1)

            sequence_ID_write = "Sequence ID: " + sequence_ID.split("/")[-1]
            cv2.putText(frame, sequence_ID_write, (700, 635), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 0), 2)
        # Write the video frame
        video.write(frame)
        num_frames += 1
    # Release the video
    video.release()

# This is the format of the 3D data, outputted from the Inverse Kinematics model
def getSkeletalModelStructure():
    # Definition of skeleton model structure:
    #   The structure is an n-tuple of:
    #
    #   (index of a start point, index of an end point, index of a bone)
    #
    #   E.g., this simple skeletal model
    #
    #             (0)
    #              |
    #              |
    #              0
    #              |
    #              |
    #     (2)--1--(1)--1--(3)
    #      |               |
    #      |               |
    #      2               2
    #      |               |
    #      |               |
    #     (4)             (5)
    #
    #   has this structure:
    #
    #   (
    #     (0, 1, 0),
    #     (1, 2, 1),
    #     (1, 3, 1),
    #     (2, 4, 2),
    #     (3, 5, 2),
    #   )
    #
    #  Warning 1: The structure has to be a tree.
    #  Warning 2: The order isn't random. The order is from a root to lists.
    #

    return (
        # head
        (0, 1, 0),

        # left shoulder
        (1, 2, 1),

        # left arm
        (2, 3, 2),
        # (3, 4, 3),
        # Changed to avoid wrist, go straight to hands
        (3, 29, 3),

        # right shoulder
        (1, 5, 1),

        # right arm
        (5, 6, 2),
        # (6, 7, 3),
        # Changed to avoid wrist, go straight to hands
        (6, 8, 3),

        # left hand - wrist
        # (7, 8, 4),

        # left hand - palm
        (8, 9, 5),
        (8, 13, 9),
        (8, 17, 13),
        (8, 21, 17),
        (8, 25, 21),

        # left hand - 1st finger
        (9, 10, 6),
        (10, 11, 7),
        (11, 12, 8),

        # left hand - 2nd finger
        (13, 14, 10),
        (14, 15, 11),
        (15, 16, 12),

        # left hand - 3rd finger
        (17, 18, 14),
        (18, 19, 15),
        (19, 20, 16),

        # left hand - 4th finger
        (21, 22, 18),
        (22, 23, 19),
        (23, 24, 20),

        # left hand - 5th finger
        (25, 26, 22),
        (26, 27, 23),
        (27, 28, 24),

        # right hand - wrist
        # (4, 29, 4),

        # right hand - palm
        (29, 30, 5),
        (29, 34, 9),
        (29, 38, 13),
        (29, 42, 17),
        (29, 46, 21),

        # right hand - 1st finger
        (30, 31, 6),
        (31, 32, 7),
        (32, 33, 8),

        # right hand - 2nd finger
        (34, 35, 10),
        (35, 36, 11),
        (36, 37, 12),

        # right hand - 3rd finger
        (38, 39, 14),
        (39, 40, 15),
        (40, 41, 16),

        # right hand - 4th finger
        (42, 43, 18),
        (43, 44, 19),
        (44, 45, 20),

        # right hand - 5th finger
        (46, 47, 22),
        (47, 48, 23),
        (48, 49, 24),
    )

# Draw a line between two points, if they are positive points
def draw_line(im, joint1, joint2, c=(0, 0, 255),t=1, width=3):
    thresh = -100
    if joint1[0] > thresh and  joint1[1] > thresh and joint2[0] > thresh and joint2[1] > thresh:

        center = (int((joint1[0] + joint2[0]) / 2), int((joint1[1] + joint2[1]) / 2))

        length = int(math.sqrt(((joint1[0] - joint2[0]) ** 2) + ((joint1[1] - joint2[1]) ** 2))/2)

        angle = math.degrees(math.atan2((joint1[0] - joint2[0]),(joint1[1] - joint2[1])))

        cv2.ellipse(im, center, (width,length), -angle,0.0,360.0, c, -1)

# Draw the frame given 2D joints that are in the Inverse Kinematics format
def draw_frame_2D(frame, joints, global_min=None, global_max=None):
    # Line to be between the stacked
    draw_line(frame, [1, 650], [1, 1], c=(0,0,0), t=1, width=1)

    # Get the skeleton structure details of each bone, and size
    skeleton = getSkeletalModelStructure()
    skeleton = np.array(skeleton)

    number = skeleton.shape[0]

    # Use global bounds (shared across all frames) to scale joints into the 650x650 frame.
    margin = 50
    if global_min is not None and global_max is not None:
        x_min, y_min = global_min
        x_max, y_max = global_max
    else:
        valid = joints[np.any(joints != 0, axis=1)]
        if len(valid) > 0:
            x_min, y_min = valid.min(axis=0)
            x_max, y_max = valid.max(axis=0)
        else:
            x_min, y_min, x_max, y_max = 0, 0, 1, 1
    data_range = max(x_max - x_min, y_max - y_min, 1e-6)
    frame_size = 650 - 2 * margin
    scale = frame_size / data_range
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    joints = (joints - np.array([x_center, y_center])) * scale + np.array([325, 325])

    # Loop through each of the bone structures, and plot the bone
    for j in range(number):

        c = get_bone_colour(skeleton,j)

        draw_line(frame, [joints[skeleton[j, 0]][0], joints[skeleton[j, 0]][1]],
                  [joints[skeleton[j, 1]][0], joints[skeleton[j, 1]][1]], c=c, t=1, width=1)

# get bone colour given index
def get_bone_colour(skeleton,j):
    bone = skeleton[j, 2]

    if bone == 0:  # head
        c = (0, 153, 0)
    elif bone == 1:  # Shoulder
        c = (0, 0, 255)

    elif bone == 2 and skeleton[j, 1] == 3:  # left arm
        c = (0, 102, 204)
    elif bone == 3 and skeleton[j, 0] == 3:  # left lower arm
        c = (0, 204, 204)

    elif bone == 2 and skeleton[j, 1] == 6:  # right arm
        c = (0, 153, 0)
    elif bone == 3 and skeleton[j, 0] == 6:  # right lower arm
        c = (0, 204, 0)

    # Hands
    elif bone in [5, 6, 7, 8]:
        c = (0, 0, 255)
    elif bone in [9, 10, 11, 12]:
        c = (51, 255, 51)
    elif bone in [13, 14, 15, 16]:
        c = (255, 0, 0)
    elif bone in [17, 18, 19, 20]:
        c = (204, 153, 255)
    elif bone in [21, 22, 23, 24]:
        c = (51, 255, 255)

    return c

# Apply DTW to the produced sequence, so it can be visually compared to the reference sequence
def alter_DTW_timing(pred_seq,ref_seq):

    # Define a cost function
    euclidean_norm = lambda x, y: np.sum(np.abs(x - y))

    # Cut the reference down to the max count value
    _ , ref_max_idx = torch.max(ref_seq[:, -1], 0)
    if ref_max_idx == 0: ref_max_idx += 1
    # Cut down frames by counter
    ref_seq = ref_seq[:ref_max_idx,:].cpu().numpy()

    # Cut the hypothesis down to the max count value
    _, hyp_max_idx = torch.max(pred_seq[:, -1], 0)
    if hyp_max_idx == 0: hyp_max_idx += 1
    # Cut down frames by counter
    pred_seq = pred_seq[:hyp_max_idx,:].cpu().numpy()

    # Run DTW on the reference and predicted sequence
    d, cost_matrix, acc_cost_matrix, path = dtw(ref_seq[:,:-1], pred_seq[:,:-1], dist=euclidean_norm)

    # Normalise the dtw cost by sequence length
    d = d / acc_cost_matrix.shape[0]

    # Initialise new sequence
    new_pred_seq = np.zeros_like(ref_seq)
    # j tracks the position in the reference sequence
    j = 0
    skips = 0
    squeeze_frames = []
    for (i, pred_num) in enumerate(path[0]):

        if i == len(path[0]) - 1:
            break

        if path[1][i] == path[1][i + 1]:
            skips += 1

        # If a double coming up
        if path[0][i] == path[0][i + 1]:
            squeeze_frames.append(pred_seq[i - skips])
            j += 1
        # Just finished a double
        elif path[0][i] == path[0][i - 1]:
            new_pred_seq[pred_num] = avg_frames(squeeze_frames)
            squeeze_frames = []
        else:
            new_pred_seq[pred_num] = pred_seq[i - skips]

    return new_pred_seq, ref_seq, d

# Find the average of the given frames
def avg_frames(frames):
    frames_sum = np.zeros_like(frames[0])
    for frame in frames:
        frames_sum += frame

    avg_frame = frames_sum / len(frames)
    return avg_frame