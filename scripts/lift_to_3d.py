#!/usr/bin/env python3
"""
Lift 2D poses to 3D using the gopeith/SignLanguageProcessing 3DposeEstimator.

Pipeline (from the paper / repo):
  1. Decompose (x, y, confidence) triplets into separate matrices
  2. Normalize x/y to have zero mean and unit variance
  3. Prune frames where upper body has too many missing joints
  4. Interpolate missing points via weighted linear interpolation
  5. Initial 3D estimation (bone lengths + angles from 2D)
  6. Backpropagation-based filtering (TensorFlow optimisation)

Reads:  data/usl-suspilne/poses/2d/{videoId}/{clipIdx}.npy  — shape (T, 150), triplets of (x, y, confidence)
Writes: data/usl-suspilne/poses/3d/{videoId}/{clipIdx}.npy   — shape (T, 150), triplets of (x, y, z)

The 50-joint skeletal model:
  0: Nose, 1: Neck, 2: RShoulder, 3: RElbow, 4: RWrist,
  5: LShoulder, 6: LElbow, 7: LWrist,
  8-28: Left hand (21 MediaPipe landmarks),
  29-49: Right hand (21 MediaPipe landmarks)

Based on: https://github.com/gopeith/SignLanguageProcessing/tree/master/3DposeEstimator
License: MIT
"""

import math
import sys
from pathlib import Path

import numpy as np

# TensorFlow with TF1 compatibility
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
# Suppress excessive TF logging
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

ROOT = Path(__file__).resolve().parent.parent

# ============================================================================
# Skeletal model (from gopeith/SignLanguageProcessing/3DposeEstimator)
# ============================================================================

def get_skeletal_model_structure():
    """50-joint skeletal model: 8 upper body + 21 left hand + 21 right hand.

    Each tuple: (start_joint, end_joint, bone_type_index).
    The structure is a tree rooted at joint 0 (Nose).
    """
    return (
        # head
        (0, 1, 0),
        # left shoulder
        (1, 2, 1),
        # left arm
        (2, 3, 2),
        (3, 4, 3),
        # right shoulder
        (1, 5, 1),
        # right arm
        (5, 6, 2),
        (6, 7, 3),
        # left hand - wrist connection
        (7, 8, 4),
        # left hand - palm
        (8, 9, 5),
        (8, 13, 9),
        (8, 17, 13),
        (8, 21, 17),
        (8, 25, 21),
        # left hand - thumb
        (9, 10, 6),
        (10, 11, 7),
        (11, 12, 8),
        # left hand - index
        (13, 14, 10),
        (14, 15, 11),
        (15, 16, 12),
        # left hand - middle
        (17, 18, 14),
        (18, 19, 15),
        (19, 20, 16),
        # left hand - ring
        (21, 22, 18),
        (22, 23, 19),
        (23, 24, 20),
        # left hand - pinky
        (25, 26, 22),
        (26, 27, 23),
        (27, 28, 24),
        # right hand - wrist connection
        (4, 29, 4),
        # right hand - palm
        (29, 30, 5),
        (29, 34, 9),
        (29, 38, 13),
        (29, 42, 17),
        (29, 46, 21),
        # right hand - thumb
        (30, 31, 6),
        (31, 32, 7),
        (32, 33, 8),
        # right hand - index
        (34, 35, 10),
        (35, 36, 11),
        (36, 37, 12),
        # right hand - middle
        (38, 39, 14),
        (39, 40, 15),
        (40, 41, 16),
        # right hand - ring
        (42, 43, 18),
        (43, 44, 19),
        (44, 45, 20),
        # right hand - pinky
        (46, 47, 22),
        (47, 48, 23),
        (48, 49, 24),
    )


def structure_stats(structure):
    """Return (num_bone_types, num_joints)."""
    ps = set()
    ls = set()
    for a, b, l in structure:
        ps.add(a)
        ps.add(b)
        ls.add(l)
    return len(ls), len(ps)


# ============================================================================
# 2D Pose Processing (pose2D.py from 3DposeEstimator)
# ============================================================================

def normalization(Xx, Xy):
    """Normalize x/y coordinates to zero mean and unit variance (jointly)."""
    T, n = Xx.shape
    sum0 = T * n
    sum1Xx = np.sum(Xx)
    sum2Xx = np.sum(Xx * Xx)
    sum1Xy = np.sum(Xy)
    sum2Xy = np.sum(Xy * Xy)
    mux = sum1Xx / sum0
    muy = sum1Xy / sum0
    sum0_2 = 2 * sum0
    sum1 = sum1Xx + sum1Xy
    sum2 = sum2Xx + sum2Xy
    mu = sum1 / sum0_2
    sigma2 = (sum2 / sum0_2) - mu * mu
    if sigma2 < 1e-10:
        sigma2 = 1e-10
    sigma = math.sqrt(sigma2)
    return (Xx - mux) / sigma, (Xy - muy) / sigma


def prune(Xx, Xy, Xw, watch_joints, threshold, dtype):
    """Remove frames where core joints have low average confidence."""
    T, N = Xw.shape
    Yx = np.zeros((T, N), dtype=dtype)
    Yy = np.zeros((T, N), dtype=dtype)
    Yw = np.zeros((T, N), dtype=dtype)
    for t in range(T):
        avg_w = np.mean([Xw[t, i] for i in watch_joints])
        if avg_w >= threshold:
            Yx[t] = Xx[t]
            Yy[t] = Xy[t]
            Yw[t] = Xw[t]
    return Yx, Yy, Yw


def interpolation(Xx, Xy, Xw, threshold, dtype):
    """Weighted linear interpolation of missing points."""
    T, N = Xw.shape
    Yx = np.zeros((T, N), dtype=dtype)
    Yy = np.zeros((T, N), dtype=dtype)
    for t in range(T):
        for i in range(N):
            a1 = Xx[t, i]
            a2 = Xy[t, i]
            p = Xw[t, i]
            sumpa1 = p * a1
            sumpa2 = p * a2
            sump = p
            delta = 0
            while sump < threshold:
                change = False
                delta += 1
                t2 = t + delta
                if t2 < T:
                    a1 = Xx[t2, i]
                    a2 = Xy[t2, i]
                    p = Xw[t2, i]
                    sumpa1 += p * a1
                    sumpa2 += p * a2
                    sump += p
                    change = True
                t2 = t - delta
                if t2 >= 0:
                    a1 = Xx[t2, i]
                    a2 = Xy[t2, i]
                    p = Xw[t2, i]
                    sumpa1 += p * a1
                    sumpa2 += p * a2
                    sump += p
                    change = True
                if not change:
                    break
            if sump <= 0.0:
                sump = 1e-10
            Yx[t, i] = sumpa1 / sump
            Yy[t, i] = sumpa2 / sump
    return Yx, Yy, Xw


# ============================================================================
# 2D → 3D Initialization (pose2Dto3D.py from 3DposeEstimator)
# ============================================================================

def _norm(x):
    return math.sqrt(sum(i * i for i in x))


def _perc(lst, p):
    lst.sort()
    return lst[int(p * (len(lst) - 1))]


def _compute_b(ax, ay, az, tx, ty, L):
    """Estimate the 3D angle of a bone given 2D target position."""
    hyps = [[tx - ax, ty - ay, 0]]
    foo = L**2 - (tx - ax)**2 - (ty - ay)**2
    if foo >= 0:
        hyps.append([tx - ax, ty - ay, -math.sqrt(foo)])
        hyps.append([tx - ax, ty - ay, +math.sqrt(foo)])

    foo1 = ax**2 - 2*ax*tx + ay**2 - 2*ay*ty + tx**2 + ty**2
    if foo1 > 1e-10:
        foo2 = (1/foo1)**(1/2)
        foo3 = (ay**3/foo1 + L*ay*foo2 - L*ty*foo2
                + (ax**2*ay)/foo1 + (ay*tx**2)/foo1 + (ay*ty**2)/foo1
                - (2*ay**2*ty)/foo1 - (2*ax*ay*tx)/foo1)
        foo4 = (ay**3/foo1 - L*ay*foo2 + L*ty*foo2
                + (ax**2*ay)/foo1 + (ay*tx**2)/foo1 + (ay*ty**2)/foo1
                - (2*ay**2*ty)/foo1 - (2*ax*ay*tx)/foo1)
        if (ay - ty) != 0:
            xx1 = -(ax*ty - ay*tx - ax*foo3 + tx*foo3) / (ay - ty)
            xx2 = -(ax*ty - ay*tx - ax*foo4 + tx*foo4) / (ay - ty)
            xy1 = foo3
            xy2 = foo4
            if 0 * xx1 * xx2 * xy1 * xy2 == 0:
                hyps.append([xx1 - ax, xy1 - ay, 0])
                hyps.append([xx2 - ax, xy2 - ay, 0])

    angle = [1.0, 1.0, 0.0]
    Lmin = None
    for hypangle in hyps:
        normHypangle = _norm(hypangle) + 1e-10
        xi = [
            ax + L * hypangle[0] / normHypangle,
            ay + L * hypangle[1] / normHypangle,
            az + L * hypangle[2] / normHypangle,
        ]
        Li = (xi[0] - tx)**2 + (xi[1] - ty)**2
        if Lmin is None or Lmin > Li:
            Lmin = Li
            angle = hypangle
    return angle


def initialization(Xx, Xy, Xw, structure, sigma, rng, dtype):
    """Initial 3D pose estimation from 2D.

    Returns:
        lines: log of bone lengths, shape (nBoneTypes,)
        rootsx/y/z: head position, shape (T, 1)
        anglesx/y/z: limb angles, shape (T, nLimbs)
        Yx/Yy/Yz: 3D coordinates, shape (T, nJoints)
    """
    T, n = Xx.shape
    nBones, nPoints = structure_stats(structure)
    nLimbs = len(structure)

    lines = np.zeros((nBones,), dtype=dtype)
    rootsx = Xx[:, 0].copy() + np.asarray(
        rng.uniform(-sigma, sigma, (T,)), dtype=dtype)
    rootsy = Xy[:, 0].copy() + np.asarray(
        rng.uniform(-sigma, sigma, (T,)), dtype=dtype)
    rootsz = np.zeros((T,), dtype=dtype) + np.asarray(
        rng.uniform(-sigma, sigma, (T,)), dtype=dtype)

    anglesx = np.zeros((T, nLimbs), dtype=dtype)
    anglesy = np.zeros((T, nLimbs), dtype=dtype)
    anglesz = np.zeros((T, nLimbs), dtype=dtype)

    Yx = np.zeros((T, n), dtype=dtype)
    Yy = np.zeros((T, n), dtype=dtype)
    Yz = np.zeros((T, n), dtype=dtype)
    Yx[:, 0] = rootsx
    Yy[:, 0] = rootsy
    Yz[:, 0] = rootsz

    # Estimate bone lengths from 2D distances
    Ls = {}
    for iBone, (a, b, line) in enumerate(structure):
        if line not in Ls:
            Ls[line] = []
        for t in range(T):
            w = min(Xw[t, a], Xw[t, b])
            L = _norm([Xx[t, a] - Xx[t, b], Xy[t, a] - Xy[t, b]])
            Ls[line].append(L)

    for i in range(len(lines)):
        val = _perc(Ls[i], 0.5)
        lines[i] = math.log(max(val, 1e-10))

    # Compute angles and initial 3D positions
    for iBone, (a, b, line) in enumerate(structure):
        L = math.exp(lines[line])
        for t in range(T):
            ax, ay, az = Yx[t, a], Yy[t, a], Yz[t, a]
            tx, ty = Xx[t, b], Xy[t, b]
            anglex, angley, anglez = _compute_b(ax, ay, az, tx, ty, L)

            # Handle inf/nan
            if not (0.0 * anglex == 0.0):
                anglex = 0.0
            if not (0.0 * angley == 0.0):
                angley = 0.0
            if not (0.0 * anglez == 0.0):
                anglez = 0.0
            if anglex == 0.0 and angley == 0.0 and anglez == 0.0:
                anglex = angley = anglez = 1.0
            if anglez < 0.0:
                anglez = -anglez
            anglez += 0.001

            normAngle = math.sqrt(
                anglex**2 + angley**2 + anglez**2) + 1e-10
            anglesx[t, iBone] = anglex / normAngle
            anglesy[t, iBone] = angley / normAngle
            anglesz[t, iBone] = anglez / normAngle

        for t in range(T):
            Yx[t, b] = Yx[t, a] + L * anglesx[t, iBone]
            Yy[t, b] = Yy[t, a] + L * anglesy[t, iBone]
            Yz[t, b] = Yz[t, a] + L * anglesz[t, iBone]

    rootsx = rootsx.reshape((T, 1))
    rootsy = rootsy.reshape((T, 1))
    rootsz = rootsz.reshape((T, 1))

    return lines, rootsx, rootsy, rootsz, anglesx, anglesy, anglesz, Yx, Yy, Yz


# ============================================================================
# 3D Pose Filtering via Backpropagation (pose3D.py from 3DposeEstimator)
# ============================================================================

def backpropagation_based_filtering(
    lines0_values, rootsx0_values, rootsy0_values, rootsz0_values,
    anglesx0_values, anglesy0_values, anglesz0_values,
    tarx_values, tary_values, w_values,
    structure, dtype,
    learning_rate=0.1, n_cycles=1000,
    regulator_rates=(0.001, 0.1),
):
    """Refine 3D poses using TF gradient-descent optimisation."""
    T = rootsx0_values.shape[0]
    nBones, nPoints = structure_stats(structure)
    nLimbs = len(structure)

    tf.reset_default_graph()

    lines = tf.Variable(lines0_values, dtype=dtype)
    rootsx = tf.Variable(rootsx0_values, dtype=dtype)
    rootsy = tf.Variable(rootsy0_values, dtype=dtype)
    rootsz = tf.Variable(rootsz0_values, dtype=dtype)
    anglesx = tf.Variable(anglesx0_values, dtype=dtype)
    anglesy = tf.Variable(anglesy0_values, dtype=dtype)
    anglesz = tf.Variable(anglesz0_values, dtype=dtype)

    tarx = tf.placeholder(dtype=dtype)
    tary = tf.placeholder(dtype=dtype)
    w = tf.placeholder(dtype=dtype)

    x = [None] * nPoints
    y = [None] * nPoints
    z = [None] * nPoints

    x[0] = rootsx
    y[0] = rootsy
    z[0] = rootsz

    epsilon = 1e-10
    i = 0
    for a, b, l in structure:
        L = tf.exp(lines[l])
        Ax = anglesx[:, i:(i + 1)]
        Ay = anglesy[:, i:(i + 1)]
        Az = anglesz[:, i:(i + 1)]
        normA = tf.sqrt(tf.square(Ax) + tf.square(Ay) + tf.square(Az)) + epsilon
        x[b] = x[a] + L * Ax / normA
        y[b] = y[a] + L * Ay / normA
        z[b] = z[a] + L * Az / normA
        i += 1

    x = tf.concat(x, axis=1)
    y = tf.concat(y, axis=1)
    z = tf.concat(z, axis=1)

    loss = tf.reduce_sum(
        w * tf.square(x - tarx) + w * tf.square(y - tary)
    ) / (T * nPoints)

    reg1 = tf.reduce_sum(tf.exp(lines))
    dx = x[:(T - 1), :nPoints] - x[1:T, :nPoints]
    dy = y[:(T - 1), :nPoints] - y[1:T, :nPoints]
    dz = z[:(T - 1), :nPoints] - z[1:T, :nPoints]
    reg2 = tf.reduce_sum(
        tf.square(dx) + tf.square(dy) + tf.square(dz)
    ) / ((T - 1) * nPoints)

    optimize_this = loss + regulator_rates[0] * reg1 + regulator_rates[1] * reg2

    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train = optimizer.minimize(optimize_this)
    init = tf.variables_initializer(tf.global_variables())

    sess = tf.Session()
    sess.run(init)
    feed = {tarx: tarx_values, tary: tary_values, w: w_values}
    for cycle in range(n_cycles):
        sess.run(train, feed)
        if cycle % 100 == 0 or cycle == n_cycles - 1:
            loss_val = sess.run(loss, feed)
            print(f"    cycle {cycle:4d}/{n_cycles}, loss = {loss_val:.6e}")

    result = sess.run([x, y, z], {})
    sess.close()
    return result


# ============================================================================
# Main pipeline
# ============================================================================

def lift_clip(poses_2d: np.ndarray, structure, dtype="float32") -> np.ndarray:
    """Lift a single clip's 2D poses to 3D.

    Args:
        poses_2d: shape (T, 150) with (x, y, confidence) triplets per joint
        structure: skeletal model structure

    Returns:
        poses_3d: shape (T, 150) with (x, y, z) triplets per joint
    """
    T = poses_2d.shape[0]
    if T < 2:
        # Need at least 2 frames for the pipeline
        return np.zeros((T, 150), dtype=dtype)

    # Decompose into x, y, w matrices (each shape T×50)
    Xx = poses_2d[:, 0::3].astype(dtype)  # every 3rd starting at 0
    Xy = poses_2d[:, 1::3].astype(dtype)  # every 3rd starting at 1
    Xw = poses_2d[:, 2::3].astype(dtype)  # every 3rd starting at 2

    # Step 1: Normalization
    Xx, Xy = normalization(Xx, Xy)

    # Step 2: Prune frames with too many missing upper-body joints
    Xx, Xy, Xw = prune(Xx, Xy, Xw, (0, 1, 2, 3, 4, 5, 6, 7), 0.3, dtype)

    # Step 3: Interpolation of missing points
    Xx, Xy, Xw = interpolation(Xx, Xy, Xw, 0.99, dtype)

    # Step 4: Initial 3D estimation
    rng = np.random.RandomState(1234)
    (lines0, rootsx0, rootsy0, rootsz0,
     anglesx0, anglesy0, anglesz0,
     Yx0, Yy0, Yz0) = initialization(
        Xx, Xy, Xw, structure, 0.001, rng, dtype
    )

    # Step 5: Backpropagation-based filtering
    Yx, Yy, Yz = backpropagation_based_filtering(
        lines0, rootsx0, rootsy0, rootsz0,
        anglesx0, anglesy0, anglesz0,
        Xx, Xy, Xw,
        structure, dtype,
    )

    # Reassemble into (T, 150) with (x, y, z) triplets
    poses_3d = np.zeros((T, 150), dtype=dtype)
    for j in range(50):
        poses_3d[:, j * 3 + 0] = Yx[:, j]
        poses_3d[:, j * 3 + 1] = Yy[:, j]
        poses_3d[:, j * 3 + 2] = Yz[:, j]

    return poses_3d


def lift_all(src, dst):
    """Lift all 2D poses to 3D.

    Args:
        src: Directory with 2D pose npy files.
        dst: Output directory for 3D pose npy files.

    Returns:
        dict with 'total' and 'skipped' counts.
    """
    src, dst = Path(src), Path(dst)
    clips = sorted(src.glob("*/*.npy"))
    print(f"Found {len(clips)} 2D pose files in {src}\n")

    if not clips:
        print("No 2D poses found. Run extract_poses.py --mode 50joint first.")
        sys.exit(1)

    dst.mkdir(parents=True, exist_ok=True)
    structure = get_skeletal_model_structure()
    skipped = 0

    for i, clip_path in enumerate(clips):
        video_id = clip_path.parent.name
        clip_name = clip_path.stem
        out_path = dst / video_id / f"{clip_name}.npy"

        if out_path.exists():
            arr = np.load(out_path)
            print(f"  [{i+1}/{len(clips)}] {video_id}/{clip_name}: "
                  f"{arr.shape[0]} frames (cached)")
            skipped += 1
            continue

        poses_2d = np.load(clip_path)
        T = poses_2d.shape[0]
        print(f"  [{i+1}/{len(clips)}] {video_id}/{clip_name}: "
              f"{T} frames, lifting to 3D ...")

        poses_3d = lift_clip(poses_2d, structure)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, poses_3d)
        print(f"    → saved {out_path}")

    print(f"\nDone. 3D poses written to {dst}")
    return {"total": len(clips), "skipped": skipped}


def main():
    import argparse

    default_src = ROOT / "data/usl-suspilne/poses/2d"
    default_dst = ROOT / "data/usl-suspilne/poses/3d"

    parser = argparse.ArgumentParser(description="Lift 2D poses to 3D")
    parser.add_argument("--src", type=Path, default=default_src,
                        help=f"2D poses directory (default: {default_src})")
    parser.add_argument("--dst", type=Path, default=default_dst,
                        help=f"3D poses output directory (default: {default_dst})")
    args = parser.parse_args()

    lift_all(src=args.src, dst=args.dst)


if __name__ == "__main__":
    main()
