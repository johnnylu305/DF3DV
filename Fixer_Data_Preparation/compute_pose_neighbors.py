# Pose distance used for nearest-neighbor retrieval:
#   distance =
#       ||Ca - Cb|| / scene_radius
#       + lambda_rot_deg * rotation_angle_deg
#
# Translation is normalized by a robust scene radius so that the same
# lambda_rot_deg can be used across scenes with different scales.
#
# Default:
#   lambda_rot_deg = 0.05
#
# Interpretation:
#   1° rotation  ≈ 0.05 normalized translation units
#   20° rotation ≈ 1 scene-radius worth of translation
#
# This balances viewpoint similarity and camera proximity when ranking
# neighboring images.
#
# Increase lambda_rot_deg to prefer images with similar orientations,
# even if they are farther apart.
#
# Decrease lambda_rot_deg to prefer physically closer cameras,
# even if their viewing directions differ.
#
# ex:
#   python compute_pose_neighbors.py --root DF3DV-1K-Fixer --overwrite

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


# ----------------------------
# Data structure
# ----------------------------
@dataclass(frozen=True)
class PoseW2C:
    """COLMAP stores world-to-camera: x_cam = R * x_world + t."""
    image_id: int
    name: str
    R: np.ndarray  # (3,3)
    t: np.ndarray  # (3,)


# ----------------------------
# Path inference
# ----------------------------
def infer_sparse0_from_scene(scene_root: Path) -> Tuple[Path, Path]:
    """
    scene_root example:
      Root/Val/DF3DV-41-Fixer/071125-CD/

    Infer:
      scene_name = "071125-CD"
      scene_all  = scene_root/"071125-CD-All"
      sparse0    = scene_all/undistortion_sparse/0

    Returns:
      (sparse0_path, scene_all_path)
    """
    root = scene_root.resolve()
    if not root.exists():
        raise FileNotFoundError(f"Scene root not found: {root}")

    scene_name = root.name
    scene_all = root / f"{scene_name}-All"
    if not scene_all.exists():
        raise FileNotFoundError(f"Expected scene-All folder not found: {scene_all}")

    sparse0 = scene_all / "undistortion_sparse" / "0"
    if not sparse0.exists():
        raise FileNotFoundError(
            "Could not find COLMAP sparse model folder. Tried:\n"
            f"  {sparse0}"
        )

    return sparse0, scene_all


def default_scene_nn_out_path(scene_all: Path) -> Path:
    out_dir = scene_all / "NN"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / "nn_rank_by_name.json"


# ----------------------------
# Math utilities
# ----------------------------
def qvec_to_rotmat(q: np.ndarray) -> np.ndarray:
    """COLMAP quaternion qvec: [qw, qx, qy, qz] -> R (3x3)."""
    qw, qx, qy, qz = q.astype(np.float64)
    n = np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)

    if n == 0:
        raise ValueError("Zero-norm quaternion.")

    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n

    R = np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )
    return R


def rot_angle_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    """Geodesic angle between two rotation matrices in degrees."""
    Rrel = Ra.T @ Rb
    tr = np.clip((np.trace(Rrel) - 1.0) / 2.0, -1.0, 1.0)
    ang = np.arccos(tr)
    return float(np.degrees(ang))


def camera_center_w2c(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Camera center in world coordinates from W2C extrinsics: C = -R^T t."""
    return -R.T @ t


def robust_scene_radius(centers: np.ndarray) -> float:
    """radius = median(||Ci - median(C)||), robust scene scale."""
    c0 = np.median(centers, axis=0)
    d = np.linalg.norm(centers - c0[None, :], axis=1)
    r = float(np.median(d))
    return max(r, 1e-8)


def pose_distance_normalized(
    a: PoseW2C,
    b: PoseW2C,
    scene_radius: float,
    lambda_rot_deg: float,
) -> float:
    Ca = camera_center_w2c(a.R, a.t)
    Cb = camera_center_w2c(b.R, b.t)

    dt = np.linalg.norm(Ca - Cb) / scene_radius

    if lambda_rot_deg == 0.0:
        return float(dt)

    dr = rot_angle_deg(a.R, b.R)

    return float(dt + lambda_rot_deg * dr)


# ----------------------------
# COLMAP loading
# ----------------------------
def load_poses_from_images_txt(images_txt_path: Path) -> Dict[int, PoseW2C]:
    poses: Dict[int, PoseW2C] = {}
    lines = images_txt_path.read_text(encoding="utf-8").splitlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1

        if not line or line.startswith("#"):
            continue

        parts = line.split()

        if len(parts) < 10:
            continue

        image_id = int(parts[0])
        q = np.array(list(map(float, parts[1:5])), dtype=np.float64)
        t = np.array(list(map(float, parts[5:8])), dtype=np.float64)
        name = " ".join(parts[9:])
        R = qvec_to_rotmat(q)

        poses[image_id] = PoseW2C(
            image_id=image_id,
            name=name,
            R=R,
            t=t,
        )

        # Skip POINTS2D line.
        if i < len(lines):
            i += 1

    if not poses:
        raise RuntimeError(f"No poses parsed from {images_txt_path}")

    return poses


def load_colmap_poses(sparse0_dir: Path) -> Dict[int, PoseW2C]:
    if not sparse0_dir.exists():
        raise FileNotFoundError(f"Folder not found: {sparse0_dir}")

    images_txt = sparse0_dir / "images.txt"

    if not images_txt.exists():
        raise FileNotFoundError(
            "images.txt not found.\n"
            f"Expected: {images_txt}\n"
            "Please export the COLMAP sparse model to text format."
        )

    return load_poses_from_images_txt(images_txt)


# ----------------------------
# Scene NN ranking
# ----------------------------
def compute_scene_radius_from_poses(poses: Dict[int, PoseW2C]) -> float:
    centers = np.stack(
        [camera_center_w2c(p.R, p.t) for p in poses.values()],
        axis=0,
    )
    return robust_scene_radius(centers)


def rank_all_neighbors_by_name(
    poses: Dict[int, PoseW2C],
    scene_radius: float,
    lambda_rot_deg: float,
) -> Dict[str, List[str]]:
    ids = sorted(poses.keys())
    id2name: Dict[int, str] = {i: poses[i].name for i in ids}

    out: Dict[str, List[str]] = {}

    for qid in ids:
        qpose = poses[qid]
        dists: List[Tuple[int, float]] = []

        for cid in ids:
            if cid == qid:
                continue

            d = pose_distance_normalized(
                qpose,
                poses[cid],
                scene_radius=scene_radius,
                lambda_rot_deg=lambda_rot_deg,
            )

            dists.append((cid, d))

        dists.sort(key=lambda x: x[1])
        out[id2name[qid]] = [id2name[cid] for cid, _ in dists]

    return out


def save_scene_nn_json_by_name(nn_dict: Dict[str, List[str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(nn_dict, f, indent=2)


# ----------------------------
# Dataset traversal
# ----------------------------
def iter_scenes_train(train_root: Path) -> List[Path]:
    """
    Train structure:
      root/Train/<dataset>/<chunk>/<scene>/
    """
    scenes: List[Path] = []

    if not train_root.exists():
        return scenes

    dataset_dirs = sorted([p for p in train_root.iterdir() if p.is_dir()])

    for dataset_dir in dataset_dirs:
        chunk_dirs = sorted([p for p in dataset_dir.iterdir() if p.is_dir()])

        for chunk_dir in chunk_dirs:
            scene_dirs = sorted([p for p in chunk_dir.iterdir() if p.is_dir()])
            scenes.extend(scene_dirs)

    return scenes


def iter_scenes_val(val_root: Path) -> List[Path]:
    """
    Val structure:
      root/Val/<dataset>/<scene>/
    """
    scenes: List[Path] = []

    if not val_root.exists():
        return scenes

    dataset_dirs = sorted([p for p in val_root.iterdir() if p.is_dir()])

    for dataset_dir in dataset_dirs:
        scene_dirs = sorted([p for p in dataset_dir.iterdir() if p.is_dir()])
        scenes.extend(scene_dirs)

    return scenes


# ----------------------------
# Per-scene processing
# ----------------------------
def process_one_scene(
    scene_root: Path,
    lambda_rot_deg: float,
    overwrite: bool,
) -> str:
    sparse0, scene_all = infer_sparse0_from_scene(scene_root)
    out_path = default_scene_nn_out_path(scene_all)

    if out_path.exists() and not overwrite:
        return f"[SKIP] exists: {out_path}"

    poses = load_colmap_poses(sparse0)
    scene_radius = compute_scene_radius_from_poses(poses)

    nn_dict = rank_all_neighbors_by_name(
        poses=poses,
        scene_radius=scene_radius,
        lambda_rot_deg=lambda_rot_deg,
    )

    save_scene_nn_json_by_name(nn_dict, out_path)

    return f"[OK] {scene_root} -> {out_path} (N={len(poses)}, radius={scene_radius:.6f})"


# ----------------------------
# CLI
# ----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--root",
        type=str,
        required=True,
        help="Dataset root containing Train/ and Val/",
    )

    ap.add_argument(
        "--lambda_rot_deg",
        type=float,
        default=0.05,
        help="Rotation weight in degrees. Use 0.0 for translation-only.",
    )

    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing nn_rank_by_name.json.",
    )

    args = ap.parse_args()

    root = Path(args.root).resolve()
    train_root = root / "Train"
    val_root = root / "Val"

    scenes: List[Path] = []
    scenes.extend(iter_scenes_train(train_root))
    scenes.extend(iter_scenes_val(val_root))
    scenes = sorted(set(scenes))

    print(f"[Info] root          : {root}")
    print(f"[Info] split         : Train + Val")
    print(f"[Info] datasets      : all")
    print(f"[Info] scenes        : {len(scenes)}")
    print(f"[Info] lambda_rot_deg: {args.lambda_rot_deg}")
    print(f"[Info] overwrite     : {args.overwrite}")

    if len(scenes) == 0:
        raise RuntimeError(
            "No scenes found. Expected structure:\n"
            "  root/Train/<dataset>/<chunk>/<scene>/\n"
            "  root/Val/<dataset>/<scene>/"
        )

    ok_cnt = 0
    skip_cnt = 0

    for idx, scene_root in enumerate(scenes, 1):
        msg = process_one_scene(
            scene_root=scene_root,
            lambda_rot_deg=args.lambda_rot_deg,
            overwrite=args.overwrite,
        )

        print(f"[{idx:05d}/{len(scenes):05d}] {msg}")

        if msg.startswith("[OK]"):
            ok_cnt += 1
        elif msg.startswith("[SKIP]"):
            skip_cnt += 1

    print(f"\n[Done] OK={ok_cnt} SKIP={skip_cnt}")


if __name__ == "__main__":
    main()
