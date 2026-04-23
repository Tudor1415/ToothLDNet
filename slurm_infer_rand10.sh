#!/bin/bash
#SBATCH --job-name=toothld_infer_rand10
#SBATCH -A wbw@v100
#SBATCH -C v100-32g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=01:00:00
#SBATCH --export=ALL
#SBATCH --output=slurm_logs/toothld_infer_rand10_%j.out
#SBATCH --error=slurm_logs/toothld_infer_rand10_%j.err

set -euo pipefail
set -x

module purge
module load cuda/11.7.1

REPO_DIR="${REPO_DIR:-$WORK/ToothLDNet}"
cd "$REPO_DIR"

export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID

VENV_PATH="${VENV_PATH:-$REPO_DIR/landmark_env/bin/activate}"
if [ ! -f "$VENV_PATH" ]; then
  echo "ERROR: virtualenv not found at $VENV_PATH"
  exit 1
fi

# shellcheck disable=SC1090
source "$VENV_PATH"

PYTHON_BIN="$(command -v python)"
echo "Using python: $PYTHON_BIN"
"$PYTHON_BIN" --version

DATA_ROOT="${DATA_ROOT:-$SCRATCH/seg_data}"
NUM_CASES="${NUM_CASES:-10}"
OUT_DIR="${OUT_DIR:-$REPO_DIR/outputs/inference_${SLURM_JOB_ID:-local}}"
GNN_CKPT="${GNN_CKPT:-$REPO_DIR/inference/runs/all_tooth/version_0/checkpoints/best1.ckpt}"
LAND_CKPT="${LAND_CKPT:-$REPO_DIR/inference/runs/tooth_landmark/version_0/checkpoints/best3.ckpt}"

if [ ! -d "$DATA_ROOT" ]; then
  echo "ERROR: data root not found: $DATA_ROOT"
  exit 1
fi

if [ ! -f "$GNN_CKPT" ]; then
  echo "ERROR: GNN checkpoint not found: $GNN_CKPT"
  exit 1
fi

if [ ! -f "$LAND_CKPT" ]; then
  echo "ERROR: landmark checkpoint not found: $LAND_CKPT"
  exit 1
fi

mkdir -p "$OUT_DIR" "$REPO_DIR/slurm_logs"

export REPO_DIR DATA_ROOT NUM_CASES OUT_DIR GNN_CKPT LAND_CKPT
"$PYTHON_BIN" - <<'PY'
import csv
import os
import random
import sys
from pathlib import Path

repo_dir = Path(os.environ["REPO_DIR"]).resolve()
sys.path.insert(0, str(repo_dir / "inference"))

import numpy as np
import torch
import trimesh

from pl_model_land import LitModel
from utils.gnn.predictor import gnn_run
from data.land.common import calc_features
from scripts.simplify import Process
from scripts.seg_to_single import segment_patch_box
from scripts.knn import knn_map

data_root = Path(os.environ["DATA_ROOT"]).expanduser().resolve()
num_cases = int(os.environ["NUM_CASES"])
out_dir = Path(os.environ["OUT_DIR"]).expanduser().resolve()
gnn_ckpt = Path(os.environ["GNN_CKPT"]).expanduser().resolve()
land_ckpt = Path(os.environ["LAND_CKPT"]).expanduser().resolve()


def find_candidates(root: Path):
    patterns = ("*_lower.obj", "*.obj")
    seen = set()
    picked = []
    for pattern in patterns:
        for path in sorted(root.rglob(pattern)):
            if path in seen:
                continue
            seen.add(path)
            picked.append(path)
    return picked


def run_landmarks(mesh, mesh_sim, patches, case_name, land_model):
    args = land_model.hparams.args
    pts_all = np.empty((0, 3))
    labels_all = np.array([], dtype=int)

    for patch in patches:
        if patch.is_empty:
            continue

        vs, fs = patch.vertices, patch.faces
        vs_offset = vs.mean(0)
        vs = vs - vs_offset
        _, fids = trimesh.sample.sample_surface_even(patch, args.num_points)
        vs = torch.tensor(vs, dtype=torch.float32)
        fs = torch.tensor(fs[fids], dtype=torch.long)
        features1 = calc_features(vs, fs).unsqueeze(0).cuda()
        vs_offset_t = torch.tensor(vs_offset, dtype=torch.float32).unsqueeze(0).cuda()

        vs_sim = torch.tensor(mesh_sim.vertices, dtype=torch.float32)
        fs_sim = torch.tensor(mesh_sim.faces, dtype=torch.long)
        features2 = calc_features(vs_sim, fs_sim).unsqueeze(0).cuda()

        with torch.no_grad():
            pts, p_labels = land_model.infer(features1, features2, vs_offset_t)
            pts = pts.cpu().numpy() + vs_offset
            p_labels = p_labels.cpu().numpy()
            keep = np.nonzero(p_labels)
            pts = pts[keep]
            p_labels = p_labels[keep]

        if len(pts):
            pts_all = np.concatenate((pts_all, pts), axis=0)
            labels_all = np.concatenate((labels_all, p_labels), axis=0)

    label_names = {
        1: "Mesial",
        2: "Distal",
        3: "InnerPoint",
        4: "OuterPoint",
        5: "FacialPoint",
        6: "Cusp",
    }
    label_colors = {
        1: (255, 0, 0, 255),
        2: (0, 255, 0, 255),
        3: (255, 255, 0, 255),
        4: (0, 255, 255, 255),
        5: (255, 0, 255, 255),
        6: (0, 0, 255, 255),
    }

    pred_pts = []
    for pt, label in zip(pts_all, labels_all):
        sphere = trimesh.primitives.Sphere(radius=0.7, center=pt).to_mesh()
        sphere.visual.vertex_colors = label_colors.get(int(label), (255, 255, 255, 255))
        pred_pts.append(sphere)

    vertex_colors = np.array([[0.7, 0.7, 0.7] for _ in range(len(mesh.vertices))])
    mesh.visual.vertex_colors = vertex_colors

    scene = trimesh.Scene(pred_pts + [mesh])
    mesh_out = out_dir / f"{case_name}_landmarks.obj"
    csv_out = out_dir / f"{case_name}_landmarks.csv"
    scene.export(mesh_out)

    with csv_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["key", "coord_x", "coord_y", "coord_z", "class", "score"])
        for pt, label in zip(pts_all, labels_all):
            writer.writerow(
                [
                    case_name,
                    float(pt[0]),
                    float(pt[1]),
                    float(pt[2]),
                    label_names.get(int(label), f"Label{int(label)}"),
                    0.8,
                ]
            )

    return mesh_out, csv_out


candidates = find_candidates(data_root)
if not candidates:
    raise RuntimeError(f"No .obj files found under {data_root}")
if len(candidates) < num_cases:
    raise RuntimeError(
        f"Requested {num_cases} cases but found only {len(candidates)} .obj files under {data_root}"
    )

selected = random.sample(candidates, num_cases)
print(f"Selected {len(selected)} random cases from {data_root}:")
for case in selected:
    print(case)

land_model = LitModel.load_from_checkpoint(str(land_ckpt)).cuda()
land_model.eval()

for case_path in selected:
    case_path = case_path.resolve()
    case_name = case_path.stem
    print(f"[INFO] Running inference for {case_name}: {case_path}")

    mesh = trimesh.load(case_path)

    target_faces = 10000
    sim_path = out_dir / f"{case_name}_sim.obj"
    Process(str(case_path), target_faces, str(sim_path))

    mesh_sim = trimesh.load(sim_path)
    sim_path.unlink(missing_ok=True)

    pred_sim, mesh_sim = gnn_run(mesh_sim, str(gnn_ckpt))

    patches = []
    for labels_sim in pred_sim:
        pred = knn_map(mesh_sim, mesh, labels_sim)
        pred = np.array(pred.flatten().tolist())
        patches.append(segment_patch_box(mesh, pred))

    mesh_out, csv_out = run_landmarks(mesh, mesh_sim, patches, case_name, land_model)
    print(f"[INFO] Wrote {mesh_out}")
    print(f"[INFO] Wrote {csv_out}")

print(f"[INFO] Finished. Outputs are under {out_dir}")
PY
