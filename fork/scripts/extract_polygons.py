#!/usr/bin/python3
"""Run RoomFormer inference and save predicted polygons as JSON files.

Usage:
    python3 fork/scripts/extract_polygons.py
"""

import json
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from shapely.geometry import Polygon

from models import build_model
from eval import get_args_parser
from datasets import build_dataset

FORK_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_CONFIGS = {
    "stru3d": dict(
        dataset_root=os.path.join(FORK_ROOT, "input/stru3d"),
        eval_set="test",
        checkpoint=os.path.join(FORK_ROOT, "input/checkpoints/roomformer_stru3d.pth"),
        canonical_name="STRUCTURED_3D",
    ),
    "scenecad": dict(
        dataset_root=os.path.join(FORK_ROOT, "input/scenecad"),
        eval_set="val",
        checkpoint=os.path.join(FORK_ROOT, "input/checkpoints/roomformer_scenecad.pth"),
        canonical_name="SCENE_CAD",
    ),
}


def extract_room_polys(fg_mask_per_scene, pred_corners_per_scene):
    """Extract valid room polygons for one scene."""
    rooms = []
    for j in range(fg_mask_per_scene.shape[0]):
        fg_mask_per_room = fg_mask_per_scene[j]
        valid_corners = pred_corners_per_scene[j][fg_mask_per_room]
        if len(valid_corners) == 0:
            continue
        corners = (valid_corners * 255).cpu().numpy()
        corners = np.around(corners).astype(np.int32)
        if len(corners) >= 4 and Polygon(corners).area >= 100:
            rooms.append(corners.tolist())
    return rooms


@torch.no_grad()
def main(dataset_name: str) -> None:
    output_dir = os.path.join(FORK_ROOT, "output", dataset_name)
    dataset_cfg = DATASET_CONFIGS[dataset_name]

    # Reuse eval.py defaults for all model architecture params
    args = get_args_parser().parse_args([])
    args.dataset_name = dataset_name
    args.dataset_root = dataset_cfg["dataset_root"]
    args.eval_set = dataset_cfg["eval_set"]
    args.checkpoint = dataset_cfg["checkpoint"]

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Build model
    model = build_model(args, train=False)
    model.to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=False)
    model.eval()
    print(f"Loaded checkpoint from {args.checkpoint}")

    # Build dataset
    eval_dataset = build_dataset(image_set=args.eval_set, args=args)
    loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        sampler=torch.utils.data.SequentialSampler(eval_dataset),
        drop_last=False,
        collate_fn=lambda batch: batch,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    os.makedirs(output_dir, exist_ok=True)

    all_scene_polys: dict[str, list] = {}
    for batched_inputs in loader:
        samples = [x["image"].to(device) for x in batched_inputs]
        scene_ids = [x["image_id"] for x in batched_inputs]

        outputs = model(samples)
        pred_logits = outputs["pred_logits"]
        pred_corners = outputs["pred_coords"]
        fg_mask = torch.sigmoid(pred_logits) > 0.5

        for i in range(pred_logits.shape[0]):
            scene_id = str(scene_ids[i])
            rooms = extract_room_polys(fg_mask[i], pred_corners[i])
            all_scene_polys[scene_id] = rooms

    result = {
        "dataset": dataset_cfg["canonical_name"],
        "model": "ROOMFORMER",
        "data": all_scene_polys,
    }

    out_path = os.path.join(output_dir, "polygons.json")
    with open(out_path, "w") as f:
        json.dump(result, f)

    print(f"Done. Saved {len(all_scene_polys)} scene predictions to {out_path}")


if __name__ == "__main__":
    for dataset_name in DATASET_CONFIGS:
        main(dataset_name)
