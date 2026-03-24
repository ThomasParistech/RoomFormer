#!/usr/bin/python3
"""Run RoomFormer evaluation and extract predicted polygons as JSON files.

Calls the original eval pipeline (eval.main -> engine.evaluate_floor),
passing a dict that evaluate_floor populates with per-scene room polygons.

Usage:
    python3 fork/scripts/extract_polygons.py
"""

import json
import os

from eval import get_args_parser, main as eval_main

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


def main(dataset_name: str) -> None:
    output_dir = os.path.join(FORK_ROOT, "output", dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    dataset_cfg = DATASET_CONFIGS[dataset_name]

    args = get_args_parser().parse_args([])
    args.dataset_name = dataset_name
    args.dataset_root = dataset_cfg["dataset_root"]
    args.eval_set = dataset_cfg["eval_set"]
    args.checkpoint = dataset_cfg["checkpoint"]
    args.output_dir = output_dir
    args.plot_pred = False
    args.plot_density = False
    args.plot_gt = False

    # evaluate_floor will populate this dict with {scene_id: [[x,y],...]}
    scene_polys: dict[str, list] = {}
    args.scene_polys_out = scene_polys

    eval_main(args)

    out_path = os.path.join(output_dir, "polygons.json")
    with open(out_path, "w") as f:
        json.dump({"dataset": dataset_cfg["canonical_name"], "model": "ROOMFORMER", "data": scene_polys}, f)
    print(f"Saved {len(scene_polys)} scene predictions to {out_path}")


if __name__ == "__main__":
    for dataset_name in DATASET_CONFIGS:
        main(dataset_name)
