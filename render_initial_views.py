"""Render the four established demo views at t=0 without stepping the scene."""
from __future__ import annotations

import argparse
import json
from importlib.metadata import version
from pathlib import Path

from scene import HERE, build, positions, read_config
from initial_views import render_initial_views


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "config.json")
    parser.add_argument("--output", type=Path, default=HERE / "deliverables/initial_views")
    parser.add_argument("--reference-rest", type=Path,
                        help="Optional saved rest.npz to check initial soft geometry")
    args = parser.parse_args()
    config = read_config(args.config)
    if config.get("initial_robot_targets"):
        raise ValueError("This renderer requires the scene's default home robot poses")
    if any(args.output.glob("initial_*")):
        raise FileExistsError("Choose a fresh output directory; existing images are preserved")
    scene, arms, bag, bear, cameras = build(config, viewer=False, initial_views=True)
    reference = None
    if args.reference_rest:
        import numpy as np
        with np.load(args.reference_rest, allow_pickle=False) as saved:
            errors = {}
            for name, entity in (("bag", bag), ("bear", bear)):
                current = positions(entity)
                if current.shape != saved[name].shape:
                    raise RuntimeError(f"{name} geometry differs from the saved initial state")
                errors[name] = float(np.max(np.abs(current - saved[name])))
                if errors[name] > 1e-10:
                    raise RuntimeError(f"{name} geometry differs from the saved initial state")
            reference = {"reference_file": args.reference_rest.name,
                         "maximum_vertex_coordinate_error_m": errors}
    metadata = render_initial_views(scene, cameras, args.output, config=config)
    import torch
    metadata.update(
        camera_preset_source="Same four canonical-world camera presets as the prior cloth/plastic bag demos",
        genesis_version=version("genesis-world"),
        genesis_backend="cuda",
        gpu_device=torch.cuda.get_device_name(0),
        robot_initial_dofs={side: arm.get_dofs_position().detach().cpu().numpy().tolist()
                            for side, arm in arms.items()},
    )
    if reference is not None:
        metadata["saved_initial_geometry_comparison"] = reference
    (args.output / "initial_views.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()),
                      "images": [item["file"] for item in metadata["images"]],
                      "simulation_time_s": float(scene.sim.cur_t)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
