#!/usr/bin/env python3
"""
Batch-collect CT expansion logs with rollout labels for learned conflict selection.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional, cast

from core.instance import instance_from_scen
from mapf_env.io.movingai_map import load_map
from mapf_env.io.movingai_scene import load_scen
from planners.cbs import CBSSolver, ConflictPolicy, RolloutLabelConfig


def default_scen_path(map_name: str, scen_dir: str) -> Path:
    scen_dir_path = Path(scen_dir)
    scen_files = list(scen_dir_path.glob(f"{map_name}-random-*.scen"))
    if scen_files:
        return scen_files[0]
    return scen_dir_path / f"{map_name}-random-1.scen"


def _append_jsonl(fp, row: dict) -> None:
    fp.write(json.dumps(row) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect CBS conflict-ranking data into train/val/test JSONL shards."
    )
    parser.add_argument("--map", required=True, help="Map basename without .map.")
    parser.add_argument("--k", type=int, required=True, help="Number of agents.")
    parser.add_argument(
        "--map_path",
        type=str,
        default=None,
        help="Explicit path to .map (overrides --map).",
    )
    parser.add_argument(
        "--scen_path",
        type=str,
        default=None,
        help="Explicit path to .scen (overrides automatic lookup).",
    )
    parser.add_argument(
        "--maps_dir",
        type=str,
        default="data/mapf-map",
        help="Directory containing .map files.",
    )
    parser.add_argument(
        "--scen_dir",
        type=str,
        default="data/scens",
        help="Directory containing .scen files.",
    )
    parser.add_argument(
        "--offset_start",
        type=int,
        default=0,
        help="Scenario row offset for the first instance.",
    )
    parser.add_argument(
        "--offset_step",
        type=int,
        default=None,
        help="Scenario row stride between instances (default: k, for disjoint windows).",
    )
    parser.add_argument(
        "--train_instances",
        type=int,
        default=0,
        help="Number of training instances to collect.",
    )
    parser.add_argument(
        "--val_instances",
        type=int,
        default=0,
        help="Number of validation instances to collect.",
    )
    parser.add_argument(
        "--test_instances",
        type=int,
        default=0,
        help="Number of test instances to collect.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Output directory for train/val/test JSONL shards and manifest.",
    )
    parser.add_argument(
        "--max_time",
        type=int,
        default=128,
        help="Space-time A* horizon (max timestep index).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        metavar="SEC",
        help="Wall-clock limit for each CBS solve.",
    )
    parser.add_argument(
        "--conflict_policy",
        choices=("earliest", "random"),
        default="earliest",
        help="Conflict policy used while collecting training traces.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed for random conflict policy.",
    )
    parser.add_argument(
        "--rollout_max_pops",
        type=int,
        required=True,
        metavar="N",
        help="Per-child pop budget for rollout effort labels.",
    )
    parser.add_argument(
        "--rollout_wall_s",
        type=float,
        default=None,
        metavar="SEC",
        help="Per-child wall-clock budget for rollout labeling.",
    )
    parser.add_argument(
        "--rollout_policy",
        choices=("earliest", "random"),
        default="earliest",
        help="Conflict policy inside rollout subtrees.",
    )
    args = parser.parse_args()

    if args.map_path is not None:
        map_path = Path(args.map_path)
    else:
        map_path = Path(args.maps_dir) / f"{args.map}.map"
    if args.scen_path is not None:
        scen_path = Path(args.scen_path)
    else:
        scen_path = default_scen_path(args.map, args.scen_dir)

    if not map_path.exists():
        raise FileNotFoundError(f"map not found: {map_path}")
    if not scen_path.exists():
        raise FileNotFoundError(f"scenario not found: {scen_path}")

    total_instances = args.train_instances + args.val_instances + args.test_instances
    if total_instances <= 0:
        raise ValueError("at least one of train_instances/val_instances/test_instances must be > 0")

    offset_step = int(args.offset_step) if args.offset_step is not None else int(args.k)
    if offset_step <= 0:
        raise ValueError("offset_step must be positive")

    print(f"Map path : {map_path}")
    print(f"Scen path: {scen_path}")
    print(f"Collecting {total_instances} instances with k={args.k}, offset_step={offset_step}")

    grid = load_map(map_path)
    scen_starts, scen_goals = load_scen(scen_path)
    max_needed = args.offset_start + (total_instances - 1) * offset_step + args.k
    if max_needed > scen_starts.shape[0]:
        raise ValueError(
            f"Requested up to scenario row {max_needed}, but scenario only has {scen_starts.shape[0]} rows."
        )

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    split_counts = {
        "train": int(args.train_instances),
        "val": int(args.val_instances),
        "test": int(args.test_instances),
    }
    split_paths = {
        split: out_dir / f"{split}.jsonl" for split, count in split_counts.items() if count > 0
    }
    manifest: Dict[str, object] = {
        "map": str(map_path),
        "scen": str(scen_path),
        "k": int(args.k),
        "offset_start": int(args.offset_start),
        "offset_step": offset_step,
        "max_time_horizon": int(args.max_time),
        "timeout_s": args.timeout,
        "conflict_policy": args.conflict_policy,
        "seed": args.seed,
        "rollout_max_pops": int(args.rollout_max_pops),
        "rollout_wall_s": args.rollout_wall_s,
        "rollout_policy": args.rollout_policy,
        "splits": {},
    }

    policy = cast(ConflictPolicy, args.conflict_policy)
    rng = None
    if policy == "random":
        import numpy as np

        rng = np.random.default_rng(args.seed)

    rollout_cfg = RolloutLabelConfig(
        max_ct_pops=int(args.rollout_max_pops),
        wall_time_s=args.rollout_wall_s,
        policy=cast(ConflictPolicy, args.rollout_policy),
    )

    global_idx = 0
    for split, count in split_counts.items():
        if count <= 0:
            continue
        split_summary = {
            "path": str(split_paths[split]),
            "instances": count,
            "solved": 0,
            "timed_out": 0,
            "ct_nodes_popped": 0,
            "ct_children_enqueued": 0,
        }
        with split_paths[split].open("w", encoding="utf-8") as fp:
            for local_idx in range(count):
                offset = args.offset_start + global_idx * offset_step
                global_idx += 1
                instance = instance_from_scen(
                    grid=grid,
                    scen_starts=scen_starts,
                    scen_goals=scen_goals,
                    k=args.k,
                    offset=offset,
                )
                log_context = {
                    "map": map_path.stem,
                    "k": args.k,
                    "offset": offset,
                    "split": split,
                    "instance_index": global_idx - 1,
                }

                solver = CBSSolver(
                    grid=instance.grid,
                    starts=instance.starts,
                    goals=instance.goals,
                    max_time=args.max_time,
                    wall_time_limit_s=args.timeout,
                    conflict_policy=policy,
                    rng=rng,
                    on_ct_expand=lambda row, fp=fp: _append_jsonl(fp, row),
                    ct_log_features=True,
                    log_context=log_context,
                    rollout_label_config=rollout_cfg,
                )
                paths = solver.solve()
                if solver.last_stats is None:
                    raise RuntimeError("CBSSolver.solve did not set last_stats")
                stats = solver.last_stats
                split_summary["ct_nodes_popped"] += int(stats.ct_nodes_popped)
                split_summary["ct_children_enqueued"] += int(stats.ct_children_enqueued)
                if paths is not None:
                    split_summary["solved"] += 1
                if stats.timed_out:
                    split_summary["timed_out"] += 1

                print(
                    f"[{split} {local_idx + 1}/{count}] offset={offset} "
                    f"success={stats.success} timed_out={stats.timed_out} "
                    f"ct_pops={stats.ct_nodes_popped}"
                )

        manifest["splits"][split] = split_summary

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote manifest: {manifest_path}")
    for split, summary in manifest["splits"].items():
        print(
            f"{split}: instances={summary['instances']} solved={summary['solved']} "
            f"timed_out={summary['timed_out']} path={summary['path']}"
        )


if __name__ == "__main__":
    main()
