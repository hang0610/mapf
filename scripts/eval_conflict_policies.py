#!/usr/bin/env python3
"""
Batch-evaluate CBS conflict-selection policies on a held-out slice of instances.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast

import numpy as np

from core.instance import MAPFInstance, instance_from_scen
from core.validate import validate_paths
from mapf_env.io.movingai_map import load_map
from mapf_env.io.movingai_scene import load_scen
from mapf_env.viz.animate import animate_paths
from planners.cbs import CBSSolver, ConflictPolicy
from planners.conflict_ranker import MLPConflictRanker


def default_scen_path(map_name: str, scen_dir: str) -> Path:
    scen_dir_path = Path(scen_dir)
    scen_files = list(scen_dir_path.glob(f"{map_name}-random-*.scen"))
    if scen_files:
        return scen_files[0]
    return scen_dir_path / f"{map_name}-random-1.scen"


def _resolve_paths(args: argparse.Namespace) -> Tuple[Path, Path]:
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
    return map_path, scen_path


def _build_instance(
    grid: np.ndarray,
    scen_starts: np.ndarray,
    scen_goals: np.ndarray,
    *,
    k: int,
    offset: int,
) -> MAPFInstance:
    return instance_from_scen(
        grid=grid,
        scen_starts=scen_starts,
        scen_goals=scen_goals,
        k=k,
        offset=offset,
    )


def _run_policy(
    instance: MAPFInstance,
    *,
    policy: ConflictPolicy,
    max_time: int,
    timeout: Optional[float],
    rng: Optional[np.random.Generator],
    learned_ranker: Optional[MLPConflictRanker],
) -> Tuple[Optional[np.ndarray], Any, float]:
    t0 = time.perf_counter()
    solver = CBSSolver(
        grid=instance.grid,
        starts=instance.starts,
        goals=instance.goals,
        max_time=max_time,
        wall_time_limit_s=timeout,
        conflict_policy=policy,
        rng=rng,
        learned_ranker=learned_ranker,
    )
    paths = solver.solve()
    elapsed = time.perf_counter() - t0
    if solver.last_stats is None:
        raise RuntimeError("CBSSolver.solve did not set last_stats")
    return paths, solver.last_stats, elapsed


def _policy_rng(policy: ConflictPolicy, seed: Optional[int], offset: int) -> Optional[np.random.Generator]:
    if policy != "random":
        return None
    base_seed = 0 if seed is None else int(seed)
    return np.random.default_rng(base_seed + int(offset))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _summary_field_order(rows: Sequence[Dict[str, Any]]) -> List[str]:
    preferred = [
        "map",
        "scen",
        "k",
        "offset",
        "policy",
        "effective_conflict_policy",
        "success",
        "timed_out",
        "validate_ok",
        "validate_success",
        "ct_nodes_popped",
        "ct_children_enqueued",
        "sum_of_costs",
        "wall_clock_s",
        "learned_policy_calls",
        "learned_policy_fallbacks",
        "learned_policy_avg_ms",
        "paths_path",
        "stats_path",
        "gif_path",
    ]
    extra = sorted({key for row in rows for key in row.keys() if key not in preferred})
    return preferred + extra


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch-run CBS conflict policies on held-out instances and summarize results."
    )
    parser.add_argument("--map", required=True, help="Map basename without .map.")
    parser.add_argument("--k", type=int, required=True, help="Number of agents per instance.")
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
        "--map_path",
        type=str,
        default=None,
        help="Explicit path to .map (overrides --map).",
    )
    parser.add_argument(
        "--scen_path",
        type=str,
        default=None,
        help="Explicit path to .scen (overrides default lookup).",
    )
    parser.add_argument(
        "--offset_start",
        type=int,
        default=0,
        help="First scenario row offset to evaluate.",
    )
    parser.add_argument(
        "--offset_step",
        type=int,
        default=None,
        help="Offset stride between held-out instances (default: k).",
    )
    parser.add_argument(
        "--num_instances",
        type=int,
        required=True,
        help="Number of held-out instances to evaluate.",
    )
    parser.add_argument(
        "--policies",
        type=str,
        nargs="+",
        default=("earliest", "learned"),
        choices=("earliest", "random", "learned"),
        help="Conflict policies to evaluate on each held-out instance.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to a learned .npz ranker (required if policies include learned).",
    )
    parser.add_argument(
        "--max_time",
        type=int,
        default=128,
        help="Space-time A* horizon.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        metavar="SEC",
        help="Wall-clock timeout for each CBS solve.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base RNG seed when evaluating the random policy.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Directory for per-instance outputs and summary tables.",
    )
    parser.add_argument(
        "--make_gif_for_offsets",
        type=int,
        nargs="*",
        default=[],
        help="Optional list of offsets for which to save policy GIFs.",
    )
    parser.add_argument(
        "--gif_fps",
        type=int,
        default=5,
        help="GIF frames per second when --make_gif_for_offsets is used.",
    )
    parser.add_argument(
        "--gif_stride",
        type=int,
        default=1,
        help="Temporal downsampling for generated GIFs.",
    )
    parser.add_argument(
        "--no_collision_highlight",
        action="store_true",
        help="Disable collision highlighting in generated GIFs.",
    )
    args = parser.parse_args()

    if args.num_instances <= 0:
        raise ValueError("--num_instances must be positive")

    offset_step = int(args.offset_step) if args.offset_step is not None else int(args.k)
    if offset_step <= 0:
        raise ValueError("--offset_step must be positive")

    requested_policies = [cast(ConflictPolicy, p) for p in args.policies]
    learned_ranker: Optional[MLPConflictRanker] = None
    model_path: Optional[Path] = None
    if "learned" in requested_policies:
        if args.model_path is None:
            raise ValueError("--model_path is required when evaluating the learned policy")
        model_path = Path(args.model_path).expanduser().resolve()
        learned_ranker = MLPConflictRanker.load_npz(model_path)

    map_path, scen_path = _resolve_paths(args)
    print(f"Map path : {map_path}")
    print(f"Scen path: {scen_path}")

    grid = load_map(map_path)
    scen_starts, scen_goals = load_scen(scen_path)
    max_needed = args.offset_start + (args.num_instances - 1) * offset_step + args.k
    if max_needed > scen_starts.shape[0]:
        raise ValueError(
            f"Requested up to scenario row {max_needed}, but scenario only has {scen_starts.shape[0]} rows."
        )

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    gif_offsets = set(int(x) for x in args.make_gif_for_offsets)
    rows: List[Dict[str, Any]] = []

    for inst_idx in range(args.num_instances):
        offset = args.offset_start + inst_idx * offset_step
        instance = _build_instance(
            grid,
            scen_starts,
            scen_goals,
            k=args.k,
            offset=offset,
        )
        print(f"[instance {inst_idx + 1}/{args.num_instances}] offset={offset}")

        for policy in requested_policies:
            policy_dir = out_dir / policy
            policy_dir.mkdir(parents=True, exist_ok=True)
            prefix = f"offset_{offset:06d}"
            paths_out = policy_dir / f"{prefix}_paths.npy"
            stats_out = policy_dir / f"{prefix}_stats.json"
            gif_out = policy_dir / f"{prefix}.gif"
            saved_paths_path: Optional[Path] = None
            saved_gif_path: Optional[Path] = None

            rng = _policy_rng(policy, args.seed, offset)
            policy_ranker = learned_ranker if policy == "learned" else None
            paths, stats, elapsed = _run_policy(
                instance,
                policy=policy,
                max_time=args.max_time,
                timeout=args.timeout,
                rng=rng,
                learned_ranker=policy_ranker,
            )

            validate_result: Optional[Dict[str, Any]] = None
            if paths is not None:
                np.save(str(paths_out), paths)
                saved_paths_path = paths_out
                validate_result = validate_paths(
                    instance.grid,
                    paths,
                    starts=instance.starts,
                    goals=instance.goals,
                    connectivity="4",
                )
                if offset in gif_offsets:
                    animate_paths(
                        grid=instance.grid,
                        paths=paths,
                        starts=instance.starts,
                        goals=instance.goals,
                        out=str(gif_out),
                        fps=args.gif_fps,
                        stride=args.gif_stride,
                        highlight_collisions=not args.no_collision_highlight,
                    )
                    saved_gif_path = gif_out

            learned_avg_ms = (
                1000.0 * stats.learned_policy_wall_s / max(stats.learned_policy_calls, 1)
                if stats.learned_policy_calls > 0
                else None
            )
            payload: Dict[str, Any] = {
                **asdict(stats),
                "wall_clock_s": elapsed,
                "map": str(map_path),
                "scen": str(scen_path),
                "k": args.k,
                "offset": offset,
                "max_time_horizon": args.max_time,
                "timeout_s": args.timeout,
                "policy": policy,
                "effective_conflict_policy": policy,
                "seed": args.seed if policy == "random" else None,
                "model_path": str(model_path) if policy == "learned" and model_path else None,
                "learned_policy_avg_ms": learned_avg_ms,
                "paths_path": str(saved_paths_path) if saved_paths_path is not None else None,
                "gif_path": str(saved_gif_path) if saved_gif_path is not None else None,
                "validate_ok": validate_result["ok"] if validate_result is not None else None,
                "validate_success": validate_result["success"] if validate_result is not None else None,
                "validate_first_error": (
                    validate_result["first_error"] if validate_result is not None else None
                ),
            }
            _write_json(stats_out, payload)
            payload["stats_path"] = str(stats_out)
            rows.append(payload)
            print(
                f"  {policy}: success={stats.success} timed_out={stats.timed_out} "
                f"ct_pops={stats.ct_nodes_popped} wall_s={elapsed:.3f}"
            )

    summary_jsonl = out_dir / "summary.jsonl"
    with summary_jsonl.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")

    summary_csv = out_dir / "summary.csv"
    fieldnames = _summary_field_order(rows)
    with summary_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Wrote summary: {summary_jsonl}")
    print(f"Wrote summary: {summary_csv}")


if __name__ == "__main__":
    main()
