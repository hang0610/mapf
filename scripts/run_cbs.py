#!/usr/bin/env python3
"""
Run Conflict-Based Search (CBS) on a MovingAI map + scenario and save paths.npy.
"""

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional, cast

import numpy as np

from mapf_env.io.movingai_map import load_map
from mapf_env.io.movingai_scene import load_scen
from core.instance import instance_from_scen
from core.validate import validate_paths
from planners.cbs import CBSSolver, ConflictPolicy, RolloutLabelConfig
from planners.conflict_ranker import LinearConflictRanker


def default_scen_path(map_name: str, scen_dir: str) -> Path:
    scen_dir_path = Path(scen_dir)
    scen_files = list(scen_dir_path.glob(f"{map_name}-random-*.scen"))
    if scen_files:
        return scen_files[0]
    return scen_dir_path / f"{map_name}-random-1.scen"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run CBS MAPF planner; write paths.npy (T x N x 2)."
    )
    parser.add_argument(
        "--map",
        required=True,
        help="Map basename without .map (e.g. empty-32-32).",
    )
    parser.add_argument(
        "--k",
        type=int,
        required=True,
        help="Number of agents (scenario rows).",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Starting row in scenario file (default: 0).",
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
        "--map_path",
        type=str,
        default=None,
        help="Explicit path to .map (overrides --map).",
    )
    parser.add_argument(
        "--scen_path",
        type=str,
        default=None,
        help="Explicit path to .scen (overrides default glob).",
    )
    parser.add_argument(
        "--max_time",
        type=int,
        default=128,
        help="Space-time A* horizon (max timestep index); increase for hard instances.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="paths.npy",
        help="Output .npy path (default: paths.npy).",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run validate_paths after planning (same connectivity as planner: 4).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        metavar="SEC",
        help="Wall-clock limit for high-level CBS search (seconds); exit with error if exceeded.",
    )
    parser.add_argument(
        "--stats_json",
        type=str,
        default=None,
        help="Write run metrics (runtime, CT pops, children enqueued, etc.) to this JSON file.",
    )
    parser.add_argument(
        "--conflict_policy",
        type=str,
        choices=("earliest", "random", "learned"),
        default="earliest",
        help=(
            "How to pick a conflict when |C(n)|>1: earliest in time/agent order, "
            "uniform random, or a learned linear ranker."
        ),
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to a learned conflict-ranker .npz model (used by --conflict_policy learned).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed for --conflict_policy random (optional).",
    )
    parser.add_argument(
        "--log_ct",
        type=str,
        default=None,
        help="Append one JSON object per CT expansion (JSONL) for instrumentation.",
    )
    parser.add_argument(
        "--log_ct_features",
        action="store_true",
        help="With --log_ct, include conflict_features matrix (φ per conflict) in each line.",
    )
    parser.add_argument(
        "--rollout_max_pops",
        type=int,
        default=None,
        metavar="N",
        help=(
            "If set with --log_ct, run bounded CBS from each child of every candidate "
            "conflict and attach rollout_labels (expensive)."
        ),
    )
    parser.add_argument(
        "--rollout_wall_s",
        type=float,
        default=None,
        metavar="SEC",
        help="Per-child wall time for rollout labeling (optional).",
    )
    parser.add_argument(
        "--rollout_policy",
        type=str,
        choices=("earliest", "random"),
        default="earliest",
        help="Conflict policy inside each rollout subtree (labeling only).",
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
        print(f"Error: map not found: {map_path}", file=sys.stderr)
        sys.exit(1)
    if not scen_path.exists():
        print(f"Error: scenario not found: {scen_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Map path : {map_path}")
    print(f"Scen path: {scen_path}")

    grid = load_map(map_path)
    scen_starts, scen_goals = load_scen(scen_path)
    instance = instance_from_scen(
        grid=grid,
        scen_starts=scen_starts,
        scen_goals=scen_goals,
        k=args.k,
        offset=args.offset,
    )

    policy = cast(ConflictPolicy, args.conflict_policy)
    effective_policy = policy
    rng = None
    if policy == "random":
        rng = np.random.default_rng(args.seed)

    learned_ranker: Optional[LinearConflictRanker] = None
    resolved_model_path: Optional[Path] = None
    if policy == "learned":
        if args.model_path is None:
            print(
                "[WARN] --conflict_policy learned requested without --model_path; "
                "falling back to earliest.",
                file=sys.stderr,
            )
            effective_policy = cast(ConflictPolicy, "earliest")
        else:
            resolved_model_path = Path(args.model_path).expanduser().resolve()
            try:
                learned_ranker = LinearConflictRanker.load_npz(resolved_model_path)
            except Exception as exc:
                print(
                    f"[WARN] failed to load learned model from {resolved_model_path}: {exc}; "
                    "falling back to earliest.",
                    file=sys.stderr,
                )
                effective_policy = cast(ConflictPolicy, "earliest")
                learned_ranker = None

    if args.rollout_max_pops is not None and not args.log_ct:
        print(
            "Error: --rollout_max_pops requires --log_ct (JSONL sink).",
            file=sys.stderr,
        )
        sys.exit(1)

    rollout_cfg: Optional[RolloutLabelConfig] = None
    if args.rollout_max_pops is not None:
        rollout_cfg = RolloutLabelConfig(
            max_ct_pops=int(args.rollout_max_pops),
            wall_time_s=args.rollout_wall_s,
            policy=cast(ConflictPolicy, args.rollout_policy),
        )

    # Minimal fields merged into each --log_ct line (no UUID / full paths).
    log_context = {
        "map": map_path.stem,
        "k": args.k,
        "offset": args.offset,
    }

    log_fp = None
    try:
        if args.log_ct:
            log_path = Path(args.log_ct).expanduser().resolve()
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_fp = open(log_path, "w", encoding="utf-8")

        def on_ct_expand(row: dict) -> None:
            if log_fp is not None:
                log_fp.write(json.dumps(row) + "\n")

        t0 = time.perf_counter()
        solver = CBSSolver(
            grid=instance.grid,
            starts=instance.starts,
            goals=instance.goals,
            max_time=args.max_time,
            wall_time_limit_s=args.timeout,
            conflict_policy=effective_policy,
            rng=rng,
            on_ct_expand=on_ct_expand if log_fp is not None else None,
            ct_log_features=bool(log_fp is not None and args.log_ct_features),
            log_context=log_context,
            rollout_label_config=rollout_cfg,
            learned_ranker=learned_ranker,
        )
        paths = solver.solve()
        elapsed = time.perf_counter() - t0
    finally:
        if log_fp is not None:
            log_fp.close()

    if solver.last_stats is None:
        raise RuntimeError("CBSSolver.solve did not set last_stats")
    stats = solver.last_stats

    def _print_stats_line() -> None:
        print(
            f"CBS stats: success={stats.success} timed_out={stats.timed_out} "
            f"ct_pops={stats.ct_nodes_popped} ct_children={stats.ct_children_enqueued} "
            f"max_open={stats.max_open_size} soc={stats.sum_of_costs} wall_s={elapsed:.6f}"
        )
        if stats.learned_policy_calls > 0:
            avg_ms = 1000.0 * stats.learned_policy_wall_s / max(stats.learned_policy_calls, 1)
            print(
                "Learned conflict selector: "
                f"calls={stats.learned_policy_calls} "
                f"fallbacks={stats.learned_policy_fallbacks} "
                f"score_wall_s={stats.learned_policy_wall_s:.6f} "
                f"score_avg_ms={avg_ms:.3f}"
            )

    if paths is None:
        _print_stats_line()
        if stats.timed_out:
            print("CBS: wall-clock timeout.", file=sys.stderr)
        else:
            print(
                "CBS: no solution (increase --max_time or check instance).",
                file=sys.stderr,
            )
        if args.stats_json:
            payload = {
                **asdict(stats),
                "wall_clock_s": elapsed,
                "map": str(map_path),
                "scen": str(scen_path),
                "k": args.k,
                "max_time_horizon": args.max_time,
                "conflict_policy": args.conflict_policy,
                "effective_conflict_policy": effective_policy,
                "seed": args.seed,
                "model_path": str(resolved_model_path) if resolved_model_path else None,
                "log_ct": args.log_ct,
                "log_ct_features": args.log_ct_features,
                "rollout_max_pops": args.rollout_max_pops,
                "rollout_wall_s": args.rollout_wall_s,
                "rollout_policy": args.rollout_policy,
                "learned_policy_avg_ms": (
                    1000.0 * stats.learned_policy_wall_s / max(stats.learned_policy_calls, 1)
                    if stats.learned_policy_calls > 0
                    else None
                ),
            }
            p = Path(args.stats_json).expanduser().resolve()
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(f"Wrote stats: {p}")
        sys.exit(1)

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out_path), paths)

    T, N, _ = paths.shape
    print(f"Saved: {out_path}  shape=({T}, {N}, 2)  time={elapsed:.3f}s")
    _print_stats_line()
    if args.log_ct:
        print(f"CT log (JSONL): {Path(args.log_ct).expanduser().resolve()}")

    if args.stats_json:
        payload = {
            **asdict(stats),
            "wall_clock_s": elapsed,
            "map": str(map_path),
            "scen": str(scen_path),
            "k": args.k,
            "max_time_horizon": args.max_time,
            "conflict_policy": args.conflict_policy,
            "effective_conflict_policy": effective_policy,
            "seed": args.seed,
            "model_path": str(resolved_model_path) if resolved_model_path else None,
            "log_ct": args.log_ct,
            "log_ct_features": args.log_ct_features,
            "rollout_max_pops": args.rollout_max_pops,
            "rollout_wall_s": args.rollout_wall_s,
            "rollout_policy": args.rollout_policy,
            "learned_policy_avg_ms": (
                1000.0 * stats.learned_policy_wall_s / max(stats.learned_policy_calls, 1)
                if stats.learned_policy_calls > 0
                else None
            ),
        }
        p = Path(args.stats_json).expanduser().resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote stats: {p}")

    if args.validate:
        r = validate_paths(
            instance.grid,
            paths,
            starts=instance.starts,
            goals=instance.goals,
            connectivity="4",
        )
        print(f"validate_paths ok={r['ok']} success={r.get('success')}")
        if not r["ok"]:
            print(r.get("first_error"))
            sys.exit(1)


if __name__ == "__main__":
    main()
