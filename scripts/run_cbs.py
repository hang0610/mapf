#!/usr/bin/env python3
"""
Run Conflict-Based Search (CBS) on a MovingAI map + scenario and save paths.npy.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from mapf_env.io.movingai_map import load_map
from mapf_env.io.movingai_scene import load_scen
from core.instance import instance_from_scen
from core.validate import validate_paths
from planners.cbs import CBSSolver


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

    t0 = time.perf_counter()
    solver = CBSSolver(
        grid=instance.grid,
        starts=instance.starts,
        goals=instance.goals,
        max_time=args.max_time,
    )
    paths = solver.solve()
    elapsed = time.perf_counter() - t0

    if paths is None:
        print("CBS: no solution (increase --max_time or check instance).", file=sys.stderr)
        sys.exit(1)

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out_path), paths)

    T, N, _ = paths.shape
    print(f"Saved: {out_path}  shape=({T}, {N}, 2)  time={elapsed:.3f}s")

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
