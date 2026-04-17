# scripts/playback_paths.py

import argparse
from pathlib import Path
from typing import Optional

import numpy as np

from mapf_env.io.movingai_map import load_map
from mapf_env.io.movingai_scene import load_scen
from core.instance import instance_from_scen
from mapf_env.viz.animate import animate_paths


def default_scen_path(map_name: str, scen_dir: str) -> Path:
    scen_dir_path = Path(scen_dir)
    scen_files = list(scen_dir_path.glob(f"{map_name}-random-*.scen"))
    if scen_files:
        return scen_files[0]
    return scen_dir_path / f"{map_name}-random-1.scen"


def _filter_valid_scen_rows(
    grid: np.ndarray,
    scen_starts: np.ndarray,
    scen_goals: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    h, w = grid.shape
    starts_in_bounds = (
        (scen_starts[:, 0] >= 0)
        & (scen_starts[:, 0] < h)
        & (scen_starts[:, 1] >= 0)
        & (scen_starts[:, 1] < w)
    )
    goals_in_bounds = (
        (scen_goals[:, 0] >= 0)
        & (scen_goals[:, 0] < h)
        & (scen_goals[:, 1] >= 0)
        & (scen_goals[:, 1] < w)
    )
    valid_mask = starts_in_bounds & goals_in_bounds
    if valid_mask.any():
        valid_mask = valid_mask.copy()
        valid_idx = np.where(valid_mask)[0]
        valid_mask[valid_idx] = (
            (grid[scen_starts[valid_idx, 0], scen_starts[valid_idx, 1]] == 0)
            & (grid[scen_goals[valid_idx, 0], scen_goals[valid_idx, 1]] == 0)
        )
    return scen_starts[valid_mask], scen_goals[valid_mask]


def scenario_instance(
    grid: np.ndarray,
    scen_starts: np.ndarray,
    scen_goals: np.ndarray,
    *,
    k: int,
    offset: int,
    filter_invalid_rows: bool = False,
):
    starts = scen_starts
    goals = scen_goals
    if filter_invalid_rows:
        starts, goals = _filter_valid_scen_rows(grid, scen_starts, scen_goals)
    return instance_from_scen(
        grid=grid,
        scen_starts=starts,
        scen_goals=goals,
        k=k,
        offset=offset,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Playback MAPF paths and save as GIF."
    )
    parser.add_argument(
        "--map",
        required=True,
        help="Map basename, e.g. 'den312d' (without extension).",
    )
    parser.add_argument(
        "--paths",
        required=True,
        help="Path to paths.npy (shape T x N x 2, (row, col)).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="paths.gif",
        help="Output GIF filename (default: paths.gif).",
    )
    parser.add_argument(
        "--maps_dir",
        type=str,
        default="data/mapf-map",
        help="Directory with .map files (default: data/mapf-map).",
    )
    parser.add_argument(
        "--map_path",
        type=str,
        default=None,
        help="Explicit .map path (overrides --map & --maps_dir).",
    )
    parser.add_argument(
        "--scen_dir",
        type=str,
        default="data/scens",
        help="Directory with .scen files (default: data/scens).",
    )
    parser.add_argument(
        "--scen_path",
        type=str,
        default=None,
        help="Explicit .scen path. If not set, tries '<map>-random-*.scen' then '-random-1.scen'.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="If provided, use scenario to build starts/goals for first k agents.",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Scenario row offset (default: 0).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=5,
        help="Frames per second for the GIF (default: 5).",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Temporal downsampling factor for frames (default: 1 = use all frames).",
    )
    parser.add_argument(
        "--no_collision_highlight",
        action="store_true",
        help="Disable red collision highlighting in the GIF.",
    )
    parser.add_argument(
        "--filter_invalid_scen_rows",
        action="store_true",
        help=(
            "Drop scenario rows that are out-of-bounds or on obstacles before applying "
            "--offset/--k. Off by default so playback matches run_cbs and validate_paths."
        ),
    )

    args = parser.parse_args()

    # -------- load map --------
    if args.map_path is not None:
        map_path = Path(args.map_path)
    else:
        map_path = Path(args.maps_dir) / f"{args.map}.map"
    if not map_path.exists():
        raise FileNotFoundError(f"Map file not found: {map_path}")

    print(f"Using map : {map_path}")
    grid = load_map(map_path)
    H, W = grid.shape
    print(f"Grid shape: H={H}, W={W}")

    # -------- load paths --------
    paths_path = Path(args.paths)
    # Resolve relative paths properly
    if not paths_path.is_absolute():
        # Handle ./ prefix
        if args.paths.startswith('./'):
            paths_path = Path.cwd() / args.paths[2:]
        else:
            paths_path = Path.cwd() / paths_path
    
    # Try to resolve the path
    try:
        paths_path = paths_path.resolve()
    except (OSError, RuntimeError):
        pass  # If resolve fails, use the path as-is
    
    if not paths_path.exists():
        # Try to provide helpful suggestions
        cwd = Path.cwd()
        suggestions = ""
        
        # Check if the directory exists
        parent_dir = paths_path.parent
        if not parent_dir.exists():
            suggestions += f"\nDirectory does not exist: {parent_dir}\n"
            if parent_dir.name == "results":
                suggestions += f"  You may need to create it: mkdir -p {parent_dir}\n"
        
        # Check if there are any .npy files in the current directory or results directory
        npy_files = list(cwd.glob("*.npy"))
        results_dir = cwd / "results"
        if results_dir.exists():
            npy_files.extend(results_dir.glob("*.npy"))
        
        if npy_files:
            suggestions += f"\nFound .npy files:\n"
            for f in npy_files[:10]:  # Show up to 10 files
                rel_path = f.relative_to(cwd) if f.is_relative_to(cwd) else f
                suggestions += f"  - {rel_path}\n"
            if len(npy_files) > 10:
                suggestions += f"  ... and {len(npy_files) - 10} more\n"
        
        error_msg = (
            f"Paths file not found: {args.paths}\n"
            f"  Resolved to: {paths_path}\n"
            f"  Current directory: {cwd}\n"
            f"{suggestions}"
            f"\nPlease provide a valid path to a .npy file containing paths with shape (T, N, 2)."
        )
        raise FileNotFoundError(error_msg)

    paths = np.load(str(paths_path))
    if paths.ndim != 3 or paths.shape[2] != 2:
        raise ValueError(
            f"paths.npy must have shape (T, N, 2), got {paths.shape}"
        )
    T, N, _ = paths.shape
    print(f"Paths shape: T={T}, N={N}, 2")

    # -------- optional starts/goals from scenario --------
    starts = goals = None
    if args.k is not None:
        if args.scen_path is not None:
            scen_path = Path(args.scen_path)
        else:
            scen_path = default_scen_path(args.map, args.scen_dir)

        print(f"Using scen: {scen_path}")
        if not scen_path.exists():
            raise FileNotFoundError(f"Scenario file not found: {scen_path}")

        scen_starts, scen_goals = load_scen(scen_path)
        instance = scenario_instance(
            grid,
            scen_starts,
            scen_goals,
            k=args.k,
            offset=args.offset,
            filter_invalid_rows=bool(args.filter_invalid_scen_rows),
        )
        starts = instance.starts
        goals = instance.goals

        if N != instance.num_agents:
            print(
                f"[WARN] paths has N={N}, but scenario instance has "
                f"{instance.num_agents}; trimming to min(N, num_agents)."
            )
            min_n = min(N, instance.num_agents)
            starts = starts[:min_n]
            goals = goals[:min_n]
            paths = paths[:, :min_n, :]

    # -------- animate --------
    out_path = Path(args.out)
    print(f"Saving GIF to: {out_path}")

    animate_paths(
        grid=grid,
        paths=paths,
        starts=starts,
        goals=goals,
        out=str(out_path),
        fps=args.fps,
        stride=args.stride,
        highlight_collisions=not args.no_collision_highlight,
    )

    print("Done. GIF created.")


if __name__ == "__main__":
    main()
