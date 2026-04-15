# planners/conflict_features.py
# Hand-crafted φ(c, n) for conflict ranking (small MLP input).

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

ConflictTuple = Tuple


def _first_goal_index(path: List[Tuple[int, int]], goal: Tuple[int, int]) -> int:
    for ti, p in enumerate(path):
        if p[0] == goal[0] and p[1] == goal[1]:
            return ti
    return len(path) - 1

FEATURE_NAMES: Tuple[str, ...] = (
    "is_vertex",
    "is_edge",
    "t_norm",
    "agent_i_norm",
    "agent_j_norm",
    "goal_sep_norm",
    "cost_i_norm",
    "cost_j_norm",
    "pair_cost_share",
    "inv_num_conflicts",
    "depth_norm",
    "soc_norm",
    "free_neighbor_norm",
)

_FEATURE_DIM = len(FEATURE_NAMES)


def conflict_to_dict(c: ConflictTuple) -> dict:
    """JSON-serializable description of a conflict (matches CBS tuple layout)."""
    if c[0] == "vertex":
        _, t, r, cc, ai, aj = c
        return {
            "kind": "vertex",
            "t": int(t),
            "r": int(r),
            "c": int(cc),
            "i": int(ai),
            "j": int(aj),
        }
    if c[0] == "edge":
        (
            _,
            t,
            pi0,
            pi1,
            ci0,
            ci1,
            pj0,
            pj1,
            cj0,
            cj1,
            ai,
            aj,
        ) = c
        return {
            "kind": "edge",
            "t": int(t),
            "i_edge": [int(pi0), int(pi1), int(ci0), int(ci1)],
            "j_edge": [int(pj0), int(pj1), int(cj0), int(cj1)],
            "i": int(ai),
            "j": int(aj),
        }
    raise ValueError(f"unknown conflict tag {c[0]!r}")


def _free_neighbor_frac(grid: np.ndarray, r: int, c: int) -> float:
    H, W = grid.shape
    nbr = 0
    free = 0
    for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        nr, nc = r + dr, c + dc
        if nr < 0 or nr >= H or nc < 0 or nc >= W:
            continue
        nbr += 1
        if grid[nr, nc] == 0:
            free += 1
    return free / max(nbr, 1)


def compute_conflict_features(
    conflict: ConflictTuple,
    paths: List[List[Tuple[int, int]]],
    goals: np.ndarray,
    grid: np.ndarray,
    *,
    node_soc: int,
    depth: int,
    num_conflicts: int,
) -> np.ndarray:
    """
    φ(c, n) ∈ R^d with d = len(FEATURE_NAMES). All values finite for valid inputs.
    """
    N = len(paths)
    goal_tuples = [(int(goals[i, 0]), int(goals[i, 1])) for i in range(N)]
    costs = [_first_goal_index(paths[i], goal_tuples[i]) for i in range(N)]
    T = max(c + 1 for c in costs) if costs else 1
    H, W = grid.shape
    diag = max(H + W, 1)
    max_sc = max(max(costs) if costs else 1, 1)
    sum_sc = max(sum(costs), 1)

    if conflict[0] == "vertex":
        _, t, r, c, ai, aj = conflict
        is_v, is_e = 1.0, 0.0
        fr, fc = int(r), int(c)
    elif conflict[0] == "edge":
        _, t, pi0, pi1, ci0, ci1, pj0, pj1, cj0, cj1, ai, aj = conflict
        is_v, is_e = 0.0, 1.0
        fr = (int(pi0) + int(ci0)) // 2
        fc = (int(pi1) + int(ci1)) // 2
    else:
        raise ValueError(f"bad conflict {conflict}")

    ai, aj = int(ai), int(aj)
    g0, g1 = goal_tuples[ai], goal_tuples[aj]
    goal_sep = abs(g0[0] - g1[0]) + abs(g0[1] - g1[1])

    ci, cj = costs[ai], costs[aj]
    feat = np.zeros(_FEATURE_DIM, dtype=np.float64)
    feat[0] = is_v
    feat[1] = is_e
    feat[2] = float(t) / float(T)
    denom_ag = max(N - 1, 1)
    feat[3] = min(ai, aj) / denom_ag
    feat[4] = max(ai, aj) / denom_ag
    feat[5] = goal_sep / float(diag)
    feat[6] = ci / float(max_sc)
    feat[7] = cj / float(max_sc)
    feat[8] = (ci + cj) / float(sum_sc)
    feat[9] = 1.0 / float(1 + max(num_conflicts, 0))
    feat[10] = float(depth) / float(depth + 10)
    # Average agent cost relative to the longest current path cost.
    feat[11] = float(node_soc) / float(max(N * max_sc, 1))
    feat[12] = _free_neighbor_frac(grid, fr, fc)
    return feat


def compute_all_conflict_features(
    conflicts: Sequence[ConflictTuple],
    paths: List[List[Tuple[int, int]]],
    goals: np.ndarray,
    grid: np.ndarray,
    *,
    node_soc: int,
    depth: int,
) -> np.ndarray:
    """Shape (len(conflicts), d)."""
    m = len(conflicts)
    if m == 0:
        return np.zeros((0, _FEATURE_DIM), dtype=np.float64)
    out = np.empty((m, _FEATURE_DIM), dtype=np.float64)
    for k, c in enumerate(conflicts):
        out[k] = compute_conflict_features(
            c,
            paths,
            goals,
            grid,
            node_soc=node_soc,
            depth=depth,
            num_conflicts=m,
        )
    return out
