# planners/cbs.py
# Conflict-Based Search (CBS) for MAPF with sum-of-costs objective, 4-connected grid.

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Tuple

import numpy as np

# Constraint tuples (per agent):
# Vertex: ("v", t, r, c) — cannot occupy (r, c) at time t
# Edge: ("e", t, r1, c1, r2, c2) — cannot move from (r1,c1) to (r2,c2) on transition ending at t
Constraint = Tuple[str, ...]

# 4-connected: WAIT, RIGHT, DOWN, UP, LEFT (matches core/env.py)
_DELTAS = [(0, 0), (0, 1), (1, 0), (-1, 0), (0, -1)]


def _manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def space_time_astar(
    grid: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    constraints: FrozenSet[Constraint],
    max_time: int,
) -> Optional[List[Tuple[int, int]]]:
    """
    Space-time A* from start to goal with vertex/edge constraints.
    Returns list of (row, col) from t=0 .. T-1 inclusive, or None if infeasible.
    """
    H, W = grid.shape
    sr, sc = int(start[0]), int(start[1])
    gr, gc = int(goal[0]), int(goal[1])

    if grid[sr, sc] == 1 or grid[gr, gc] == 1:
        return None

    def violates_vertex(t: int, r: int, c: int) -> bool:
        for cstr in constraints:
            if cstr[0] == "v":
                _, tt, tr, tc = cstr
                if tt == t and tr == r and tc == c:
                    return True
        return False

    def violates_edge(t_end: int, r1: int, c1: int, r2: int, c2: int) -> bool:
        for cstr in constraints:
            if cstr[0] == "e":
                _, tt, a, b, rr, cc = cstr
                if tt == t_end and a == r1 and b == c1 and rr == r2 and cc == c2:
                    return True
        return False

    counter = 0
    open_heap: List[Tuple[int, int, int, int, int]] = []
    g_best: Dict[Tuple[int, int, int], int] = {}
    came_from: Dict[Tuple[int, int, int], Tuple[int, int, int]] = {}

    def h(r: int, c: int) -> int:
        return _manhattan((r, c), (gr, gc))

    heapq.heappush(open_heap, (h(sr, sc), counter, sr, sc, 0))
    counter += 1
    g_best[(sr, sc, 0)] = 0

    while open_heap:
        f, _, r, c, t = heapq.heappop(open_heap)
        g = g_best.get((r, c, t))
        if g is None:
            continue
        if g + h(r, c) != f:
            continue  # stale entry

        if (r, c) == (gr, gc):
            out: List[Tuple[int, int]] = []
            cur: Tuple[int, int, int] = (r, c, t)
            while True:
                out.append((cur[0], cur[1]))
                if cur[0] == sr and cur[1] == sc and cur[2] == 0:
                    break
                cur = came_from[cur]
            out.reverse()
            return out

        if t >= max_time:
            continue

        for dr, dc in _DELTAS:
            nr, nc = r + dr, c + dc
            if nr < 0 or nr >= H or nc < 0 or nc >= W:
                continue
            if grid[nr, nc] == 1:
                continue
            t2 = t + 1
            if t2 > max_time:
                continue
            if violates_vertex(t2, nr, nc):
                continue
            if violates_edge(t2, r, c, nr, nc):
                continue

            ng = g + 1
            if g_best.get((nr, nc, t2), 10**9) <= ng:
                continue
            g_best[(nr, nc, t2)] = ng
            came_from[(nr, nc, t2)] = (r, c, t)
            nf = ng + h(nr, nc)
            heapq.heappush(open_heap, (nf, counter, nr, nc, t2))
            counter += 1

    return None


def _first_goal_index(path: List[Tuple[int, int]], goal: Tuple[int, int]) -> int:
    for ti, p in enumerate(path):
        if p[0] == goal[0] and p[1] == goal[1]:
            return ti
    return len(path) - 1


def _path_cost(path: List[Tuple[int, int]], goal: Tuple[int, int]) -> int:
    """Sum-of-costs edge count until first reaching goal."""
    return _first_goal_index(path, goal)


def _pad_paths_to_array(
    paths: List[List[Tuple[int, int]]],
    goals: np.ndarray,
) -> np.ndarray:
    """Pad to common length T with goal-wait; shape (T, N, 2)."""
    N = len(paths)
    goal_tuples = [(int(goals[i, 0]), int(goals[i, 1])) for i in range(N)]
    costs = [_first_goal_index(paths[i], goal_tuples[i]) for i in range(N)]
    T = max(c + 1 for c in costs)
    out = np.zeros((T, N, 2), dtype=np.int32)
    for i in range(N):
        g = goal_tuples[i]
        p = paths[i]
        fg = costs[i]
        for t in range(fg + 1):
            if t < len(p):
                out[t, i, 0] = p[t][0]
                out[t, i, 1] = p[t][1]
            else:
                out[t, i, 0] = g[0]
                out[t, i, 1] = g[1]
        for t in range(fg + 1, T):
            out[t, i, 0] = g[0]
            out[t, i, 1] = g[1]
    return out


def _find_conflict(
    paths: List[List[Tuple[int, int]]],
    goals: np.ndarray,
) -> Optional[Tuple]:
    """
    First conflict: ('vertex', t, r, c, ai, aj) or ('edge', t, ar, ac, br, bc, ai, aj).
    """
    N = len(paths)
    goal_tuples = [(int(goals[i, 0]), int(goals[i, 1])) for i in range(N)]
    costs = [_first_goal_index(paths[i], goal_tuples[i]) for i in range(N)]
    T = max(c + 1 for c in costs)

    def pos(i: int, t: int) -> Tuple[int, int]:
        pi = paths[i]
        g = goal_tuples[i]
        fg = costs[i]
        if t <= fg:
            if t < len(pi):
                return int(pi[t][0]), int(pi[t][1])
            return g
        return g

    for t in range(T):
        for i in range(N):
            for j in range(i + 1, N):
                a, b = pos(i, t), pos(j, t)
                if a == b:
                    return ("vertex", t, a[0], a[1], i, j)

    for t in range(1, T):
        for i in range(N):
            for j in range(i + 1, N):
                pti = pos(i, t - 1)
                ptj = pos(j, t - 1)
                cti = pos(i, t)
                ctj = pos(j, t)
                if pti == ctj and ptj == cti and pti != cti:
                    # i moves pti->cti, j moves ptj->ctj (swap)
                    return ("edge", t, pti[0], pti[1], cti[0], cti[1], ptj[0], ptj[1], ctj[0], ctj[1], i, j)

    return None


@dataclass
class _CTNode:
    cost: int
    constraints: Dict[int, FrozenSet[Constraint]]
    paths: List[List[Tuple[int, int]]]


class CBSSolver:
    """
    CBS with sum-of-costs, 4-connected grid, vertex + edge conflicts.
    """

    def __init__(
        self,
        grid: np.ndarray,
        starts: np.ndarray,
        goals: np.ndarray,
        max_time: int = 128,
    ):
        self.grid = grid
        self.starts = starts.astype(int)
        self.goals = goals.astype(int)
        self.N = starts.shape[0]
        self.max_time = max_time

    def _plan_one(
        self,
        agent: int,
        cons: FrozenSet[Constraint],
    ) -> Optional[List[Tuple[int, int]]]:
        return space_time_astar(
            self.grid,
            (int(self.starts[agent, 0]), int(self.starts[agent, 1])),
            (int(self.goals[agent, 0]), int(self.goals[agent, 1])),
            cons,
            self.max_time,
        )

    def _plan_all_independent(self) -> Optional[List[List[Tuple[int, int]]]]:
        paths: List[List[Tuple[int, int]]] = []
        for i in range(self.N):
            p = self._plan_one(i, frozenset())
            if p is None:
                return None
            paths.append(p)
        return paths

    def _soc(self, paths: List[List[Tuple[int, int]]]) -> int:
        total = 0
        for i in range(self.N):
            g = (int(self.goals[i, 0]), int(self.goals[i, 1]))
            total += _path_cost(paths[i], g)
        return total

    def solve(self) -> Optional[np.ndarray]:
        """
        Returns paths array (T, N, 2) or None if no solution within search limits.
        """
        root_paths = self._plan_all_independent()
        if root_paths is None:
            return None

        root = _CTNode(
            cost=self._soc(root_paths),
            constraints={i: frozenset() for i in range(self.N)},
            paths=[list(p) for p in root_paths],
        )

        open_heap: List[Tuple[int, int, _CTNode]] = []
        heap_counter = 0

        def push(node: _CTNode) -> None:
            nonlocal heap_counter
            heapq.heappush(open_heap, (node.cost, heap_counter, node))
            heap_counter += 1

        push(root)

        while open_heap:
            _, _, node = heapq.heappop(open_heap)

            conf = _find_conflict(node.paths, self.goals)
            if conf is None:
                return _pad_paths_to_array(node.paths, self.goals)

            if conf[0] == "vertex":
                _, t, r, c, ai, aj = conf
                for child_agent, _other in ((ai, aj), (aj, ai)):
                    vcon: Constraint = ("v", t, r, c)
                    new_cons = dict(node.constraints)
                    new_cons[child_agent] = frozenset(new_cons[child_agent]) | {vcon}
                    new_paths = [list(p) for p in node.paths]
                    replanned = self._plan_one(child_agent, new_cons[child_agent])
                    if replanned is None:
                        continue
                    new_paths[child_agent] = replanned
                    cnew = self._soc(new_paths)
                    push(_CTNode(cost=cnew, constraints=new_cons, paths=new_paths))
            else:
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
                ) = conf
                branches = [
                    (ai, ("e", t, pi0, pi1, ci0, ci1)),
                    (aj, ("e", t, pj0, pj1, cj0, cj1)),
                ]
                for child_agent, econ in branches:
                    new_cons = dict(node.constraints)
                    new_cons[child_agent] = frozenset(new_cons[child_agent]) | {econ}
                    new_paths = [list(p) for p in node.paths]
                    replanned = self._plan_one(child_agent, new_cons[child_agent])
                    if replanned is None:
                        continue
                    new_paths[child_agent] = replanned
                    cnew = self._soc(new_paths)
                    push(_CTNode(cost=cnew, constraints=new_cons, paths=new_paths))

        return None
