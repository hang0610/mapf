# planners/cbs.py
# Conflict-Based Search (CBS) for MAPF with sum-of-costs objective, 4-connected grid.

from __future__ import annotations

import heapq
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, FrozenSet, List, Literal, Optional, Tuple

import numpy as np

from planners.conflict_features import (
    FEATURE_NAMES,
    compute_all_conflict_features,
    conflict_to_dict,
)
from planners.conflict_ranker import LinearConflictRanker

ConflictPolicy = Literal["earliest", "random", "learned"]

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


def enumerate_conflicts(
    paths: List[List[Tuple[int, int]]],
    goals: np.ndarray,
) -> List[Tuple]:
    """
    All conflicts in deterministic order: every vertex conflict (by t, i, j), then
    every edge (swap) conflict (by t, i, j). Same tuple layout as branching uses.
    """
    N = len(paths)
    if N < 2:
        return []
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

    out: List[Tuple] = []
    for t in range(T):
        for i in range(N):
            for j in range(i + 1, N):
                a, b = pos(i, t), pos(j, t)
                if a == b:
                    out.append(("vertex", t, a[0], a[1], i, j))

    for t in range(1, T):
        for i in range(N):
            for j in range(i + 1, N):
                pti = pos(i, t - 1)
                ptj = pos(j, t - 1)
                cti = pos(i, t)
                ctj = pos(j, t)
                if pti == ctj and ptj == cti and pti != cti:
                    out.append(
                        (
                            "edge",
                            t,
                            pti[0],
                            pti[1],
                            cti[0],
                            cti[1],
                            ptj[0],
                            ptj[1],
                            ctj[0],
                            ctj[1],
                            i,
                            j,
                        )
                    )
    return out


def choose_conflict(
    conflicts: List[Tuple],
    policy: ConflictPolicy,
    rng: Optional[np.random.Generator],
    *,
    scores: Optional[np.ndarray] = None,
) -> Tuple:
    if not conflicts:
        raise ValueError("choose_conflict: empty conflict list")
    if policy == "earliest":
        return conflicts[0]
    if policy == "random":
        r = rng if rng is not None else np.random.default_rng()
        return conflicts[int(r.integers(0, len(conflicts)))]
    if policy == "learned":
        if scores is None:
            raise ValueError("choose_conflict: learned policy requires scores")
        if len(scores) != len(conflicts):
            raise ValueError("choose_conflict: scores length mismatch")
        return conflicts[int(np.argmin(np.asarray(scores, dtype=np.float64)))]
    raise ValueError(f"unknown conflict policy {policy!r}")


def _find_conflict(
    paths: List[List[Tuple[int, int]]],
    goals: np.ndarray,
) -> Optional[Tuple]:
    """First conflict in ``enumerate_conflicts`` order (legacy helper)."""
    xs = enumerate_conflicts(paths, goals)
    return xs[0] if xs else None


@dataclass
class _CTNode:
    node_id: int
    parent_id: Optional[int]
    depth: int
    cost: int
    constraints: Dict[int, FrozenSet[Constraint]]
    paths: List[List[Tuple[int, int]]]


@dataclass
class RolloutLabelConfig:
    """
    Bounded CBS continuation from each child of a conflict split (Week 5–6 labeling).

    For every candidate conflict c at a node, we build CBS children, then run a
    capped high-level search from each child. Pops and solve flags become targets.
    """

    max_ct_pops: int
    wall_time_s: Optional[float] = None
    policy: ConflictPolicy = "earliest"


@dataclass
class CBSSolveStats:
    """
    High-level CBS run metrics (cheap observability; not per-conflict logging).

    - ct_nodes_popped: CT nodes removed from the open list and expanded.
    - ct_children_enqueued: child CT nodes successfully pushed after low-level replans.
    - max_open_size: peak high-level open-list size.
    - sum_of_costs: SOC at the goal node when success; else None.
    """

    success: bool
    timed_out: bool
    ct_nodes_popped: int
    ct_children_enqueued: int
    max_open_size: int
    sum_of_costs: Optional[int]
    learned_policy_calls: int = 0
    learned_policy_fallbacks: int = 0
    learned_policy_wall_s: float = 0.0


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
        wall_time_limit_s: Optional[float] = None,
        *,
        conflict_policy: ConflictPolicy = "earliest",
        rng: Optional[np.random.Generator] = None,
        on_ct_expand: Optional[Callable[[dict], None]] = None,
        ct_log_features: bool = False,
        log_context: Optional[Dict[str, Any]] = None,
        rollout_label_config: Optional[RolloutLabelConfig] = None,
        learned_ranker: Optional[LinearConflictRanker] = None,
    ):
        self.grid = grid
        self.starts = starts.astype(int)
        self.goals = goals.astype(int)
        self.N = starts.shape[0]
        self.max_time = max_time
        self.wall_time_limit_s = wall_time_limit_s
        self.last_stats: Optional[CBSSolveStats] = None
        self.conflict_policy = conflict_policy
        self.rng = rng
        self.on_ct_expand = on_ct_expand
        self.ct_log_features = ct_log_features
        self.log_context = dict(log_context) if log_context else {}
        self.rollout_label_config = rollout_label_config
        self.learned_ranker = learned_ranker
        self._learned_policy_calls = 0
        self._learned_policy_fallbacks = 0
        self._learned_policy_wall_s = 0.0

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

    def _compute_conflict_features(
        self,
        node: _CTNode,
        conflicts: List[Tuple],
    ) -> np.ndarray:
        return compute_all_conflict_features(
            conflicts,
            node.paths,
            self.goals,
            self.grid,
            node_soc=node.cost,
            depth=node.depth,
        )

    def _choose_conflict(
        self,
        node: _CTNode,
        conflicts: List[Tuple],
        conflict_features: Optional[np.ndarray] = None,
    ) -> Tuple[Tuple, Optional[np.ndarray]]:
        if self.conflict_policy != "learned":
            return choose_conflict(conflicts, self.conflict_policy, self.rng), conflict_features

        self._learned_policy_calls += 1
        if len(conflicts) <= 1 or self.learned_ranker is None:
            self._learned_policy_fallbacks += 1
            return conflicts[0], conflict_features

        t0 = time.perf_counter()
        try:
            phi = conflict_features
            if phi is None:
                phi = self._compute_conflict_features(node, conflicts)
            scores = self.learned_ranker.score_features(phi, FEATURE_NAMES)
            if not np.all(np.isfinite(scores)):
                raise ValueError("non-finite learned conflict scores")
            chosen = choose_conflict(conflicts, "learned", self.rng, scores=scores)
            return chosen, phi
        except Exception:
            self._learned_policy_fallbacks += 1
            return conflicts[0], conflict_features
        finally:
            self._learned_policy_wall_s += time.perf_counter() - t0

    def _branch_children(
        self,
        parent: _CTNode,
        conflict: Tuple,
        *,
        start_nid: int,
    ) -> Tuple[List[_CTNode], int]:
        """CBS split on ``conflict``; returns (children, next free node id)."""
        children: List[_CTNode] = []
        nid = start_nid
        if conflict[0] == "vertex":
            _, t, r, c, ai, aj = conflict
            for child_agent, _other in ((ai, aj), (aj, ai)):
                vcon: Constraint = ("v", t, r, c)
                new_cons = dict(parent.constraints)
                new_cons[child_agent] = frozenset(new_cons[child_agent]) | {vcon}
                new_paths = [list(p) for p in parent.paths]
                replanned = self._plan_one(child_agent, new_cons[child_agent])
                if replanned is None:
                    continue
                new_paths[child_agent] = replanned
                cnew = self._soc(new_paths)
                children.append(
                    _CTNode(
                        node_id=nid,
                        parent_id=parent.node_id,
                        depth=parent.depth + 1,
                        cost=cnew,
                        constraints=new_cons,
                        paths=new_paths,
                    )
                )
                nid += 1
        elif conflict[0] == "edge":
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
            ) = conflict
            branches = [
                (ai, ("e", t, pi0, pi1, ci0, ci1)),
                (aj, ("e", t, pj0, pj1, cj0, cj1)),
            ]
            for child_agent, econ in branches:
                new_cons = dict(parent.constraints)
                new_cons[child_agent] = frozenset(new_cons[child_agent]) | {econ}
                new_paths = [list(p) for p in parent.paths]
                replanned = self._plan_one(child_agent, new_cons[child_agent])
                if replanned is None:
                    continue
                new_paths[child_agent] = replanned
                cnew = self._soc(new_paths)
                children.append(
                    _CTNode(
                        node_id=nid,
                        parent_id=parent.node_id,
                        depth=parent.depth + 1,
                        cost=cnew,
                        constraints=new_cons,
                        paths=new_paths,
                    )
                )
                nid += 1
        else:
            raise ValueError(f"unknown conflict tag {conflict[0]!r}")
        return children, nid

    def _limited_cbs_from_node(
        self,
        entry: _CTNode,
        *,
        max_ct_pops: int,
        deadline: Optional[float],
        policy: ConflictPolicy,
        rng: Optional[np.random.Generator],
    ) -> Tuple[bool, int, Optional[int], bool]:
        """
        Continue CBS from a single CT node with pop and optional wall budget.

        Returns (solved, pops_used, soc_if_solved, timed_out).
        """
        next_free = entry.node_id + 1
        open_heap: List[Tuple[int, int, _CTNode]] = []
        heap_counter = 0
        heapq.heappush(open_heap, (entry.cost, heap_counter, entry))
        heap_counter += 1
        pops = 0
        while open_heap:
            if deadline is not None and time.monotonic() >= deadline:
                return False, pops, None, True
            if pops >= max_ct_pops:
                return False, pops, None, False
            _, _, node = heapq.heappop(open_heap)
            pops += 1
            conflicts = enumerate_conflicts(node.paths, self.goals)
            if not conflicts:
                return True, pops, self._soc(node.paths), False
            conf = choose_conflict(conflicts, policy, rng)
            new_children, next_free = self._branch_children(
                node, conf, start_nid=next_free
            )
            for ch in new_children:
                heapq.heappush(open_heap, (ch.cost, heap_counter, ch))
                heap_counter += 1
        return False, pops, None, False

    def _rollout_labels_for_conflicts(
        self,
        parent: _CTNode,
        conflicts: List[Tuple],
    ) -> List[dict]:
        rc = self.rollout_label_config
        assert rc is not None
        maxp = rc.max_ct_pops
        labels: List[dict] = []
        for c in conflicts:
            children, _ = self._branch_children(parent, c, start_nid=1)
            sides: List[dict] = []
            for ch in children:
                dl: Optional[float] = None
                if rc.wall_time_s is not None:
                    dl = time.monotonic() + float(rc.wall_time_s)
                solved, pops, soc, timed_out = self._limited_cbs_from_node(
                    ch,
                    max_ct_pops=maxp,
                    deadline=dl,
                    policy=rc.policy,
                    rng=None,
                )
                sides.append(
                    {
                        "solved": solved,
                        "pops": int(pops),
                        "soc": soc,
                        "timed_out": timed_out,
                    }
                )
            while len(sides) < 2:
                sides.append(
                    {
                        "solved": False,
                        "pops": maxp,
                        "soc": None,
                        "timed_out": False,
                        "missing_child": True,
                    }
                )
            a, b = sides[0], sides[1]
            pa = int(a["pops"])
            pb = int(b["pops"])
            label_sum = pa + pb
            if a["solved"] and b["solved"]:
                label_min = min(pa, pb)
            elif a["solved"]:
                label_min = pa
            elif b["solved"]:
                label_min = pb
            else:
                label_min = None
            labels.append(
                {
                    "conflict": conflict_to_dict(c),
                    "side_a": a,
                    "side_b": b,
                    "label_sum_pops": label_sum,
                    "label_min_pops": label_min,
                    "max_ct_pops_budget": maxp,
                }
            )
        return labels

    def _emit_ct_expand(
        self,
        node: _CTNode,
        conflicts: List[Tuple],
        chosen: Tuple,
        ct_pop_index: int,
        rollout_labels: Optional[List[dict]] = None,
        conflict_features: Optional[np.ndarray] = None,
    ) -> None:
        if self.on_ct_expand is None:
            return
        payload: dict = {
            **self.log_context,
            "schema_version": "ct_expand/2" if rollout_labels else "ct_expand/1",
            "event": "ct_expand",
            "node_id": node.node_id,
            "parent_id": node.parent_id,
            "depth": node.depth,
            "soc": node.cost,
            "ct_pop_index": ct_pop_index,
            "num_agents": self.N,
            "conflict_policy": self.conflict_policy,
            "conflicts": [conflict_to_dict(c) for c in conflicts],
            "chosen": conflict_to_dict(chosen),
            "constraint_counts": {str(i): len(node.constraints[i]) for i in range(self.N)},
        }
        if self.ct_log_features:
            phi = conflict_features
            if phi is None:
                phi = self._compute_conflict_features(node, conflicts)
            payload["feature_names"] = list(FEATURE_NAMES)
            payload["conflict_features"] = [row.tolist() for row in phi]
        if rollout_labels is not None:
            payload["rollout_labels"] = rollout_labels
        self.on_ct_expand(payload)

    def solve(self) -> Optional[np.ndarray]:
        """
        Returns paths array (T, N, 2) or None if no solution within search limits.
        After the call, see ``last_stats`` for CT expansions, timeouts, etc.
        """
        empty_stats = CBSSolveStats(
            success=False,
            timed_out=False,
            ct_nodes_popped=0,
            ct_children_enqueued=0,
            max_open_size=0,
            sum_of_costs=None,
            learned_policy_calls=self._learned_policy_calls,
            learned_policy_fallbacks=self._learned_policy_fallbacks,
            learned_policy_wall_s=self._learned_policy_wall_s,
        )

        root_paths = self._plan_all_independent()
        if root_paths is None:
            self.last_stats = empty_stats
            return None

        root = _CTNode(
            node_id=0,
            parent_id=None,
            depth=0,
            cost=self._soc(root_paths),
            constraints={i: frozenset() for i in range(self.N)},
            paths=[list(p) for p in root_paths],
        )
        next_node_id = 1

        open_heap: List[Tuple[int, int, _CTNode]] = []
        heap_counter = 0
        ct_nodes_popped = 0
        ct_children_enqueued = 0
        max_open_size = 0
        deadline: Optional[float] = None
        if self.wall_time_limit_s is not None:
            deadline = time.monotonic() + float(self.wall_time_limit_s)

        def push(node: _CTNode) -> None:
            nonlocal heap_counter, max_open_size
            heapq.heappush(open_heap, (node.cost, heap_counter, node))
            heap_counter += 1
            max_open_size = max(max_open_size, len(open_heap))

        push(root)

        while open_heap:
            if deadline is not None and time.monotonic() >= deadline:
                self.last_stats = CBSSolveStats(
                    success=False,
                    timed_out=True,
                    ct_nodes_popped=ct_nodes_popped,
                    ct_children_enqueued=ct_children_enqueued,
                    max_open_size=max_open_size,
                    sum_of_costs=None,
                    learned_policy_calls=self._learned_policy_calls,
                    learned_policy_fallbacks=self._learned_policy_fallbacks,
                    learned_policy_wall_s=self._learned_policy_wall_s,
                )
                return None

            _, _, node = heapq.heappop(open_heap)
            ct_nodes_popped += 1

            conflicts = enumerate_conflicts(node.paths, self.goals)
            if not conflicts:
                soc = self._soc(node.paths)
                self.last_stats = CBSSolveStats(
                    success=True,
                    timed_out=False,
                    ct_nodes_popped=ct_nodes_popped,
                    ct_children_enqueued=ct_children_enqueued,
                    max_open_size=max_open_size,
                    sum_of_costs=soc,
                    learned_policy_calls=self._learned_policy_calls,
                    learned_policy_fallbacks=self._learned_policy_fallbacks,
                    learned_policy_wall_s=self._learned_policy_wall_s,
                )
                return _pad_paths_to_array(node.paths, self.goals)

            conflict_features: Optional[np.ndarray] = None
            if self.ct_log_features:
                conflict_features = self._compute_conflict_features(node, conflicts)
            conf, conflict_features = self._choose_conflict(
                node,
                conflicts,
                conflict_features=conflict_features,
            )
            rollout_labels: Optional[List[dict]] = None
            if (
                self.rollout_label_config is not None
                and self.on_ct_expand is not None
            ):
                rollout_labels = self._rollout_labels_for_conflicts(node, conflicts)
            self._emit_ct_expand(
                node,
                conflicts,
                conf,
                ct_nodes_popped,
                rollout_labels=rollout_labels,
                conflict_features=conflict_features,
            )

            new_children, next_node_id = self._branch_children(
                node, conf, start_nid=next_node_id
            )
            for ch in new_children:
                push(ch)
                ct_children_enqueued += 1

        self.last_stats = CBSSolveStats(
            success=False,
            timed_out=False,
            ct_nodes_popped=ct_nodes_popped,
            ct_children_enqueued=ct_children_enqueued,
            max_open_size=max_open_size,
            sum_of_costs=None,
            learned_policy_calls=self._learned_policy_calls,
            learned_policy_fallbacks=self._learned_policy_fallbacks,
            learned_policy_wall_s=self._learned_policy_wall_s,
        )
        return None
