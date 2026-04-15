from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class NodeRankingExample:
    """Conflict-ranking training data for a single CT node expansion."""

    features: np.ndarray
    labels: np.ndarray
    feature_names: Tuple[str, ...]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PairwiseRankingDataset:
    """Pairwise ranking examples derived from node-level supervision."""

    pair_features: np.ndarray
    sample_weights: np.ndarray
    num_pairs: int
    num_nodes: int
    skipped_ties: int


@dataclass
class LinearConflictRanker:
    """
    Small linear scorer for conflict selection.

    The runtime path only needs standardization and a dot product:
    score(c) = w^T z(c) + b, where lower is better.
    """

    feature_names: Tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    bias: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def score_features(
        self,
        features: np.ndarray,
        feature_names: Sequence[str],
    ) -> np.ndarray:
        x = np.asarray(features, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError(f"features must be 2D, got shape {x.shape}")
        idx = _feature_indices(feature_names, self.feature_names)
        x_sel = x[:, idx]
        z = (x_sel - self.mean) / self.scale
        return z @ self.weights + float(self.bias)

    def save_npz(self, path: Path) -> None:
        payload = {
            "feature_names": np.asarray(self.feature_names, dtype="<U64"),
            "mean": self.mean.astype(np.float64),
            "scale": self.scale.astype(np.float64),
            "weights": self.weights.astype(np.float64),
            "bias": np.asarray([self.bias], dtype=np.float64),
            "metadata_json": np.asarray(
                [json.dumps(self.metadata, sort_keys=True)], dtype="<U32768"
            ),
        }
        np.savez(path, **payload)

    @classmethod
    def load_npz(cls, path: Path) -> "LinearConflictRanker":
        data = np.load(path, allow_pickle=False)
        metadata_json = data["metadata_json"][0]
        metadata = json.loads(str(metadata_json)) if str(metadata_json) else {}
        return cls(
            feature_names=tuple(str(x) for x in data["feature_names"].tolist()),
            mean=np.asarray(data["mean"], dtype=np.float64),
            scale=np.asarray(data["scale"], dtype=np.float64),
            weights=np.asarray(data["weights"], dtype=np.float64),
            bias=float(np.asarray(data["bias"], dtype=np.float64).reshape(-1)[0]),
            metadata=metadata,
        )


def summarize_rollout_label(label: Dict[str, Any]) -> Dict[str, Any]:
    budget = int(label["max_ct_pops_budget"])
    left_effort, left_censored = _side_effort(label["side_a"], budget)
    right_effort, right_censored = _side_effort(label["side_b"], budget)

    solved_efforts: List[int] = []
    if bool(label["side_a"].get("solved", False)):
        solved_efforts.append(left_effort)
    if bool(label["side_b"].get("solved", False)):
        solved_efforts.append(right_effort)

    return {
        "effort_left": left_effort,
        "effort_right": right_effort,
        "effort_sum": left_effort + right_effort,
        "effort_min": min(solved_efforts) if solved_efforts else None,
        "left_censored": left_censored,
        "right_censored": right_censored,
        "censored": left_censored or right_censored,
        "budget": budget,
    }


def load_node_examples(
    jsonl_paths: Iterable[Path],
    *,
    label_key: str = "effort_sum",
    drop_feature_names: Sequence[str] = (),
) -> List[NodeRankingExample]:
    examples: List[NodeRankingExample] = []
    for path in jsonl_paths:
        with Path(path).open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                ex = _row_to_example(
                    row,
                    label_key=label_key,
                    drop_feature_names=drop_feature_names,
                )
                if ex is not None:
                    examples.append(ex)
    return examples


def compute_normalization(
    examples: Sequence[NodeRankingExample],
) -> Tuple[np.ndarray, np.ndarray]:
    if not examples:
        raise ValueError("cannot normalize empty dataset")
    all_features = np.concatenate([ex.features for ex in examples], axis=0)
    mean = all_features.mean(axis=0)
    scale = all_features.std(axis=0)
    scale = np.where(scale < 1e-8, 1.0, scale)
    return mean.astype(np.float64), scale.astype(np.float64)


def standardize_examples(
    examples: Sequence[NodeRankingExample],
    mean: np.ndarray,
    scale: np.ndarray,
) -> List[NodeRankingExample]:
    out: List[NodeRankingExample] = []
    for ex in examples:
        z = (ex.features - mean) / scale
        out.append(
            NodeRankingExample(
                features=z,
                labels=ex.labels.copy(),
                feature_names=ex.feature_names,
                metadata=dict(ex.metadata),
            )
        )
    return out


def build_pairwise_dataset(
    examples: Sequence[NodeRankingExample],
    *,
    delta: float,
) -> PairwiseRankingDataset:
    pair_rows: List[np.ndarray] = []
    pair_weights: List[float] = []
    skipped_ties = 0

    for ex in examples:
        if ex.features.shape[0] < 2:
            continue
        for i in range(ex.features.shape[0]):
            li = float(ex.labels[i])
            if not np.isfinite(li):
                continue
            for j in range(i + 1, ex.features.shape[0]):
                lj = float(ex.labels[j])
                if not np.isfinite(lj):
                    continue
                if li + delta < lj:
                    better, worse = i, j
                    gap = lj - li
                elif lj + delta < li:
                    better, worse = j, i
                    gap = li - lj
                else:
                    skipped_ties += 1
                    continue
                diff = ex.features[worse] - ex.features[better]
                denom = max(abs(float(ex.labels[worse])), abs(float(ex.labels[better])), 1.0)
                pair_rows.append(diff.astype(np.float64))
                pair_weights.append(float(gap / denom))

    if pair_rows:
        x = np.vstack(pair_rows)
        w = np.asarray(pair_weights, dtype=np.float64)
    else:
        x = np.zeros((0, 0), dtype=np.float64)
        w = np.zeros((0,), dtype=np.float64)
    return PairwiseRankingDataset(
        pair_features=x,
        sample_weights=w,
        num_pairs=len(pair_rows),
        num_nodes=len(examples),
        skipped_ties=skipped_ties,
    )


def fit_linear_ranker(
    examples: Sequence[NodeRankingExample],
    *,
    c: float = 1.0,
    delta: float = 0.5,
    max_iter: int = 200,
    random_seed: int = 0,
) -> Tuple[LinearConflictRanker, Dict[str, Any]]:
    if not examples:
        raise ValueError("no training examples found")

    feature_names = examples[0].feature_names
    for ex in examples[1:]:
        if ex.feature_names != feature_names:
            raise ValueError("feature_names do not match across examples")

    mean, scale = compute_normalization(examples)
    standardized = standardize_examples(examples, mean, scale)
    pair_ds = build_pairwise_dataset(standardized, delta=delta)
    if pair_ds.num_pairs == 0:
        raise ValueError("no informative pairwise ranking examples found")

    weights, bias, backend = _fit_pairwise_hinge(
        pair_ds.pair_features,
        pair_ds.sample_weights,
        c=c,
        max_iter=max_iter,
        random_seed=random_seed,
    )
    ranker = LinearConflictRanker(
        feature_names=feature_names,
        mean=mean,
        scale=scale,
        weights=weights,
        bias=bias,
        metadata={
            "backend": backend,
            "c": c,
            "delta": delta,
            "max_iter": max_iter,
            "random_seed": random_seed,
        },
    )
    metrics = evaluate_ranker(ranker, examples, delta=delta)
    summary = {
        **metrics,
        "num_pairs": pair_ds.num_pairs,
        "num_nodes": pair_ds.num_nodes,
        "skipped_ties": pair_ds.skipped_ties,
        "backend": backend,
    }
    return ranker, summary


def evaluate_ranker(
    ranker: LinearConflictRanker,
    examples: Sequence[NodeRankingExample],
    *,
    delta: float,
) -> Dict[str, Any]:
    weighted_correct = 0.0
    weighted_total = 0.0
    top1_correct = 0
    top1_total = 0
    for ex in examples:
        if ex.features.shape[0] == 0:
            continue
        scores = ranker.score_features(ex.features, ex.feature_names)
        best_idx = int(np.argmin(scores))
        label_min = float(np.nanmin(ex.labels))
        if np.isfinite(label_min):
            top1_total += 1
            if float(ex.labels[best_idx]) <= label_min + delta:
                top1_correct += 1

        for i in range(ex.features.shape[0]):
            li = float(ex.labels[i])
            if not np.isfinite(li):
                continue
            for j in range(i + 1, ex.features.shape[0]):
                lj = float(ex.labels[j])
                if not np.isfinite(lj):
                    continue
                gap = abs(li - lj)
                if gap <= delta:
                    continue
                denom = max(abs(li), abs(lj), 1.0)
                pair_weight = gap / denom
                pred_better = i if scores[i] <= scores[j] else j
                true_better = i if li < lj else j
                weighted_total += pair_weight
                if pred_better == true_better:
                    weighted_correct += pair_weight

    return {
        "pairwise_accuracy": (
            weighted_correct / weighted_total if weighted_total > 0.0 else None
        ),
        "top1_accuracy": top1_correct / top1_total if top1_total > 0 else None,
        "eval_nodes": top1_total,
        "eval_pair_weight": weighted_total,
    }


def _row_to_example(
    row: Dict[str, Any],
    *,
    label_key: str,
    drop_feature_names: Sequence[str],
) -> Optional[NodeRankingExample]:
    feature_names = tuple(str(x) for x in row.get("feature_names", []))
    feature_rows = row.get("conflict_features")
    rollout_labels = row.get("rollout_labels")
    if not feature_names or feature_rows is None or rollout_labels is None:
        return None

    features = np.asarray(feature_rows, dtype=np.float64)
    if features.ndim != 2 or features.shape[0] != len(rollout_labels):
        return None

    selected_idx = [
        i for i, name in enumerate(feature_names) if name not in set(drop_feature_names)
    ]
    if not selected_idx:
        raise ValueError("all features were dropped")

    labels: List[float] = []
    for rollout_label in rollout_labels:
        summary = summarize_rollout_label(rollout_label)
        value = summary.get(label_key)
        labels.append(np.nan if value is None else float(value))

    return NodeRankingExample(
        features=features[:, selected_idx],
        labels=np.asarray(labels, dtype=np.float64),
        feature_names=tuple(feature_names[i] for i in selected_idx),
        metadata={
            "node_id": row.get("node_id"),
            "map": row.get("map"),
            "k": row.get("k"),
            "offset": row.get("offset"),
        },
    )


def _feature_indices(
    raw_feature_names: Sequence[str],
    wanted_feature_names: Sequence[str],
) -> List[int]:
    raw_idx = {name: i for i, name in enumerate(raw_feature_names)}
    idx: List[int] = []
    for name in wanted_feature_names:
        if name not in raw_idx:
            raise ValueError(f"missing feature {name!r} in runtime feature matrix")
        idx.append(raw_idx[name])
    return idx


def _fit_pairwise_hinge(
    x: np.ndarray,
    sample_weights: np.ndarray,
    *,
    c: float,
    max_iter: int,
    random_seed: int,
) -> Tuple[np.ndarray, float, str]:
    if x.ndim != 2:
        raise ValueError(f"pair_features must be 2D, got {x.shape}")
    if x.shape[0] == 0:
        raise ValueError("cannot train with zero pairs")

    try:
        from sklearn.svm import LinearSVC
    except ImportError:
        return _fit_pairwise_hinge_numpy(
            x,
            sample_weights,
            c=c,
            max_iter=max_iter,
            random_seed=random_seed,
        )

    clf = LinearSVC(
        C=float(c),
        fit_intercept=True,
        loss="hinge",
        dual=True,
        random_state=int(random_seed),
        max_iter=max(1000, int(max_iter) * 10),
    )
    x_train = np.vstack([x, -x])
    y = np.concatenate(
        [
            np.ones(x.shape[0], dtype=np.int32),
            -np.ones(x.shape[0], dtype=np.int32),
        ]
    )
    weights = np.concatenate([sample_weights, sample_weights])
    clf.fit(x_train, y, sample_weight=weights)
    return (
        clf.coef_.reshape(-1).astype(np.float64),
        float(clf.intercept_.reshape(-1)[0]),
        "sklearn_linear_svc",
    )


def _fit_pairwise_hinge_numpy(
    x: np.ndarray,
    sample_weights: np.ndarray,
    *,
    c: float,
    max_iter: int,
    random_seed: int,
) -> Tuple[np.ndarray, float, str]:
    rng = np.random.default_rng(int(random_seed))
    weights = np.zeros(x.shape[1], dtype=np.float64)
    bias = 0.0
    norm_weights = sample_weights / max(float(sample_weights.sum()), 1e-8)

    for step in range(max(50, int(max_iter))):
        order = rng.permutation(x.shape[0])
        xb = x[order]
        wb = norm_weights[order]
        margins = xb @ weights + bias
        active = margins < 1.0
        grad_w = weights.copy()
        grad_b = 0.0
        if np.any(active):
            weighted_x = xb[active] * wb[active, None]
            grad_w -= float(c) * weighted_x.sum(axis=0)
            grad_b -= float(c) * wb[active].sum()
        lr = 0.2 / np.sqrt(step + 1.0)
        weights -= lr * grad_w
        bias -= lr * grad_b

    return weights.astype(np.float64), float(bias), "numpy_batch_hinge"


def _side_effort(side: Dict[str, Any], budget: int) -> Tuple[int, bool]:
    solved = bool(side.get("solved", False))
    timed_out = bool(side.get("timed_out", False))
    missing_child = bool(side.get("missing_child", False))
    censored = (not solved) or timed_out or missing_child
    pops = int(side.get("pops", budget))
    return (budget if censored else pops), censored
