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
    """Pairwise training examples derived from node-level supervision."""

    better_features: np.ndarray
    worse_features: np.ndarray
    sample_weights: np.ndarray
    num_pairs: int
    num_nodes: int
    skipped_ties: int


@dataclass
class MLPConflictRanker:
    """
    Small two-hidden-layer MLP scorer for conflict selection.

    Runtime path:
    - standardize features
    - 13 -> 16 -> 8 -> 1 with ReLU hidden activations
    - choose the conflict with the smallest score
    """

    feature_names: Tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray
    w3: np.ndarray
    b3: np.ndarray
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
        return self._score_standardized(z)

    def _score_standardized(self, x: np.ndarray) -> np.ndarray:
        h1 = np.maximum(x @ self.w1 + self.b1, 0.0)
        h2 = np.maximum(h1 @ self.w2 + self.b2, 0.0)
        scores = h2 @ self.w3 + self.b3
        return scores.reshape(-1).astype(np.float64)

    def save_npz(self, path: Path) -> None:
        payload = {
            "feature_names": np.asarray(self.feature_names, dtype="<U64"),
            "mean": self.mean.astype(np.float64),
            "scale": self.scale.astype(np.float64),
            "w1": self.w1.astype(np.float64),
            "b1": self.b1.astype(np.float64),
            "w2": self.w2.astype(np.float64),
            "b2": self.b2.astype(np.float64),
            "w3": self.w3.astype(np.float64),
            "b3": self.b3.astype(np.float64),
            "metadata_json": np.asarray(
                [json.dumps(self.metadata, sort_keys=True)], dtype="<U32768"
            ),
        }
        np.savez(path, **payload)

    @classmethod
    def load_npz(cls, path: Path) -> "MLPConflictRanker":
        data = np.load(path, allow_pickle=False)
        metadata_json = data["metadata_json"][0]
        metadata = json.loads(str(metadata_json)) if str(metadata_json) else {}
        model_type = metadata.get("model_type")
        if model_type not in (None, "mlp_2layer"):
            raise ValueError(f"unsupported model_type {model_type!r}")
        ranker = cls(
            feature_names=tuple(str(x) for x in data["feature_names"].tolist()),
            mean=np.asarray(data["mean"], dtype=np.float64),
            scale=np.asarray(data["scale"], dtype=np.float64),
            w1=np.asarray(data["w1"], dtype=np.float64),
            b1=np.asarray(data["b1"], dtype=np.float64),
            w2=np.asarray(data["w2"], dtype=np.float64),
            b2=np.asarray(data["b2"], dtype=np.float64),
            w3=np.asarray(data["w3"], dtype=np.float64),
            b3=np.asarray(data["b3"], dtype=np.float64),
            metadata=metadata,
        )
        ranker._validate_shapes()
        return ranker

    def _validate_shapes(self) -> None:
        d = len(self.feature_names)
        if self.mean.shape != (d,) or self.scale.shape != (d,):
            raise ValueError("mean/scale shapes do not match feature count")
        if self.w1.shape[0] != d or self.b1.shape != (self.w1.shape[1],):
            raise ValueError("bad first-layer parameter shapes")
        if self.w2.shape != (self.w1.shape[1], self.b2.shape[0]):
            raise ValueError("bad second-layer parameter shapes")
        if self.w3.shape != (self.b2.shape[0], 1) or self.b3.shape != (1,):
            raise ValueError("bad output-layer parameter shapes")


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
    better_rows: List[np.ndarray] = []
    worse_rows: List[np.ndarray] = []
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
                denom = max(abs(float(ex.labels[worse])), abs(float(ex.labels[better])), 1.0)
                better_rows.append(ex.features[better].astype(np.float64))
                worse_rows.append(ex.features[worse].astype(np.float64))
                pair_weights.append(float(gap / denom))

    if better_rows:
        better_features = np.vstack(better_rows)
        worse_features = np.vstack(worse_rows)
        sample_weights = np.asarray(pair_weights, dtype=np.float64)
    else:
        better_features = np.zeros((0, 0), dtype=np.float64)
        worse_features = np.zeros((0, 0), dtype=np.float64)
        sample_weights = np.zeros((0,), dtype=np.float64)
    return PairwiseRankingDataset(
        better_features=better_features,
        worse_features=worse_features,
        sample_weights=sample_weights,
        num_pairs=len(better_rows),
        num_nodes=len(examples),
        skipped_ties=skipped_ties,
    )


def fit_mlp_ranker(
    train_examples: Sequence[NodeRankingExample],
    *,
    val_examples: Optional[Sequence[NodeRankingExample]] = None,
    hidden_dims: Tuple[int, int] = (16, 8),
    epochs: int = 200,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 15,
    delta: float = 0.5,
    random_seed: int = 0,
) -> Tuple[MLPConflictRanker, Dict[str, Any]]:
    if not train_examples:
        raise ValueError("no training examples found")

    feature_names = train_examples[0].feature_names
    for ex in train_examples[1:]:
        if ex.feature_names != feature_names:
            raise ValueError("feature_names do not match across training examples")
    if val_examples is not None:
        for ex in val_examples:
            if ex.feature_names != feature_names:
                raise ValueError("feature_names do not match between train and val examples")

    mean, scale = compute_normalization(train_examples)
    train_std = standardize_examples(train_examples, mean, scale)
    val_std = (
        standardize_examples(val_examples, mean, scale) if val_examples is not None else None
    )
    train_pairs = build_pairwise_dataset(train_std, delta=delta)
    if train_pairs.num_pairs == 0:
        raise ValueError("no informative pairwise ranking examples found")
    val_pairs = build_pairwise_dataset(val_std, delta=delta) if val_std is not None else None

    params = _init_mlp_params(
        input_dim=len(feature_names),
        hidden_dims=hidden_dims,
        random_seed=random_seed,
    )
    best_params = _copy_params(params)
    optimizer = _AdamState.from_params(params)

    best_metric = np.inf
    best_epoch = 0
    best_train_loss = np.inf
    best_val_loss: Optional[float] = None
    no_improve = 0
    rng = np.random.default_rng(int(random_seed))

    for epoch in range(1, max(int(epochs), 1) + 1):
        order = rng.permutation(train_pairs.num_pairs)
        for start in range(0, train_pairs.num_pairs, max(int(batch_size), 1)):
            idx = order[start : start + max(int(batch_size), 1)]
            batch = PairwiseRankingDataset(
                better_features=train_pairs.better_features[idx],
                worse_features=train_pairs.worse_features[idx],
                sample_weights=train_pairs.sample_weights[idx],
                num_pairs=len(idx),
                num_nodes=train_pairs.num_nodes,
                skipped_ties=train_pairs.skipped_ties,
            )
            _, grads = _pairwise_logistic_loss_and_grads(
                params,
                batch,
                weight_decay=weight_decay,
            )
            optimizer.step(params, grads, lr=float(learning_rate))

        train_loss = _pairwise_logistic_loss(params, train_pairs, weight_decay=weight_decay)
        if val_pairs is not None and val_pairs.num_pairs > 0:
            val_loss = _pairwise_logistic_loss(params, val_pairs, weight_decay=0.0)
            monitor = val_loss
        else:
            val_loss = None
            monitor = train_loss

        if monitor + 1e-8 < best_metric:
            best_metric = monitor
            best_epoch = epoch
            best_train_loss = train_loss
            best_val_loss = val_loss
            best_params = _copy_params(params)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= max(int(patience), 1):
                break

    ranker = MLPConflictRanker(
        feature_names=feature_names,
        mean=mean,
        scale=scale,
        w1=best_params["w1"],
        b1=best_params["b1"],
        w2=best_params["w2"],
        b2=best_params["b2"],
        w3=best_params["w3"],
        b3=best_params["b3"],
        metadata={
            "model_type": "mlp_2layer",
            "hidden_dims": list(hidden_dims),
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "learning_rate": float(learning_rate),
            "weight_decay": float(weight_decay),
            "early_stopping_patience": int(patience),
            "random_seed": int(random_seed),
            "delta": float(delta),
        },
    )

    metrics = evaluate_ranker(ranker, train_examples, delta=delta)
    summary = {
        **metrics,
        "model_type": "mlp_2layer",
        "hidden_dims": list(hidden_dims),
        "num_pairs": train_pairs.num_pairs,
        "num_nodes": train_pairs.num_nodes,
        "skipped_ties": train_pairs.skipped_ties,
        "best_epoch": best_epoch,
        "epochs_ran": min(max(int(epochs), 1), best_epoch + no_improve),
        "pairwise_loss": best_train_loss,
        "val_pairwise_loss": best_val_loss,
    }
    return ranker, summary


def evaluate_ranker(
    ranker: MLPConflictRanker,
    examples: Sequence[NodeRankingExample],
    *,
    delta: float,
) -> Dict[str, Any]:
    weighted_correct = 0.0
    weighted_total = 0.0
    top1_correct = 0
    top1_total = 0
    standardized_examples = standardize_examples(examples, ranker.mean, ranker.scale)
    pairwise_loss = compute_pairwise_loss(ranker, standardized_examples, delta=delta)

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
        "pairwise_loss": pairwise_loss,
    }


def compute_pairwise_loss(
    ranker: MLPConflictRanker,
    examples: Sequence[NodeRankingExample],
    *,
    delta: float,
) -> Optional[float]:
    standardized_examples = standardize_examples(examples, ranker.mean, ranker.scale)
    pair_ds = build_pairwise_dataset(standardized_examples, delta=delta)
    if pair_ds.num_pairs == 0:
        return None
    return _pairwise_logistic_loss(
        {
            "w1": ranker.w1,
            "b1": ranker.b1,
            "w2": ranker.w2,
            "b2": ranker.b2,
            "w3": ranker.w3,
            "b3": ranker.b3,
        },
        pair_ds,
        weight_decay=0.0,
    )


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

    drop_set = set(drop_feature_names)
    selected_idx = [i for i, name in enumerate(feature_names) if name not in drop_set]
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


def _init_mlp_params(
    *,
    input_dim: int,
    hidden_dims: Tuple[int, int],
    random_seed: int,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(int(random_seed))
    h1, h2 = int(hidden_dims[0]), int(hidden_dims[1])
    return {
        "w1": rng.normal(0.0, np.sqrt(2.0 / max(input_dim, 1)), size=(input_dim, h1)),
        "b1": np.zeros((h1,), dtype=np.float64),
        "w2": rng.normal(0.0, np.sqrt(2.0 / max(h1, 1)), size=(h1, h2)),
        "b2": np.zeros((h2,), dtype=np.float64),
        "w3": rng.normal(0.0, np.sqrt(2.0 / max(h2, 1)), size=(h2, 1)),
        "b3": np.zeros((1,), dtype=np.float64),
    }


def _copy_params(params: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {k: v.copy() for k, v in params.items()}


@dataclass
class _AdamState:
    m: Dict[str, np.ndarray]
    v: Dict[str, np.ndarray]
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    step_num: int = 0

    @classmethod
    def from_params(cls, params: Dict[str, np.ndarray]) -> "_AdamState":
        zeros = {k: np.zeros_like(v, dtype=np.float64) for k, v in params.items()}
        return cls(m={k: v.copy() for k, v in zeros.items()}, v=zeros)

    def step(
        self,
        params: Dict[str, np.ndarray],
        grads: Dict[str, np.ndarray],
        *,
        lr: float,
    ) -> None:
        self.step_num += 1
        for name, grad in grads.items():
            self.m[name] = self.beta1 * self.m[name] + (1.0 - self.beta1) * grad
            self.v[name] = self.beta2 * self.v[name] + (1.0 - self.beta2) * (grad * grad)
            m_hat = self.m[name] / (1.0 - self.beta1**self.step_num)
            v_hat = self.v[name] / (1.0 - self.beta2**self.step_num)
            params[name] -= lr * m_hat / (np.sqrt(v_hat) + self.eps)


def _pairwise_logistic_loss(
    params: Dict[str, np.ndarray],
    dataset: PairwiseRankingDataset,
    *,
    weight_decay: float,
) -> float:
    if dataset.num_pairs == 0:
        return 0.0
    scores_better, _ = _mlp_forward(params, dataset.better_features)
    scores_worse, _ = _mlp_forward(params, dataset.worse_features)
    diff = scores_better - scores_worse
    loss_weights = dataset.sample_weights
    denom = max(float(loss_weights.sum()), 1e-8)
    pair_loss = np.sum(loss_weights * np.logaddexp(0.0, diff)) / denom
    reg = 0.5 * float(weight_decay) * (
        np.sum(params["w1"] ** 2) + np.sum(params["w2"] ** 2) + np.sum(params["w3"] ** 2)
    )
    return float(pair_loss + reg)


def _pairwise_logistic_loss_and_grads(
    params: Dict[str, np.ndarray],
    dataset: PairwiseRankingDataset,
    *,
    weight_decay: float,
) -> Tuple[float, Dict[str, np.ndarray]]:
    scores_better, cache_b = _mlp_forward(params, dataset.better_features)
    scores_worse, cache_w = _mlp_forward(params, dataset.worse_features)
    diff = scores_better - scores_worse
    loss_weights = dataset.sample_weights
    denom = max(float(loss_weights.sum()), 1e-8)

    pair_loss = np.sum(loss_weights * np.logaddexp(0.0, diff)) / denom
    grad_diff = (loss_weights / denom) * _sigmoid(diff)
    grad_scores_better = grad_diff
    grad_scores_worse = -grad_diff

    grads_b = _mlp_backward(params, cache_b, grad_scores_better)
    grads_w = _mlp_backward(params, cache_w, grad_scores_worse)
    grads = {name: grads_b[name] + grads_w[name] for name in grads_b}
    for name in ("w1", "w2", "w3"):
        grads[name] += float(weight_decay) * params[name]
    reg = 0.5 * float(weight_decay) * (
        np.sum(params["w1"] ** 2) + np.sum(params["w2"] ** 2) + np.sum(params["w3"] ** 2)
    )
    return float(pair_loss + reg), grads


def _mlp_forward(
    params: Dict[str, np.ndarray],
    x: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    z1 = x @ params["w1"] + params["b1"]
    h1 = np.maximum(z1, 0.0)
    z2 = h1 @ params["w2"] + params["b2"]
    h2 = np.maximum(z2, 0.0)
    scores = h2 @ params["w3"] + params["b3"]
    cache = {
        "x": x,
        "z1": z1,
        "h1": h1,
        "z2": z2,
        "h2": h2,
    }
    return scores.reshape(-1), cache


def _mlp_backward(
    params: Dict[str, np.ndarray],
    cache: Dict[str, np.ndarray],
    grad_scores: np.ndarray,
) -> Dict[str, np.ndarray]:
    ds = grad_scores.reshape(-1, 1)
    grads: Dict[str, np.ndarray] = {}
    grads["w3"] = cache["h2"].T @ ds
    grads["b3"] = ds.sum(axis=0)

    dh2 = ds @ params["w3"].T
    dz2 = dh2 * (cache["z2"] > 0.0)
    grads["w2"] = cache["h1"].T @ dz2
    grads["b2"] = dz2.sum(axis=0)

    dh1 = dz2 @ params["w2"].T
    dz1 = dh1 * (cache["z1"] > 0.0)
    grads["w1"] = cache["x"].T @ dz1
    grads["b1"] = dz1.sum(axis=0)
    return grads


def _sigmoid(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[~pos])
    out[~pos] = exp_x / (1.0 + exp_x)
    return out


def _side_effort(side: Dict[str, Any], budget: int) -> Tuple[int, bool]:
    solved = bool(side.get("solved", False))
    timed_out = bool(side.get("timed_out", False))
    missing_child = bool(side.get("missing_child", False))
    censored = (not solved) or timed_out or missing_child
    pops = int(side.get("pops", budget))
    return (budget if censored else pops), censored
