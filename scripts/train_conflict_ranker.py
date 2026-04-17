#!/usr/bin/env python3
"""
Train a learned conflict selector from CT expansion JSONL logs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from planners.conflict_ranker import (
    compute_pairwise_loss,
    evaluate_ranker,
    fit_mlp_ranker,
    load_node_examples,
)


def _resolve_paths(paths: List[str]) -> List[Path]:
    return [Path(p).expanduser().resolve() for p in paths]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a two-hidden-layer MLP ranker for CBS conflict selection."
    )
    parser.add_argument(
        "--train_jsonl",
        type=str,
        nargs="+",
        required=True,
        help="One or more training JSONL shards produced by collect_cbs_dataset.py.",
    )
    parser.add_argument(
        "--val_jsonl",
        type=str,
        nargs="*",
        default=[],
        help="Optional validation JSONL shards.",
    )
    parser.add_argument(
        "--test_jsonl",
        type=str,
        nargs="*",
        default=[],
        help="Optional test JSONL shards.",
    )
    parser.add_argument(
        "--label",
        choices=("effort_sum", "effort_min"),
        default="effort_sum",
        help="Supervision target derived from rollout_labels.",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=0.5,
        help="Ignore pairs whose labels differ by <= delta.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Maximum number of training epochs for the MLP ranker.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Mini-batch size used by the MLP trainer.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Adam learning rate for the MLP trainer.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="L2 regularization strength for the MLP weights.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=15,
        help="Early stopping patience in epochs.",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=0,
        help="Random seed used by the trainer backend.",
    )
    parser.add_argument(
        "--drop_feature",
        type=str,
        nargs="*",
        default=[],
        help="Optional feature names to drop for ablations.",
    )
    parser.add_argument(
        "--model_out",
        type=str,
        required=True,
        help="Output path for the exported .npz model.",
    )
    parser.add_argument(
        "--summary_json",
        type=str,
        default=None,
        help="Optional output path for training/evaluation summary JSON.",
    )
    args = parser.parse_args()

    train_paths = _resolve_paths(args.train_jsonl)
    val_paths = _resolve_paths(args.val_jsonl)
    test_paths = _resolve_paths(args.test_jsonl)

    for path in train_paths + val_paths + test_paths:
        if not path.exists():
            raise FileNotFoundError(f"JSONL shard not found: {path}")

    train_examples = load_node_examples(
        train_paths,
        label_key=args.label,
        drop_feature_names=args.drop_feature,
    )
    if not train_examples:
        raise ValueError("no usable training rows found in the provided JSONL files")

    val_examples = (
        load_node_examples(
            val_paths,
            label_key=args.label,
            drop_feature_names=args.drop_feature,
        )
        if val_paths
        else []
    )
    test_examples = (
        load_node_examples(
            test_paths,
            label_key=args.label,
            drop_feature_names=args.drop_feature,
        )
        if test_paths
        else []
    )

    ranker, train_summary = fit_mlp_ranker(
        train_examples,
        val_examples=val_examples if val_examples else None,
        delta=args.delta,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        random_seed=args.random_seed,
    )
    ranker.metadata.update(
        {
            "label_key": args.label,
            "drop_feature_names": list(args.drop_feature),
            "train_jsonl": [str(p) for p in train_paths],
            "model_type": "mlp_2layer",
        }
    )

    summary = {
        "model_out": str(Path(args.model_out).expanduser().resolve()),
        "model_type": "mlp_2layer",
        "label_key": args.label,
        "delta": args.delta,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
        "random_seed": args.random_seed,
        "drop_feature_names": list(args.drop_feature),
        "feature_names": list(ranker.feature_names),
        "train": train_summary,
    }

    if val_examples:
        summary["val"] = evaluate_ranker(ranker, val_examples, delta=args.delta)
        summary["val"]["num_nodes"] = len(val_examples)
        summary["val"]["pairwise_loss"] = compute_pairwise_loss(
            ranker,
            val_examples,
            delta=args.delta,
        )

    if test_examples:
        summary["test"] = evaluate_ranker(ranker, test_examples, delta=args.delta)
        summary["test"]["num_nodes"] = len(test_examples)
        summary["test"]["pairwise_loss"] = compute_pairwise_loss(
            ranker,
            test_examples,
            delta=args.delta,
        )

    model_path = Path(args.model_out).expanduser().resolve()
    model_path.parent.mkdir(parents=True, exist_ok=True)
    ranker.save_npz(model_path)
    print(f"Saved ranker: {model_path}")
    print(
        f"Train loss={summary['train']['pairwise_loss']} "
        f"pairwise_accuracy={summary['train']['pairwise_accuracy']} "
        f"top1_accuracy={summary['train']['top1_accuracy']}"
    )
    if "val" in summary:
        print(
            f"Val loss={summary['val']['pairwise_loss']} "
            f"pairwise_accuracy={summary['val']['pairwise_accuracy']} "
            f"top1_accuracy={summary['val']['top1_accuracy']}"
        )
    if "test" in summary:
        print(
            f"Test loss={summary['test']['pairwise_loss']} "
            f"pairwise_accuracy={summary['test']['pairwise_accuracy']} "
            f"top1_accuracy={summary['test']['top1_accuracy']}"
        )

    if args.summary_json is not None:
        summary_path = Path(args.summary_json).expanduser().resolve()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
