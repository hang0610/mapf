#!/usr/bin/env python3
"""
Smoke tests for learned conflict selection in CBS.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from planners.conflict_ranker import (  # noqa: E402
    MLPConflictRanker,
    NodeRankingExample,
    fit_mlp_ranker,
    summarize_rollout_label,
)


def _write_toy_map_and_scen(tmpdir: Path) -> tuple[Path, Path]:
    map_path = tmpdir / "toy.map"
    scen_path = tmpdir / "toy.scen"
    map_path.write_text("type octile\nheight 3\nwidth 3\nmap\n...\n...\n...\n", encoding="utf-8")

    rows = [
        (1, 0, 1, 2),
        (1, 2, 1, 0),
        (0, 1, 2, 1),
        (1, 0, 1, 2),
        (1, 2, 1, 0),
        (2, 1, 0, 1),
        (0, 1, 2, 1),
        (2, 1, 0, 1),
        (1, 0, 1, 2),
    ]
    with scen_path.open("w", encoding="utf-8") as fh:
        fh.write("version 1\n")
        for start_c, start_r, goal_c, goal_r in rows:
            fh.write(
                f"0\ttoy.map\t3\t3\t{start_c}\t{start_r}\t{goal_c}\t{goal_r}\t2\n"
            )
    return map_path, scen_path


class TestCBSLearning(unittest.TestCase):
    def test_rollout_label_summary(self) -> None:
        label = {
            "side_a": {"solved": True, "pops": 3, "soc": 10, "timed_out": False},
            "side_b": {"solved": False, "pops": 2, "soc": None, "timed_out": True},
            "max_ct_pops_budget": 7,
        }
        summary = summarize_rollout_label(label)
        self.assertEqual(summary["effort_left"], 3)
        self.assertEqual(summary["effort_right"], 7)
        self.assertEqual(summary["effort_sum"], 10)
        self.assertEqual(summary["effort_min"], 3)
        self.assertTrue(summary["right_censored"])
        self.assertTrue(summary["censored"])

    def test_ranker_roundtrip(self) -> None:
        rows_a = np.asarray(
            [
                [0.1, 0.0, 0.2, 0.0, 0.2, 0.3, 0.1, 0.4, 0.2, 0.5, 0.1, 0.3, 0.2],
                [0.8, 0.2, 0.7, 0.6, 0.7, 0.6, 0.8, 0.6, 0.7, 0.2, 0.6, 0.7, 0.5],
                [1.0, 0.3, 0.9, 0.9, 0.8, 0.7, 0.9, 0.8, 0.8, 0.1, 0.8, 0.9, 0.7],
            ],
            dtype=np.float64,
        )
        rows_b = np.asarray(
            [
                [0.2, 0.1, 0.3, 0.1, 0.1, 0.2, 0.2, 0.3, 0.2, 0.4, 0.2, 0.3, 0.1],
                [0.9, 0.3, 0.7, 0.7, 0.8, 0.7, 0.7, 0.6, 0.8, 0.1, 0.7, 0.8, 0.6],
                [1.1, 0.4, 0.9, 0.8, 0.9, 0.8, 0.8, 0.7, 0.9, 0.1, 0.8, 0.8, 0.8],
            ],
            dtype=np.float64,
        )
        examples = [
            NodeRankingExample(
                features=rows_a,
                labels=np.asarray([1.0, 3.0, 6.0], dtype=np.float64),
                feature_names=tuple(f"f{i}" for i in range(rows_a.shape[1])),
            ),
            NodeRankingExample(
                features=rows_b,
                labels=np.asarray([1.5, 3.5, 5.0], dtype=np.float64),
                feature_names=tuple(f"f{i}" for i in range(rows_b.shape[1])),
            ),
        ]
        ranker, summary = fit_mlp_ranker(
            examples,
            delta=0.5,
            epochs=250,
            batch_size=8,
            learning_rate=1e-2,
            patience=50,
            random_seed=0,
        )
        self.assertIsNotNone(summary["pairwise_accuracy"])
        self.assertGreater(summary["pairwise_accuracy"], 0.8)

        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "ranker.npz"
            ranker.save_npz(model_path)
            blob = np.load(model_path, allow_pickle=False)
            for key in ("w1", "b1", "w2", "b2", "w3", "b3"):
                self.assertIn(key, blob.files)
            loaded = MLPConflictRanker.load_npz(model_path)
            feats = np.asarray(
                [
                    [0.3, 0.2, 0.4, 0.1, 0.2, 0.3, 0.1, 0.4, 0.2, 0.4, 0.1, 0.2, 0.2],
                    [1.0, 0.2, 0.8, 0.7, 0.9, 0.6, 0.7, 0.8, 0.9, 0.2, 0.7, 0.8, 0.6],
                ],
                dtype=np.float64,
            )
            scores = ranker.score_features(feats, tuple(f"f{i}" for i in range(feats.shape[1])))
            self.assertEqual(scores.shape, (2,))
            np.testing.assert_allclose(
                scores,
                loaded.score_features(feats, tuple(f"f{i}" for i in range(feats.shape[1]))),
            )

    def test_end_to_end_collect_train_and_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            map_path, scen_path = _write_toy_map_and_scen(tmpdir)
            data_dir = tmpdir / "dataset"
            model_path = tmpdir / "ranker.npz"
            summary_path = tmpdir / "train_summary.json"
            learned_paths = tmpdir / "learned_paths.npy"
            learned_stats = tmpdir / "learned_stats.json"
            earliest_paths = tmpdir / "earliest_paths.npy"

            collect_cmd = [
                sys.executable,
                "-m",
                "scripts.collect_cbs_dataset",
                "--map",
                "toy",
                "--map_path",
                str(map_path),
                "--scen_path",
                str(scen_path),
                "--k",
                "3",
                "--train_instances",
                "2",
                "--val_instances",
                "1",
                "--out_dir",
                str(data_dir),
                "--rollout_max_pops",
                "12",
                "--max_time",
                "32",
            ]
            res = subprocess.run(
                collect_cmd,
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
            )
            self.assertEqual(res.returncode, 0, msg=res.stderr or res.stdout)

            train_cmd = [
                sys.executable,
                "-m",
                "scripts.train_conflict_ranker",
                "--train_jsonl",
                str(data_dir / "train.jsonl"),
                "--val_jsonl",
                str(data_dir / "val.jsonl"),
                "--model_out",
                str(model_path),
                "--summary_json",
                str(summary_path),
            ]
            res = subprocess.run(
                train_cmd,
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
            )
            self.assertEqual(res.returncode, 0, msg=res.stderr or res.stdout)
            self.assertTrue(model_path.exists())

            earliest_cmd = [
                sys.executable,
                "-m",
                "scripts.run_cbs",
                "--map",
                "toy",
                "--map_path",
                str(map_path),
                "--scen_path",
                str(scen_path),
                "--k",
                "3",
                "--out",
                str(earliest_paths),
                "--validate",
                "--max_time",
                "32",
            ]
            res = subprocess.run(
                earliest_cmd,
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
            )
            self.assertEqual(res.returncode, 0, msg=res.stderr or res.stdout)
            self.assertTrue(earliest_paths.exists())

            learned_cmd = [
                sys.executable,
                "-m",
                "scripts.run_cbs",
                "--map",
                "toy",
                "--map_path",
                str(map_path),
                "--scen_path",
                str(scen_path),
                "--k",
                "3",
                "--out",
                str(learned_paths),
                "--validate",
                "--max_time",
                "32",
                "--conflict_policy",
                "learned",
                "--model_path",
                str(model_path),
                "--stats_json",
                str(learned_stats),
            ]
            res = subprocess.run(
                learned_cmd,
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
            )
            self.assertEqual(res.returncode, 0, msg=res.stderr or res.stdout)
            self.assertIn("Learned conflict selector:", res.stdout)
            self.assertTrue(learned_paths.exists())
            stats = json.loads(learned_stats.read_text(encoding="utf-8"))
            self.assertEqual(stats["effective_conflict_policy"], "learned")
            self.assertGreaterEqual(stats["learned_policy_calls"], 1)


if __name__ == "__main__":
    unittest.main()
