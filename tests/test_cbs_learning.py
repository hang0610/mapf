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
    compute_pairwise_loss,
    evaluate_ranker,
    MLPConflictRanker,
    NodeRankingExample,
    fit_mlp_ranker,
    summarize_rollout_label,
)
from scripts.playback_paths import scenario_instance  # noqa: E402


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
        eval_summary = evaluate_ranker(ranker, examples, delta=0.5)
        self.assertIsNotNone(eval_summary["pairwise_loss"])
        self.assertAlmostEqual(
            float(eval_summary["pairwise_loss"]),
            float(compute_pairwise_loss(ranker, examples, delta=0.5)),
        )

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

    def test_playback_instance_matches_run_cbs_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            map_path = tmpdir / "toy.map"
            scen_path = tmpdir / "toy.scen"
            map_path.write_text("type octile\nheight 3\nwidth 3\nmap\n...\n.@.\n...\n", encoding="utf-8")
            with scen_path.open("w", encoding="utf-8") as fh:
                fh.write("version 1\n")
                fh.write("0\ttoy.map\t3\t3\t1\t1\t0\t0\t1\n")  # invalid, on obstacle
                fh.write("0\ttoy.map\t3\t3\t0\t0\t2\t2\t1\n")
                fh.write("0\ttoy.map\t3\t3\t0\t2\t2\t0\t1\n")
                fh.write("0\ttoy.map\t3\t3\t2\t0\t0\t2\t1\n")
                fh.write("0\ttoy.map\t3\t3\t2\t2\t0\t0\t1\n")

            grid = np.asarray(
                [
                    [0, 0, 0],
                    [0, 1, 0],
                    [0, 0, 0],
                ],
                dtype=np.int64,
            )
            scen_starts = np.asarray(
                [[1, 1], [0, 0], [2, 0], [0, 2], [2, 2]],
                dtype=np.int64,
            )
            scen_goals = np.asarray(
                [[0, 0], [2, 2], [0, 2], [2, 0], [0, 0]],
                dtype=np.int64,
            )

            instance = scenario_instance(
                grid,
                scen_starts,
                scen_goals,
                k=2,
                offset=1,
                filter_invalid_rows=False,
            )
            np.testing.assert_array_equal(instance.starts, scen_starts[1:3])
            np.testing.assert_array_equal(instance.goals, scen_goals[1:3])

            filtered = scenario_instance(
                grid,
                scen_starts,
                scen_goals,
                k=2,
                offset=1,
                filter_invalid_rows=True,
            )
            self.assertFalse(np.array_equal(filtered.starts, instance.starts))

            paths_path = tmpdir / "paths.npy"
            gif_path = tmpdir / "demo.gif"
            np.save(
                paths_path,
                np.asarray(
                    [
                        instance.starts,
                        instance.goals,
                    ],
                    dtype=np.int64,
                ),
            )

            playback_cmd = [
                sys.executable,
                "-m",
                "scripts.playback_paths",
                "--map",
                "toy",
                "--map_path",
                str(map_path),
                "--scen_path",
                str(scen_path),
                "--k",
                "2",
                "--offset",
                "1",
                "--paths",
                str(paths_path),
                "--out",
                str(gif_path),
                "--fps",
                "2",
            ]
            res = subprocess.run(
                playback_cmd,
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
            )
            self.assertEqual(res.returncode, 0, msg=res.stderr or res.stdout)
            self.assertTrue(gif_path.exists())
            self.assertNotIn("[WARN] paths has N=", res.stdout)

    def test_eval_conflict_policies_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            map_path, scen_path = _write_toy_map_and_scen(tmpdir)
            data_dir = tmpdir / "dataset"
            model_path = tmpdir / "ranker.npz"
            summary_path = tmpdir / "train_summary.json"
            eval_dir = tmpdir / "eval"

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
                "--offset_step",
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

            eval_cmd = [
                sys.executable,
                "-m",
                "scripts.eval_conflict_policies",
                "--map",
                "toy",
                "--map_path",
                str(map_path),
                "--scen_path",
                str(scen_path),
                "--k",
                "3",
                "--offset_start",
                "6",
                "--offset_step",
                "3",
                "--num_instances",
                "1",
                "--policies",
                "earliest",
                "learned",
                "--model_path",
                str(model_path),
                "--out_dir",
                str(eval_dir),
                "--max_time",
                "32",
                "--make_gif_for_offsets",
                "6",
            ]
            res = subprocess.run(
                eval_cmd,
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
            )
            self.assertEqual(res.returncode, 0, msg=res.stderr or res.stdout)

            summary_rows = [
                json.loads(line)
                for line in (eval_dir / "summary.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual({row["policy"] for row in summary_rows}, {"earliest", "learned"})
            learned_rows = [row for row in summary_rows if row["policy"] == "learned"]
            self.assertEqual(learned_rows[0]["effective_conflict_policy"], "learned")
            self.assertIsNotNone(learned_rows[0]["stats_path"])
            self.assertTrue(Path(learned_rows[0]["stats_path"]).exists())
            self.assertTrue((eval_dir / "earliest" / "offset_000006.gif").exists())
            self.assertTrue((eval_dir / "learned" / "offset_000006.gif").exists())


if __name__ == "__main__":
    unittest.main()
