import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MPLBACKEND", "Agg")

from results import model_comparison


STATE_COLUMNS = [
    "x",
    "y",
    "z",
    "vx",
    "vy",
    "vz",
    "rx",
    "ry",
    "rz",
    "wx",
    "wy",
    "wz",
]

METRIC_COLUMNS = [
    "pos_h1",
    "pos_h10",
    "pos_h50",
    "vel_h1",
    "vel_h10",
    "vel_h50",
    "rot_h1",
    "rot_h10",
    "rot_h50",
    "omega_h1",
    "omega_h10",
    "omega_h50",
    "sim_pos",
    "sim_vel",
    "sim_rot",
    "sim_omega",
]


def build_prediction_df(num_rows=60, scale=1.0):
    t = np.arange(num_rows, dtype=float) * 0.01
    base_values = {
        "x": scale * np.linspace(0.0, 1.0, num_rows),
        "y": scale * np.linspace(1.0, 2.0, num_rows),
        "z": scale * np.linspace(2.0, 3.0, num_rows),
        "vx": scale * np.linspace(-0.1, 0.3, num_rows),
        "vy": scale * np.linspace(0.4, 0.8, num_rows),
        "vz": scale * np.linspace(-0.2, 0.2, num_rows),
        "rx": np.zeros(num_rows),
        "ry": np.zeros(num_rows),
        "rz": np.zeros(num_rows),
        "wx": scale * np.linspace(-0.5, 0.5, num_rows),
        "wy": scale * np.linspace(0.2, 0.6, num_rows),
        "wz": scale * np.linspace(-0.3, 0.1, num_rows),
    }

    data = {"t": t}
    data.update(base_values)

    for horizon in range(1, 51):
        for state in STATE_COLUMNS:
            data[f"{state}_pred_h{horizon}"] = pd.Series(base_values[state]).shift(-(horizon - 1))

    return pd.DataFrame(data)


class ModelComparisonCLITests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def write_prediction_csv(self, filename, scale=1.0):
        csv_path = self.tmp_path / filename
        build_prediction_df(scale=scale).to_csv(csv_path, index=False)
        return csv_path

    def test_single_csv_appends_row_with_created_at(self):
        prediction_csv = self.write_prediction_csv("single.csv")
        summary_path = self.tmp_path / "summary.csv"

        model_comparison.main(
            [
                "--prediction-csv",
                str(prediction_csv),
                "--model-label",
                "lag_gru_single",
                "--model-family",
                "lag_gru",
                "--loss-name",
                "mixed_early",
                "--summary-path",
                str(summary_path),
            ]
        )

        summary_df = pd.read_csv(summary_path)
        self.assertEqual(len(summary_df), 1)
        self.assertIn("created_at", summary_df.columns)
        self.assertEqual(summary_df.loc[0, "model_label"], "lag_gru_single")
        self.assertEqual(summary_df.loc[0, "model_family"], "lag_gru")
        self.assertEqual(summary_df.loc[0, "loss_name"], "mixed_early")
        self.assertEqual(summary_df.loc[0, "model"], "lag_gru_single")
        self.assertIsInstance(summary_df.loc[0, "created_at"], str)
        self.assertGreater(len(summary_df.loc[0, "created_at"]), 0)
        self.assertAlmostEqual(summary_df.loc[0, "pos_h50"], 0.0, places=12)

    def test_multiple_csvs_share_same_timestamp_per_run(self):
        prediction_csv_a = self.write_prediction_csv("multi_a.csv", scale=1.0)
        prediction_csv_b = self.write_prediction_csv("multi_b.csv", scale=1.5)
        summary_path = self.tmp_path / "summary.csv"

        model_comparison.main(
            [
                "--prediction-csv",
                str(prediction_csv_a),
                "--prediction-csv",
                str(prediction_csv_b),
                "--model-label",
                "model_a",
                "--model-label",
                "model_b",
                "--summary-path",
                str(summary_path),
            ]
        )

        summary_df = pd.read_csv(summary_path)
        self.assertEqual(summary_df["model_label"].tolist(), ["model_a", "model_b"])
        self.assertEqual(summary_df["created_at"].nunique(), 1)

    def test_summary_path_appends_instead_of_overwriting(self):
        prediction_csv_a = self.write_prediction_csv("append_a.csv", scale=1.0)
        prediction_csv_b = self.write_prediction_csv("append_b.csv", scale=2.0)
        summary_path = self.tmp_path / "summary.csv"

        model_comparison.main(
            [
                "--prediction-csv",
                str(prediction_csv_a),
                "--model-label",
                "first_run",
                "--summary-path",
                str(summary_path),
            ]
        )
        model_comparison.main(
            [
                "--prediction-csv",
                str(prediction_csv_b),
                "--model-label",
                "second_run",
                "--summary-path",
                str(summary_path),
            ]
        )

        summary_df = pd.read_csv(summary_path)
        self.assertEqual(summary_df["model_label"].tolist(), ["first_run", "second_run"])

    def test_plot_dir_saves_comparison_plots(self):
        prediction_csv = self.write_prediction_csv("plot.csv")
        summary_path = self.tmp_path / "summary.csv"
        plot_dir = self.tmp_path / "plots"

        model_comparison.main(
            [
                "--prediction-csv",
                str(prediction_csv),
                "--model-label",
                "plot_model",
                "--summary-path",
                str(summary_path),
                "--plot-dir",
                str(plot_dir),
            ]
        )

        expected_plots = [
            plot_dir / "metrics_comparison.png",
            plot_dir / "trajectory_h50_comparison.png",
        ]
        for plot_path in expected_plots:
            self.assertTrue(plot_path.is_file())
            self.assertGreater(plot_path.stat().st_size, 0)

    def test_existing_summary_without_created_at_is_preserved(self):
        summary_path = self.tmp_path / "legacy_summary.csv"
        legacy_row = {
            "model": "legacy_model",
            "pos_h1": 1.0,
            "pos_h10": 1.1,
            "pos_h50": 1.2,
            "vel_h1": 2.0,
            "vel_h10": 2.1,
            "vel_h50": 2.2,
            "rot_h1": 3.0,
            "rot_h10": 3.1,
            "rot_h50": 3.2,
            "omega_h1": 4.0,
            "omega_h10": 4.1,
            "omega_h50": 4.2,
            "sim_pos": 5.0,
            "sim_vel": 5.1,
            "sim_rot": 5.2,
            "sim_omega": 5.3,
        }
        pd.DataFrame([legacy_row]).to_csv(summary_path, index=False)

        prediction_csv = self.write_prediction_csv("legacy_append.csv")
        model_comparison.main(
            [
                "--prediction-csv",
                str(prediction_csv),
                "--model-label",
                "new_model",
                "--summary-path",
                str(summary_path),
            ]
        )

        summary_df = pd.read_csv(summary_path)
        self.assertEqual(len(summary_df), 2)
        self.assertIn("created_at", summary_df.columns)
        self.assertEqual(summary_df.loc[0, "model_label"], "legacy_model")
        self.assertTrue(pd.isna(summary_df.loc[0, "created_at"]))
        self.assertEqual(summary_df.loc[1, "model_label"], "new_model")

    def test_prediction_and_label_counts_must_match(self):
        prediction_csv_a = self.write_prediction_csv("mismatch_a.csv")
        prediction_csv_b = self.write_prediction_csv("mismatch_b.csv")

        with self.assertRaisesRegex(ValueError, "--model-label count must match"):
            model_comparison.main(
                [
                    "--prediction-csv",
                    str(prediction_csv_a),
                    "--prediction-csv",
                    str(prediction_csv_b),
                    "--model-label",
                    "only_one_label",
                ]
            )

    def test_missing_prediction_file_raises_clear_error(self):
        missing_csv = self.tmp_path / "missing.csv"

        with self.assertRaisesRegex(FileNotFoundError, "Missing prediction file"):
            model_comparison.main(
                [
                    "--prediction-csv",
                    str(missing_csv),
                    "--model-label",
                    "missing_model",
                ]
            )

    def test_max_horizon_below_50_is_rejected(self):
        prediction_csv = self.write_prediction_csv("too_short.csv")

        with self.assertRaisesRegex(ValueError, "--max-horizon must be at least 50"):
            model_comparison.main(
                [
                    "--prediction-csv",
                    str(prediction_csv),
                    "--model-label",
                    "short_horizon",
                    "--max-horizon",
                    "49",
                ]
            )


if __name__ == "__main__":
    unittest.main()
