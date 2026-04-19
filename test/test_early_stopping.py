import sys
import unittest
from argparse import Namespace
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train.train_physres_ablation import (  # noqa: E402
    ensure_resume_checkpoint_is_resumable,
    validate_resume_checkpoint,
)
from utils.early_stopping import get_wait_count, should_stop_early  # noqa: E402


def make_args(**overrides):
    defaults = {
        "variant": "baseline",
        "horizon": 50,
        "epochs": 500,
        "loss_type": "mixed",
        "lr_start": 1e-5,
        "lr_end": 1e-8,
        "hidden_dim": 64,
        "gru_hidden_dim": 64,
        "beta_geo": 0.01,
        "beta_aux": 0.05,
        "beta_force": 0.1,
        "w_rot": 2.0,
        "w_omega": 2.0,
        "lag_mode": "per_motor",
        "alpha_init": 0.85,
        "early_stop_patience": 12,
        "early_stop_min_delta": 1e-3,
        "early_stop_start_epoch": 25,
    }
    defaults.update(overrides)
    return Namespace(**defaults)


def make_checkpoint(**config_overrides):
    config = {
        "variant": "baseline",
        "horizon": 50,
        "total_epochs": 500,
        "early_stop_patience": 12,
        "early_stop_min_delta": 1e-3,
        "early_stop_start_epoch": 25,
    }
    config.update(config_overrides)
    return {
        "model_state": {},
        "config": config,
        "train_trajs": ["random", "square"],
        "valid_trajs": ["random", "square"],
    }


class EarlyStoppingUtilsTests(unittest.TestCase):
    def test_wait_count_ignores_epochs_before_warmup(self):
        self.assertEqual(get_wait_count(20, 10, start_epoch=20), 1)
        self.assertEqual(get_wait_count(21, 10, start_epoch=20), 2)

    def test_wait_count_uses_best_epoch_after_warmup(self):
        self.assertEqual(get_wait_count(20, 20, start_epoch=20), 0)
        self.assertEqual(get_wait_count(21, 20, start_epoch=20), 1)

    def test_should_stop_after_patience_epochs_since_warmup(self):
        self.assertFalse(should_stop_early(21, 10, start_epoch=20, patience=3))
        self.assertTrue(should_stop_early(22, 10, start_epoch=20, patience=3))


class ResumeValidationTests(unittest.TestCase):
    def test_resume_rejects_best_model_checkpoint(self):
        best_model_path = Path("/tmp/physres_best.pt")
        with self.assertRaisesRegex(ValueError, "best-model checkpoint"):
            ensure_resume_checkpoint_is_resumable(best_model_path, best_model_path)

    def test_validate_resume_checkpoint_accepts_matching_early_stop_config(self):
        validate_resume_checkpoint(
            checkpoint=make_checkpoint(),
            args=make_args(),
            train_trajs=["random", "square"],
            valid_trajs=["random", "square"],
            aux_cols=["az_body"],
        )

    def test_validate_resume_checkpoint_rejects_early_stop_mismatch(self):
        with self.assertRaisesRegex(ValueError, "early_stop_patience"):
            validate_resume_checkpoint(
                checkpoint=make_checkpoint(),
                args=make_args(early_stop_patience=8),
                train_trajs=["random", "square"],
                valid_trajs=["random", "square"],
                aux_cols=["az_body"],
            )


if __name__ == "__main__":
    unittest.main()
