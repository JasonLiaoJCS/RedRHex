import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.training_panel.training_panel.convergence import (
    PRESETS,
    ConvergenceChecker,
    ConvergenceConfig,
    apply_settings,
    load_convergence_config,
    save_convergence_config,
)


def _make_scalars(n: int, start: float = 1.0, end: float = 5.0) -> list[tuple[int, float]]:
    """Generate linearly-improving scalars (still improving)."""
    step = (end - start) / max(n - 1, 1)
    return [(i, start + i * step) for i in range(n)]


def _make_flat_scalars(n: int, value: float = 4.0, noise: float = 0.01) -> list[tuple[int, float]]:
    """Generate flat (converged) scalars with tiny noise."""
    import math
    return [(i, value + noise * math.sin(i)) for i in range(n)]


class ConvergenceCheckerTests(unittest.TestCase):
    def _checker_with_scalars(self, scalars: list[tuple[int, float]]) -> ConvergenceChecker:
        checker = ConvergenceChecker()
        checker.read_scalars = lambda log_dir, tag: scalars
        return checker

    def test_detects_plateau_below_threshold(self):
        scalars = _make_flat_scalars(300, value=4.0, noise=0.01)
        checker = self._checker_with_scalars(scalars)
        result = checker.check(Path("/fake"), ConvergenceConfig(
            window_iterations=200, min_improvement_pct=2.0, min_iterations=100))
        self.assertTrue(result.detected)
        self.assertGreater(result.iteration, 0)

    def test_does_not_detect_when_still_improving(self):
        scalars = _make_scalars(300, start=1.0, end=5.0)
        checker = self._checker_with_scalars(scalars)
        result = checker.check(Path("/fake"), ConvergenceConfig(
            window_iterations=200, min_improvement_pct=2.0, min_iterations=100))
        self.assertFalse(result.detected)

    def test_does_not_detect_before_min_iterations(self):
        scalars = _make_flat_scalars(50, value=4.0)
        checker = self._checker_with_scalars(scalars)
        result = checker.check(Path("/fake"), ConvergenceConfig(min_iterations=100))
        self.assertFalse(result.detected)
        self.assertIn("only 50 iterations", result.reason)

    def test_returns_not_detected_for_empty_scalars(self):
        checker = self._checker_with_scalars([])
        result = checker.check(Path("/fake"), ConvergenceConfig())
        self.assertFalse(result.detected)
        self.assertIn("no data", result.reason)

    def test_returns_not_detected_when_mean_is_zero(self):
        scalars = [(i, 0.0) for i in range(200)]
        checker = self._checker_with_scalars(scalars)
        result = checker.check(Path("/fake"), ConvergenceConfig(min_iterations=100))
        self.assertFalse(result.detected)

    def test_check_never_raises(self):
        checker = ConvergenceChecker()
        checker.read_scalars = lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("boom"))
        try:
            checker.read_scalars = lambda *a, **kw: []
            result = checker.check(Path("/nonexistent"), ConvergenceConfig())
            self.assertFalse(result.detected)
        except Exception as exc:
            self.fail(f"check() raised unexpectedly: {exc}")


class ConvergenceConfigTests(unittest.TestCase):
    def test_preset_overrides_window_and_threshold(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_file = Path(tmp) / "convergence_config.json"
            for preset_name, preset_values in PRESETS.items():
                cfg = apply_settings({"preset": preset_name}, config_file)
                self.assertEqual(cfg.window_iterations, preset_values["window_iterations"])
                self.assertAlmostEqual(cfg.min_improvement_pct,
                                       preset_values["min_improvement_pct"])

    def test_custom_preset_keeps_user_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_file = Path(tmp) / "convergence_config.json"
            cfg = apply_settings(
                {"preset": "custom", "window_iterations": 350, "min_improvement_pct": 3.5},
                config_file,
            )
            self.assertEqual(cfg.window_iterations, 350)
            self.assertAlmostEqual(cfg.min_improvement_pct, 3.5)

    def test_save_and_load_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_file = Path(tmp) / "logs" / "convergence_config.json"
            original = ConvergenceConfig(enabled=False, preset="strict",
                                         window_iterations=400, min_improvement_pct=1.0,
                                         cooldown_minutes=30, auto_record_video=False)
            save_convergence_config(original, config_file)
            loaded = load_convergence_config(config_file)
            self.assertFalse(loaded.enabled)
            self.assertFalse(loaded.auto_record_video)
            self.assertEqual(loaded.cooldown_minutes, 30)

    def test_load_returns_defaults_for_missing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_file = Path(tmp) / "nonexistent.json"
            cfg = load_convergence_config(config_file)
            self.assertIsInstance(cfg, ConvergenceConfig)
            self.assertTrue(cfg.enabled)
            self.assertEqual(cfg.preset, "default")

    def test_load_returns_defaults_for_corrupt_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_file = Path(tmp) / "convergence_config.json"
            config_file.write_text("not valid json {{{{")
            cfg = load_convergence_config(config_file)
            self.assertIsInstance(cfg, ConvergenceConfig)
            self.assertTrue(cfg.enabled)

    def test_apply_settings_ignores_unknown_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_file = Path(tmp) / "convergence_config.json"
            cfg = apply_settings({"unknown_field": "bad", "enabled": False}, config_file)
            self.assertFalse(cfg.enabled)
            self.assertFalse(hasattr(cfg, "unknown_field"))


if __name__ == "__main__":
    unittest.main()
