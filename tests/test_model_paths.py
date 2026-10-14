import os
from datetime import datetime
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ddae_seismic.scripts._common import make_run_dir, resolve_model_path


class ModelPathTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        cwd = Path.cwd()
        os.chdir(self.tmp.name)
        self.addCleanup(os.chdir, cwd)

    def save_model(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("test model placeholder", encoding="utf-8")
        return path

    def test_relative_root_symlink_target_and_resolution(self):
        out = make_run_dir("runs", "syn")
        model = self.save_model(out / "best.keras")
        latest = Path("runs/syn/latest")
        if latest.is_symlink():
            self.assertEqual(os.readlink(latest), out.name)
            self.assertTrue((latest / "best.keras").is_file())
        self.assertEqual(resolve_model_path(latest / "best.keras"), model.resolve())

    def test_symlink_denied_falls_back_to_text(self):
        with patch.object(Path, "symlink_to", side_effect=OSError("no symlink privileges")):
            out = make_run_dir("runs", "field")
        model = self.save_model(out / "best.keras")
        self.assertFalse(Path("runs/field/latest").exists())
        self.assertEqual(resolve_model_path("runs/field/latest/best.keras"), model.resolve())

    def test_existing_latest_folder_does_not_mask_new_pointer(self):
        old = self.save_model(Path("runs/syn/latest/best.keras"))
        out = make_run_dir("runs", "syn")
        new = self.save_model(out / "best.keras")
        self.assertEqual(resolve_model_path(old), new.resolve())
        self.assertTrue(old.is_file())

    def test_successive_runs_update_pointer(self):
        with patch("ddae_seismic.scripts._common.datetime") as clock:
            clock.now.return_value = datetime(2026, 10, 14, 7, 54, 57)
            first = make_run_dir("runs", "syn")
            second = make_run_dir("runs", "syn")
        self.assertNotEqual(first, second)
        model = self.save_model(second / "best.keras")
        self.assertEqual(resolve_model_path("runs/syn/latest/best.keras"), model.resolve())

    def test_absolute_and_legacy_relative_pointers(self):
        model = self.save_model(Path("runs/syn/old/best.keras"))
        for value in ("old", "runs/syn/old", str(model.parent.resolve())):
            with self.subTest(value=value):
                Path("runs/syn/LATEST.txt").write_text(value, encoding="utf-8")
                self.assertEqual(resolve_model_path("runs/syn/latest/best.keras"), model.resolve())

    def test_relative_pointer_works_outside_original_cwd(self):
        out = make_run_dir("runs", "syn")
        model = self.save_model(out / "best.keras").resolve()
        alias = Path.cwd() / "runs/syn/latest/best.keras"
        Path("elsewhere").mkdir()
        os.chdir("elsewhere")
        self.assertEqual(resolve_model_path(alias), model)

    def test_explicit_model_path(self):
        model = self.save_model(Path("models/best.keras"))
        self.assertEqual(resolve_model_path(model), model.resolve())

    def test_missing_model_and_empty_or_stale_pointer(self):
        with self.assertRaises(FileNotFoundError):
            resolve_model_path("missing.keras")
        self.save_model(Path("runs/syn/latest/best.keras"))
        for value in ("", "does-not-exist"):
            Path("runs/syn/LATEST.txt").write_text(value, encoding="utf-8")
            with self.assertRaises(FileNotFoundError):
                resolve_model_path("runs/syn/latest/best.keras")


if __name__ == "__main__":
    unittest.main()
