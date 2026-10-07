from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import tarfile
from io import BytesIO
import tomllib
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import download_models
import export_public_release
import check_release_archives
from vfx_element_tagger import __version__
from vfx_element_tagger.web_ui import index_html
from vfx_element_tagger.review_ui import review_html


class ReleaseSetupTests(unittest.TestCase):
    def test_source_archive_owner_metadata_is_removed_without_payload_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "example.tar.gz"
            payload = b"# Public source\n"
            with tarfile.open(path, "w:gz") as archive:
                entry = tarfile.TarInfo("project/README.md")
                entry.size = len(payload)
                entry.uid, entry.gid, entry.uname, entry.gname = 501, 80, "private-user", "staff"
                archive.addfile(entry, BytesIO(payload))
            with self.assertRaises(ValueError):
                check_release_archives.check_archive(path)
            check_release_archives.normalize_sdist(path)
            self.assertEqual(check_release_archives.check_archive(path), 1)
            with tarfile.open(path) as archive:
                self.assertEqual(archive.extractfile("project/README.md").read(), payload)

    def test_release_archive_rejects_private_media_and_paths(self):
        check_release_archives.check_entry("project/setup.cfg", b"[egg_info]\n")
        for name in ("project/test_assets/example.mov", "project/screenshots/example.jpg",
                     "project/.vfx-tagger.json", "project/library.json", "../outside.py"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                check_release_archives.check_entry(name, b"example")
        with self.assertRaises(ValueError):
            check_release_archives.check_entry("project/README.md", ("/" + "Users" + "/example/file").encode())

    def test_public_export_excludes_private_data_and_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            examples = ["README.md", "src/catalog.py", "scripts/export_public_release.py",
                        "scripts/prepare_article_catalog.py", "scripts/stress_test_analysis.py",
                        "evaluation/BENCHMARK-PROTOCOL.md", "evaluation/private.json",
                        ".git/config", ".vfx-tagger.json", "test_assets/clip.mov",
                        "docs/screenshots/library.jpg", "models/weights.safetensors"]
            for name in examples:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("safe example")
            files = export_public_release.public_files(root)
            self.assertEqual({str(path.relative_to(root)) for path in files},
                             {"README.md", "src/catalog.py", "scripts/export_public_release.py",
                              "evaluation/BENCHMARK-PROTOCOL.md"})
            self.assertEqual(export_public_release.scan_files(root, files), [])
            (root / "README.md").write_text("/" + "Users" + "/example/private.mov")
            self.assertEqual(len(export_public_release.scan_files(root, files)), 1)

    def test_artist_dialog_and_mobile_layout_guards_are_present(self):
        html = index_html([])
        self.assertIn('<dialog id="rater-dialog"', html)
        self.assertIn('aria-labelledby="rater-title"', html)
        self.assertNotIn("window.prompt", html)
        self.assertIn(".inspector{z-index:25}", html)
        self.assertIn("grid-template-columns:minmax(0,1fr)", review_html())

    def test_package_version_and_license_match_metadata(self):
        metadata = tomllib.loads((ROOT / "pyproject.toml").read_text())
        self.assertEqual(metadata["project"]["version"], __version__)
        self.assertEqual(metadata["project"]["license"], "MIT")
        self.assertIn("LICENSE", metadata["project"]["license-files"])
        self.assertIn("NOTICE", metadata["project"]["license-files"])
        self.assertIn("MIT License", (ROOT / "LICENSE").read_text())

    def test_downloader_uses_current_hub_signature_and_preserves_entries(self):
        calls = []

        def snapshot_download(*, repo_id, local_dir, revision):
            self.assertEqual(len(revision), 40)
            calls.append(repo_id)
            local_dir.mkdir(parents=True, exist_ok=True)
            return str(local_dir)

        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            previous = {"models": {"custom": {"backend": "command", "command": ["local-wrapper"]}}}
            (directory / "manifest.json").write_text(json.dumps(previous))
            argv = ["download_models.py", "--models-dir", tmp]
            with patch.object(download_models, "verify_snapshot", return_value={"revision": "a" * 40}), patch.object(sys, "argv", argv), patch.dict(
                sys.modules, {"huggingface_hub": SimpleNamespace(snapshot_download=snapshot_download)}
            ):
                self.assertEqual(download_models.main(), 0)
            manifest = json.loads((directory / "manifest.json").read_text())
            self.assertIn("custom", manifest["models"])
            self.assertIn("analysis_primary", manifest["models"])
            self.assertEqual(calls, [download_models.MODELS["analysis_primary"]["repo_id"]])

    def test_unavailable_challenger_does_not_discard_primary(self):
        def snapshot_download(*, repo_id, local_dir, revision):
            if "Marlin" in repo_id:
                raise PermissionError("gated snapshot")
            local_dir.mkdir(parents=True, exist_ok=True)
            return str(local_dir)

        with tempfile.TemporaryDirectory() as tmp:
            argv = ["download_models.py", "--models-dir", tmp, "--with-challengers"]
            with patch.object(download_models, "verify_snapshot", return_value={"revision": "a" * 40}), patch.object(sys, "argv", argv), patch.dict(
                sys.modules, {"huggingface_hub": SimpleNamespace(snapshot_download=snapshot_download)}
            ):
                self.assertEqual(download_models.main(), 0)
            manifest = json.loads((Path(tmp) / "manifest.json").read_text())
            self.assertIn("analysis_primary", manifest["models"])
            self.assertIn("challenger_minicpm", manifest["models"])
            self.assertIn("challenger_marlin", manifest["unavailable_models"])
