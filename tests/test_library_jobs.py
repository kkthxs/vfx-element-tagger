from __future__ import annotations

import http.client
import io
from contextlib import redirect_stdout
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vfx_element_tagger.catalog_lock import CatalogBusy, catalog_writer
from vfx_element_tagger.library_jobs import LibraryJobs, browse_folders, local_folder, worker
from vfx_element_tagger.models import Element, SourceRepresentation
from vfx_element_tagger.store import LibraryStore
from vfx_element_tagger.web import QCServer, QCRequestHandler, _review_version
from vfx_element_tagger.web_ui import index_html
from vfx_element_tagger.review_ui import review_html


def example():
    rep = SourceRepresentation("rep", "/example/clip.mov", "/example/clip.mov", "mov")
    return Element("one", [rep], "rep", caption="Artist choice",
                   human_overrides={"fields": {"primary_family": "smoke"}})


class LibraryJobTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        self.folder = self.root / "footage"
        self.folder.mkdir()
        self.store = LibraryStore(self.root / "library.sqlite3")
        self.store.save([example()])
        self.jobs = LibraryJobs(self.store.path, self.root / "cache", self.root / "missing.json")

    def tearDown(self):
        self.jobs.shutdown()
        self.tmp.cleanup()

    def test_folder_browser_lists_directories_only(self):
        (self.folder / "subfolder").mkdir()
        (self.folder / ".private").mkdir()
        (self.folder / "clip.txt").write_text("not footage")
        data = browse_folders(str(self.folder))
        self.assertEqual([x["name"] for x in data["folders"]], ["subfolder"])
        self.assertEqual(data["parent"], str(self.root))
        with self.assertRaises(ValueError):
            local_folder("relative/path")
        with self.assertRaises(ValueError):
            local_folder(str(self.folder / "clip.txt"))

    def test_second_writer_and_process_are_refused(self):
        with catalog_writer(self.store.path):
            with self.assertRaises(CatalogBusy):
                with catalog_writer(self.store.path):
                    pass
            code = "from pathlib import Path; from vfx_element_tagger.catalog_lock import catalog_writer; " \
                   "\nwith catalog_writer(Path(__import__('sys').argv[1])): pass"
            result = subprocess.run([sys.executable, "-c", code, str(self.store.path)],
                                    capture_output=True, env={**__import__('os').environ, "PYTHONPATH": str(ROOT / "src")})
            self.assertNotEqual(result.returncode, 0)
            with self.assertRaises(CatalogBusy):
                self.jobs.start({"kind": "ingest", "folder": str(self.folder), "analyse": False})

    def test_actual_background_import_preserves_catalog_and_creates_backup(self):
        self.store.rate_element("one", "artist", 5)
        self.jobs.start({"kind": "ingest", "folder": str(self.folder), "analyse": False})
        self.jobs._thread.join(timeout=15)
        self.assertFalse(self.jobs._thread.is_alive())
        result = self.jobs.snapshot()["job"]
        self.assertEqual(result["state"], "completed", result)
        self.assertEqual(self.store.load_one("one").caption, "Artist choice")
        self.assertEqual(self.store.rating_summary("one")["rating_count"], 1)
        backup = LibraryStore(Path(result["backup"]))
        self.assertEqual(backup.load_one("one").human_overrides, example().human_overrides)
        self.assertEqual(backup.rating_summary("one")["rating_count"], 1)

    def test_force_confirmation_and_inputs(self):
        for payload in ({"kind": "force"}, {"kind": "force", "confirmed": True, "element_ids": ["missing"]},
                        {"kind": "force", "confirmed": True, "scope": "unknown", "element_ids": ["one"]},
                        {"kind": "pending", "limit": True},
                        {"kind": "ingest", "folder": str(self.folder), "default_fps": 0},
                        {"kind": "ingest", "folder": str(self.folder), "analyse": "yes"}):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                self.jobs.start(payload)
        with self.assertRaises(ValueError):
            self.jobs.start({"kind": "pending", "limit": 1})

    def request(self, **changes):
        return {"library": str(self.store.path), "cache": str(self.root / "cache"),
                "manifest": str(self.root / "manifest.json"), "cancel_path": str(self.root / "cancel"),
                "kind": "pending", "limit": 20, **changes}

    def test_empty_json_library_cannot_start_pending(self):
        jobs = LibraryJobs(self.root / "new.json", self.root / "cache")
        with self.assertRaisesRegex(ValueError, "Add elements"):
            jobs.start({"kind": "pending"})
        self.assertFalse(jobs.library.exists())

    def test_import_analysis_only_receives_scanned_ids(self):
        result = SimpleNamespace(scanned_element_ids=["new-asset"], failed_references=0)
        with patch("vfx_element_tagger.pipeline.run_ingest", return_value=result) as ingest, \
             patch("vfx_element_tagger.analysis_runner.main", return_value=0) as analyze, \
             redirect_stdout(io.StringIO()):
            code = worker(self.request(kind="ingest", folder=str(self.folder), default_fps=24,
                                       artifacts_enabled=False, analyse=True))
        self.assertEqual(code, 0)
        self.assertTrue(ingest.call_args.kwargs["writer_locked"])
        argv = analyze.call_args.args[0]
        self.assertEqual(argv[argv.index("--element-id") + 1], "new-asset")
        self.assertNotIn("one", argv)
        self.assertNotIn("--force", argv)
        self.assertNotIn("--allow-auto-accept", argv)

    def test_force_scopes_and_human_data_are_preserved(self):
        for ids, limit in ((["one"], 1), ([], 20)):
            with self.subTest(ids=ids), patch("vfx_element_tagger.analysis_runner.main", return_value=0) as analyze, \
                 redirect_stdout(io.StringIO()):
                code = worker(self.request(kind="force", element_ids=ids, limit=limit))
            self.assertEqual(code, 0)
            argv = analyze.call_args.args[0]
            self.assertIn("--force", argv)
            self.assertEqual(argv.count("--element-id"), len(ids))
            self.assertEqual(argv[argv.index("--limit") + 1], str(limit))
            self.assertNotIn("--allow-auto-accept", argv)
        self.assertEqual(self.store.load_one("one").human_overrides, example().human_overrides)

    def test_cancel_keeps_checkpoint_and_failure_is_reported(self):
        cancel = self.root / "cancel"
        def checkpoint(argv, *, progress, should_stop, writer_locked):
            item = self.store.load_one("one")
            item.caption = "Completed checkpoint"
            self.store.save_element(item)
            cancel.touch()
            self.assertTrue(should_stop())
            progress({"phase": "analysis", "completed": 1, "total": 2, "errors": 0})
            return 130
        output = io.StringIO()
        with patch("vfx_element_tagger.analysis_runner.main", side_effect=checkpoint), redirect_stdout(output):
            self.assertEqual(worker(self.request()), 130)
        self.assertEqual(self.store.load_one("one").caption, "Completed checkpoint")
        self.assertEqual(self.store.load_one("one").human_overrides, example().human_overrides)
        cancel.unlink()
        output = io.StringIO()
        with patch("vfx_element_tagger.analysis_runner.main", side_effect=RuntimeError("Model failed")), \
             redirect_stdout(output):
            self.assertEqual(worker(self.request()), 1)
        self.assertIn("Model failed", output.getvalue())
        self.assertEqual(self.store.load_one("one").caption, "Completed checkpoint")

    def test_single_job_and_matching_cancel_only(self):
        self.jobs._job = {"id": "active", "state": "running"}
        with self.assertRaises(CatalogBusy):
            self.jobs.start({"kind": "ingest", "folder": str(self.folder), "analyse": False})
        with self.assertRaises(ValueError):
            self.jobs.cancel("wrong")
        self.jobs.directory.mkdir(parents=True)
        self.assertEqual(self.jobs.cancel("active")["job"]["state"], "cancelling")
        self.assertTrue((self.jobs.directory / "active.cancel").is_file())
        self.jobs._job["state"] = "cancelled"

    def test_cancelled_worker_does_not_write_catalog(self):
        cancel = self.root / "cancel"
        cancel.touch()
        before = self.store.load_one("one").to_dict()
        code = worker({"library": str(self.store.path), "cache": str(self.root / "cache"),
                       "cancel_path": str(cancel)})
        self.assertEqual(code, 130)
        self.assertEqual(self.store.load_one("one").to_dict(), before)

    def test_controls_and_stale_review_token(self):
        html = index_html([])
        self.assertIn('id="add-elements"', html)
        self.assertIn('id="process-pending"', html)
        self.assertIn('id="force-confirm" type="checkbox" required', html)
        self.assertIn("review_version:x.review_version", review_html())
        item = example()
        version = _review_version(item)
        item.caption = "changed"
        self.assertNotEqual(version, _review_version(item))


class JobHTTPTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.state = QCServer("127.0.0.1", 0, root / "library.sqlite3", root / "cache",
                              models_manifest=root / "missing.json")
        self.state.store.save([example()])
        self.state.refresh_elements()
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), type("Handler", (QCRequestHandler,), {"state": self.state}))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.state.jobs.shutdown()
        self.tmp.cleanup()

    def request(self, method, path, payload=None, extra=None):
        port = self.server.server_address[1]
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        headers = {"Content-Type": "application/json", **(extra or {})}
        connection.request(method, path, json.dumps(payload) if payload is not None else None, headers)
        response = connection.getresponse()
        result = response.status, json.loads(response.read())
        connection.close()
        return result

    def test_jobs_and_folders_keep_local_origin_guard(self):
        self.assertEqual(self.request("GET", "/api/jobs")[0], 200)
        self.assertEqual(self.request("GET", "/api/folders", extra={"Origin": "https://attacker.example"})[0], 403)
        self.assertEqual(self.request("POST", "/api/jobs", {"kind": "force"})[0], 400)

    def test_ratings_and_reruns_block_while_job_active_but_reads_work(self):
        self.state.jobs._job = {"id": "active", "state": "running"}
        self.assertEqual(self.request("GET", "/api/elements")[0], 200)
        self.assertEqual(self.request("POST", "/api/ratings/one", {"rater_id": "artist", "rating": 5})[0], 409)
        self.assertEqual(self.request("POST", "/api/rerun/one/metadata", {})[0], 409)
        self.state.jobs._job["state"] = "cancelled"

    def test_stale_review_rejected_without_changes(self):
        self.assertEqual(self.request("POST", "/api/review/one", {"fields": {"primary_family": "fire"}, "review_version": "old"})[0], 409)
        self.assertEqual(self.state.store.load_one("one").human_overrides, example().human_overrides)
