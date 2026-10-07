from copy import deepcopy
import http.client
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_gated import _apply_outcome
from vfx_element_tagger.models import Element, SourceRepresentation, ProcessingEvent
from vfx_element_tagger.pipeline import _guard_regrouping, _preserve_analysis
from vfx_element_tagger.store import LibraryStore
from vfx_element_tagger.web import QCRequestHandler, QCServer
from vfx_element_tagger.model_integrity import sha256_file, verify_snapshot
from unittest.mock import patch
from vfx_element_tagger.proxy import build_elements_from_representations
from vfx_element_tagger.pipeline import run_ingest


def element():
    return Element("one", [SourceRepresentation("rep", "/demo/smoke.mov", "/demo/smoke.mov", "mov")], "rep")


class PreservationTests(unittest.TestCase):
    def test_real_ingest_refresh_preserves_sqlite_rating(self):
        with tempfile.TemporaryDirectory() as tmp:
            rep = element().primary()
            old = build_elements_from_representations([deepcopy(rep)])[0]
            old.semantic_analysis = {"summary": "approved"}
            old.human_overrides = {"fields": {"motion.speed": "slow"}}
            old.analysis_status = "accepted"
            store = LibraryStore(Path(tmp) / "catalog.sqlite3")
            store.save([old])
            store.rate_element(old.element_id, "artist", 5)
            with patch("vfx_element_tagger.pipeline.discover_media", return_value=[object()]), \
                 patch("vfx_element_tagger.pipeline.discover_sidecar_manifests", return_value=[]), \
                 patch("vfx_element_tagger.pipeline.build_source_representation", return_value=deepcopy(rep)), \
                 patch("vfx_element_tagger.pipeline.generate_representation_phash", return_value=None):
                result = run_ingest(Path(tmp), Path(tmp)/"cache", store.path,
                                    artifacts_enabled=False, force_recompute=True)
            self.assertEqual(result.processed_elements, 1)
            restored = store.load_one(old.element_id)
            self.assertEqual(restored.semantic_analysis, old.semantic_analysis)
            self.assertEqual(restored.human_overrides, old.human_overrides)
            self.assertEqual(store.rating_summary(old.element_id)["rating_count"], 1)

    def test_force_refresh_preserves_artist_history_and_analysis(self):
        old = element()
        old.caption = "Artist-approved smoke"
        old.semantic_analysis = {"summary": old.caption}
        old.analysis_status = "accepted"
        old.human_overrides = {"fields": {"motion.speed": "slow"}}
        old.analysis_escalation = {"human_review_history": [{"note": "keep"}], "analysis_signature": "old"}
        new = element()
        new.poster_path = "/new/poster.jpg"
        _preserve_analysis(old, new)
        self.assertEqual(new.caption, old.caption)
        self.assertEqual(new.analysis_status, "accepted")
        self.assertEqual(new.human_overrides, old.human_overrides)
        self.assertEqual(new.analysis_escalation, old.analysis_escalation)
        self.assertEqual(new.poster_path, "/new/poster.jpg")
        new.human_overrides["fields"].clear()
        self.assertTrue(old.human_overrides["fields"])

    def test_changed_source_invalidates_predictions_not_artist_data(self):
        old = element()
        old.semantic_analysis = {"action_timing": {"start": 1}}
        old.analysis_status = "accepted"
        old.image_embed = [1.0]
        old.human_overrides = {"fields": {"motion.speed": "slow"}}
        old.analysis_escalation = {"analysis_signature": "old", "human_review_history": [1]}
        new = element()
        new.primary().content_fingerprint = "changed"
        new.duration_seconds = 4
        _preserve_analysis(old, new)
        self.assertEqual(new.analysis_status, "needs_review")
        self.assertNotIn("analysis_signature", new.analysis_escalation)
        self.assertEqual(new.analysis_escalation["human_review_history"], [1])
        self.assertEqual(new.human_overrides, old.human_overrides)
        self.assertEqual(new.image_embed, [])
        self.assertNotIn("action_timing", new.semantic_analysis)

    def test_ambiguous_regrouping_refuses_merge_and_split(self):
        old = element()
        new = deepcopy(old)
        new.element_id = "regrouped"
        with self.assertRaisesRegex(ValueError, "Catalog was not written"):
            _guard_regrouping([old], [new])
        _guard_regrouping([old], [deepcopy(old)])

    def test_gate_approval_is_not_artist_approval(self):
        outcome = SimpleNamespace(analysis={"primary_family": "smoke"}, candidates=[], confidence=.99,
                                  status="accepted", model_version="demo", schema_version="demo", escalation={})
        item = element()
        _apply_outcome(item, outcome, {})
        self.assertEqual(item.analysis_status, "needs_review")
        self.assertTrue(item.analysis_escalation["human_review_required"])
        _apply_outcome(item, outcome, {}, allow_auto_accept=True)
        self.assertEqual(item.analysis_status, "accepted")


class BackupTests(unittest.TestCase):
    def test_sqlite_restore_preserves_payload_ratings_and_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = LibraryStore(Path(tmp) / "live.sqlite3")
            item = element()
            item.human_overrides = {"fields": {"primary_family": "smoke"}}
            store.save([item])
            store.rate_element("one", "demo", 5)
            store.append_events([ProcessingEvent("one", "test", "success")])
            backup = store.backup(Path(tmp) / "restored.sqlite3")
            restored = LibraryStore(backup)
            self.assertEqual(restored.load()[0].to_dict(), item.to_dict())
            self.assertEqual(restored.rating_summary("one"), store.rating_summary("one"))
            with sqlite3.connect(backup) as connection:
                self.assertEqual(connection.execute("SELECT count(*) FROM processing_events").fetchone()[0], 1)
            with self.assertRaises(ValueError):
                store.backup(store.path)
            with self.assertRaises(FileExistsError):
                store.backup(backup)


class ModelIntegrityTests(unittest.TestCase):
    def test_modified_and_extra_weights_are_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            weights = directory / "model.safetensors"
            weights.write_bytes(b"test")
            lock = directory / "lock.json"
            lock.write_text(json.dumps({"models": {"test": {"revision": "a"*40, "files": {
                weights.name: {"size": 4, "sha256": sha256_file(weights)}}}}}))
            with patch("vfx_element_tagger.model_integrity.LOCK_PATH", lock):
                self.assertEqual(verify_snapshot("test", directory)["revision"], "a"*40)
                weights.write_bytes(b"fake")
                with self.assertRaisesRegex(ValueError, "checksum"):
                    verify_snapshot("test", directory)
                weights.write_bytes(b"test")
                (directory / "extra.safetensors").write_bytes(b"unexpected")
                with self.assertRaisesRegex(ValueError, "Unexpected"):
                    verify_snapshot("test", directory)


class WebSafetyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        store = LibraryStore(Path(self.tmp.name) / "library.sqlite3")
        store.save([element()])
        state = SimpleNamespace(store=store, elements=store.load(), cache_dir=Path(self.tmp.name),
                                library_path=store.path, lock=threading.Lock(), refresh_elements=lambda: None)
        handler = type("Handler", (QCRequestHandler,), {"state": state})
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.port = self.server.server_address[1]

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.tmp.cleanup()

    def request(self, method, path, body=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=3)
        connection.request(method, path, body=body, headers=headers or {})
        response = connection.getresponse()
        result = (response.status, dict(response.getheaders()), response.read())
        connection.close()
        return result

    def test_local_read_has_security_headers(self):
        status, headers, _ = self.request("GET", "/api/elements")
        self.assertEqual(status, 200)
        self.assertEqual(headers["X-Frame-Options"], "DENY")

    def test_rebinding_and_cross_site_requests_rejected(self):
        for headers in ({"Host": "attacker.example"}, {"Origin": "https://attacker.example"},
                        {"Sec-Fetch-Site": "cross-site"}, {"Content-Length": "-1"},
                        {"Transfer-Encoding": "chunked"}):
            self.assertEqual(self.request("POST", "/api/ratings/one", headers=headers)[0], 403)

    def test_valid_json_rating_and_bad_bodies(self):
        data = json.dumps({"rater_id": "demo", "rating": 5})
        headers = {"Content-Type": "application/json", "Origin": f"http://127.0.0.1:{self.port}"}
        self.assertEqual(self.request("POST", "/api/ratings/one", data, headers)[0], 200)
        self.assertEqual(self.request("POST", "/api/ratings/one", data)[0], 400)
        self.assertEqual(self.request("POST", "/api/compare/one", '{"vote":"invalid"}', headers)[0], 400)
        self.assertEqual(self.request("GET", "/api/search?count=oops")[0], 400)

    def test_nonlocal_binding_refused(self):
        with self.assertRaisesRegex(ValueError, "local-only"):
            QCServer("0.0.0.0", 8765, Path(self.tmp.name) / "other.sqlite3", Path(self.tmp.name))
