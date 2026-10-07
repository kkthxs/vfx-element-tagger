from __future__ import annotations

import csv
import gzip
import hashlib
import json
import ipaddress
import mimetypes
import threading
import signal
from contextlib import nullcontext
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, unquote, urlparse

from .artifacts import generate_alpha_preview
from .models import Element
from .pipeline import rerun_stage
from .review import build_review_context, review_element
from .search import prewarm_ai_query_models, search_elements
from .store import LibraryStore
from .taxonomy import load_taxonomy
from .review_ui import review_html
from .web_ui import index_html
from .catalog_lock import CatalogBusy, catalog_writer
from .library_jobs import LibraryJobs, browse_folders


def _compare_path(library_path: Path) -> Path:
    return library_path.with_suffix(".compare.json")


def _compare_vote_log(library_path: Path) -> Path:
    return library_path.with_suffix(".compare-votes.csv")


def _load_compare(library_path: Path) -> dict:
    path = _compare_path(library_path)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _append_vote(library_path: Path, payload: dict) -> None:
    log_path = _compare_vote_log(library_path)
    is_new = not log_path.exists()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        if is_new:
            writer.writerow(["timestamp", "element_id", "model_a", "model_b", "vote"])
        writer.writerow(
            [
                datetime.now(tz=timezone.utc).isoformat(),
                _csv_cell(payload.get("element_id", "")),
                _csv_cell(payload.get("model_a", "")),
                _csv_cell(payload.get("model_b", "")),
                payload.get("vote", ""),
            ]
        )


def _csv_cell(value: Any) -> str:
    text = str(value)
    return "'" + text if text.lstrip().startswith(("=", "+", "-", "@")) else text


class QCServer:
    def __init__(
        self,
        host: str,
        port: int,
        library_path: Path,
        cache_dir: Path,
        artifacts_enabled: bool = True,
        demo: bool = False,
        models_manifest: Path | None = None,
    ):
        if host != "localhost":
            try:
                loopback = ipaddress.ip_address(host).is_loopback
            except ValueError:
                loopback = False
            if not loopback:
                raise ValueError("v1 is local-only: bind to localhost or a loopback IP address")
        self.host = host
        self.port = port
        self.library_path = library_path
        self.cache_dir = cache_dir
        self.artifacts_enabled = artifacts_enabled
        self.demo = demo
        self.store = LibraryStore(library_path)
        self.elements = self.store.load()
        self.catalog_revision = self.store.revision()
        self.lock = threading.RLock()
        self.mutation_lock = threading.RLock()
        self.jobs = LibraryJobs(library_path, cache_dir, models_manifest, artifacts_enabled)
        self._prewarm_ai_search()

    def serve_forever(self) -> None:
        server_state = self

        class Handler(QCRequestHandler):
            state = server_state

        httpd = ThreadingHTTPServer((self.host, self.port), Handler)
        self.port = httpd.server_address[1]
        print(f"QC view: http://{self.host}:{self.port}")
        previous = None
        if threading.current_thread() is threading.main_thread():
            def stop_server(*_):
                raise KeyboardInterrupt()
            previous = signal.signal(signal.SIGTERM, stop_server)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            self.jobs.shutdown()
            httpd.server_close()
            if previous is not None:
                signal.signal(signal.SIGTERM, previous)

    def _prewarm_ai_search(self) -> None:
        if not self.elements:
            return
        thread = threading.Thread(
            target=prewarm_ai_query_models,
            args=(self.elements,),
            daemon=True,
        )
        thread.start()

    def refresh_elements(self) -> None:
        """Notice catalog writes made by background workers without restarting."""

        revision = self.store.revision()
        if revision == self.catalog_revision:
            return
        with self.lock:
            revision = self.store.revision()
            if revision == self.catalog_revision:
                return
            self.elements = self.store.load()
            self.catalog_revision = revision


class QCRequestHandler(BaseHTTPRequestHandler):
    state: QCServer
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args) -> None:
        return

    def end_headers(self) -> None:
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' data:; "
                         "media-src 'self' blob:; style-src 'self' 'unsafe-inline'; "
                         "script-src 'self' 'unsafe-inline'; frame-ancestors 'none'; base-uri 'none'")
        super().end_headers()

    def _validate_request(self) -> bool:
        hosts = self.headers.get_all("Host", [])
        port = self.server.server_address[1]
        allowed = {f"127.0.0.1:{port}", f"localhost:{port}", f"[::1]:{port}"}
        if port == 80:
            allowed.update({"127.0.0.1", "localhost", "[::1]"})
        host = hosts[0].lower() if len(hosts) == 1 else ""
        origin = self.headers.get("Origin")
        referer = self.headers.get("Referer")
        invalid = (host not in allowed or not self.path.startswith("/") or self.path.startswith("//")
                   or self.headers.get("Sec-Fetch-Site") == "cross-site"
                   or (origin is not None and origin.lower() != f"http://{host}")
                   or (referer is not None and
                       (urlparse(referer).scheme.lower() != "http" or urlparse(referer).netloc.lower() != host)))
        lengths = self.headers.get_all("Content-Length", [])
        if self.headers.get("Transfer-Encoding") or len(lengths) > 1:
            invalid = True
        if lengths:
            try:
                length = int(lengths[0])
                invalid = invalid or length < 0 or length > 1024 * 1024
            except ValueError:
                invalid = True
        if invalid:
            self.close_connection = True
            self._send_error(HTTPStatus.FORBIDDEN, "Invalid local request origin, host or framing")
            return False
        if self.command == "POST":
            self.state.refresh_elements()
        return True

    def do_GET(self) -> None:
        if not self._validate_request():
            return
        parsed = urlparse(self.path)
        if parsed.path == "/":
            html = index_html(load_taxonomy().categories)
            if getattr(self.state, "demo", False):
                html = html.replace("Semantic catalog</span>", "Synthetic demo · Curated labels</span>")
            self._send_html(html)
            return
        if parsed.path == "/review":
            html = review_html()
            if getattr(self.state, "demo", False):
                html = html.replace("Human confirmation for uncertain analyses", "Synthetic demo · Curated review examples")
            self._send_html(html)
            return
        if parsed.path.startswith("/api/"):
            self.state.refresh_elements()
        if parsed.path == "/api/jobs":
            self._send_json(self.state.jobs.snapshot())
            return
        if parsed.path == "/api/folders":
            try:
                value = parse_qs(parsed.query).get("path", [None])[0]
                self._send_json(browse_folders(value))
            except (OSError, ValueError) as exc:
                self._send_error(HTTPStatus.BAD_REQUEST, str(exc))
            return
        if parsed.path == "/api/review":
            self._send_json(
                [
                    _review_detail(element, self.state.cache_dir)
                    for element in self.state.elements
                    if element.analysis_status == "needs_review"
                ]
            )
            return
        if parsed.path == "/api/elements":
            query = parse_qs(parsed.query)
            rater_id = query.get("rater_id", [""])[0]
            ratings = self.state.store.rating_summaries(rater_id=rater_id or None)
            self._send_json(
                [
                    _element_summary(
                        element,
                        self.state.cache_dir,
                        ratings.get(element.element_id),
                    )
                    for element in self.state.elements
                ]
            )
            return
        if parsed.path.startswith("/api/alpha-preview/"):
            element_id = unquote(parsed.path.removeprefix("/api/alpha-preview/"))
            element = _find_element(self.state.elements, element_id)
            if not element:
                self._send_error(HTTPStatus.NOT_FOUND, "Element not found")
                return
            if not element.has_alpha_channel or not element.alpha_non_empty:
                self._send_error(HTTPStatus.NOT_FOUND, "Element has no visible alpha channel")
                return
            try:
                with catalog_writer(self.state.library_path), self.state.lock:
                    alpha_path = generate_alpha_preview(
                        element.element_id, element.primary(), self.state.cache_dir,
                        enabled=self.state.artifacts_enabled,
                    )
            except CatalogBusy as exc:
                self._send_error(HTTPStatus.CONFLICT, str(exc))
                return
            alpha_url = artifact_url(alpha_path, self.state.cache_dir)
            if not alpha_url:
                self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, "Alpha preview generation failed")
                return
            self._serve_artifact(alpha_url)
            return
        if parsed.path.startswith("/api/elements/"):
            element_id = unquote(parsed.path.rsplit("/", 1)[-1])
            element = _find_element(self.state.elements, element_id)
            if not element:
                self._send_error(HTTPStatus.NOT_FOUND, "Element not found")
                return
            query = parse_qs(parsed.query)
            rater_id = query.get("rater_id", [""])[0]
            rating = self.state.store.rating_summary(element_id, rater_id or None)
            self._send_json(_element_detail(element, self.state.cache_dir, rating))
            return
        if parsed.path == "/api/search":
            query = parse_qs(parsed.query)
            text = query.get("q", [""])[0]
            try:
                count = int(query.get("count", ["20"])[0] or "20")
                if not 1 <= count <= 200:
                    raise ValueError()
            except ValueError:
                self._send_error(HTTPStatus.BAD_REQUEST, "count must be an integer from 1 to 200")
                return
            filters = {
                key.removeprefix("filter_"): values[0]
                for key, values in query.items()
                if key.startswith("filter_") and values and values[0]
            }
            hits = search_elements(self.state.elements, text, count=count, filters=filters)
            rater_id = query.get("rater_id", [""])[0]
            ratings = self.state.store.rating_summaries(rater_id=rater_id or None)
            hits = sorted(
                hits,
                key=lambda hit: (
                    hit.score * 0.94
                    + (ratings.get(hit.element.element_id, {}).get("rating_quality", 0.0) / 5.0)
                    * 0.06
                ),
                reverse=True,
            )
            self._send_json(
                [
                    {
                        **_element_summary(
                            hit.element,
                            self.state.cache_dir,
                            ratings.get(hit.element.element_id),
                        ),
                        "score": hit.score,
                        "reasons": hit.reasons,
                        "explain": hit.explain,
                    }
                    for hit in hits
                ]
            )
            return
        if parsed.path.startswith("/api/compare/"):
            element_id = unquote(parsed.path.rsplit("/", 1)[-1])
            element = _find_element(self.state.elements, element_id)
            if not element:
                self._send_error(HTTPStatus.NOT_FOUND, "Element not found")
                return
            compare = _load_compare(self.state.library_path)
            self._send_json(_compare_payload(element, compare))
            return
        if parsed.path.startswith("/artifacts/"):
            self._serve_artifact(parsed.path)
            return
        self._send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:
        if not self._validate_request():
            return
        try:
            with getattr(self.state, "mutation_lock", nullcontext()):
                if urlparse(self.path).path.startswith("/api/jobs"):
                    self._handle_job_post()
                    return
                jobs = getattr(self.state, "jobs", None)
                if jobs and jobs.busy:
                    raise CatalogBusy("Library job running. Artist saves are paused until it finishes.")
                with catalog_writer(self.state.library_path):
                    self.state.refresh_elements()
                    self._handle_post()
        except CatalogBusy as exc:
            self.close_connection = True
            self._send_error(HTTPStatus.CONFLICT, str(exc))

    def _handle_job_post(self):
        try:
            path = urlparse(self.path).path
            payload = self._read_json_body()
            if path == "/api/jobs":
                self._send_json(self.state.jobs.start(payload))
            elif path == "/api/jobs/cancel":
                self._send_json(self.state.jobs.cancel(payload.get("id")))
            else:
                self._send_error(HTTPStatus.NOT_FOUND, "Not found")
        except (OSError, ValueError, UnicodeError) as exc:
            self._send_error(HTTPStatus.BAD_REQUEST, str(exc))

    def _handle_post(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path.startswith("/api/ratings/"):
            element_id = unquote(parsed.path.rsplit("/", 1)[-1])
            try:
                payload = self._read_json_body()
                rating = self.state.store.rate_element(
                    element_id,
                    str(payload.get("rater_id") or ""),
                    payload.get("rating"),
                    str(payload.get("rater_name") or ""),
                )
            except (json.JSONDecodeError, UnicodeDecodeError):
                self._send_error(HTTPStatus.BAD_REQUEST, "Rating body must be JSON")
                return
            except KeyError:
                self._send_error(HTTPStatus.NOT_FOUND, "Element not found")
                return
            except ValueError as exc:
                self._send_error(HTTPStatus.BAD_REQUEST, str(exc))
                return
            self._send_json({"ok": True, "element_id": element_id, **rating})
            return
        if parsed.path.startswith("/api/review/"):
            element_id = unquote(parsed.path.rsplit("/", 1)[-1])
            element = _find_element(self.state.elements, element_id)
            if not element:
                self._send_error(HTTPStatus.NOT_FOUND, "Element not found")
                return
            try:
                payload = self._read_json_body()
                if payload.get("review_version") and payload["review_version"] != _review_version(element):
                    raise CatalogBusy("This analysis changed. Reload the review queue before saving.")
                with self.state.lock:
                    record = review_element(
                        element,
                        payload.get("fields") or {},
                        note=str(payload.get("note") or ""),
                    )
                    self.state.store.save_element(element)
                    self.state.catalog_revision = self.state.store.revision()
            except (json.JSONDecodeError, UnicodeDecodeError):
                self._send_error(HTTPStatus.BAD_REQUEST, "Review body must be JSON")
                return
            except ValueError as exc:
                self._send_error(HTTPStatus.BAD_REQUEST, str(exc))
                return
            self._send_json(
                {
                    "ok": True,
                    "review": record,
                    "element": _element_detail(element, self.state.cache_dir),
                    "remaining": sum(
                        item.analysis_status == "needs_review" for item in self.state.elements
                    ),
                }
            )
            return
        if parsed.path.startswith("/api/rerun/"):
            parts = parsed.path.strip("/").split("/")
            if len(parts) != 4:
                self._send_error(HTTPStatus.BAD_REQUEST, "Expected /api/rerun/<element_id>/<stage>")
                return
            _, _, element_id, stage = parts
            element = _find_element(self.state.elements, unquote(element_id))
            if not element:
                self._send_error(HTTPStatus.NOT_FOUND, "Element not found")
                return
            try:
                rerun_stage(element, unquote(stage), self.state.cache_dir, self.state.artifacts_enabled)
            except ValueError as exc:
                self._send_error(HTTPStatus.BAD_REQUEST, str(exc))
                return
            with self.state.lock:
                self.state.store.save_element(element)
                self.state.catalog_revision = self.state.store.revision()
            self._send_json(_element_detail(element, self.state.cache_dir))
            return
        if parsed.path.startswith("/api/compare/"):
            element_id = unquote(parsed.path.rsplit("/", 1)[-1])
            element = _find_element(self.state.elements, element_id)
            if not element:
                self._send_error(HTTPStatus.NOT_FOUND, "Element not found")
                return
            try:
                vote_payload = self._read_json_body()
                if vote_payload.get("vote") not in {"a", "b", "tie"}:
                    raise ValueError("vote must be a, b or tie")
            except (ValueError, UnicodeDecodeError) as exc:
                self._send_error(HTTPStatus.BAD_REQUEST, str(exc))
                return
            compare = _load_compare(self.state.library_path)
            payload = _compare_payload(element, compare)
            payload.update(
                {
                    "element_id": element.element_id,
                    "vote": vote_payload.get("vote", ""),
                }
            )
            with self.state.lock:
                _append_vote(self.state.library_path, payload)
            self._send_json({"ok": True, "vote": payload["vote"]})
            return
        self._send_error(HTTPStatus.NOT_FOUND, "Not found")

    def _read_json_body(self) -> dict:
        if self.headers.get_content_type() != "application/json":
            self.close_connection = True
            raise ValueError("Request body must have Content-Type: application/json")
        length = int(self.headers.get("Content-Length", "0") or 0)
        if not 0 <= length <= 1024 * 1024:
            raise ValueError("Request body is too large")
        body = self.rfile.read(length).decode("utf-8") if length else "{}"
        payload = json.loads(body or "{}")
        if not isinstance(payload, dict):
            raise ValueError("Request body must be a JSON object")
        return payload

    def _serve_artifact(self, request_path: str) -> None:
        relative = request_path.removeprefix("/artifacts/")
        target = (self.state.cache_dir / "artifacts" / relative).resolve()
        artifact_root = (self.state.cache_dir / "artifacts").resolve()
        if artifact_root not in target.parents and target != artifact_root:
            self._send_error(HTTPStatus.FORBIDDEN, "Forbidden")
            return
        if not target.exists() or not target.is_file():
            self._send_error(HTTPStatus.NOT_FOUND, "Artifact not found")
            return
        content_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        size = target.stat().st_size
        start, end = 0, max(0, size - 1)
        range_header = self.headers.get("Range", "")
        if range_header:
            try:
                start, end = _parse_byte_range(range_header, size)
            except ValueError as exc:
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Type", "application/json; charset=utf-8")
                data = json.dumps({"error": str(exc)}).encode("utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
        length = max(0, end - start + 1)
        self.send_response(HTTPStatus.PARTIAL_CONTENT if range_header else HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "private, max-age=3600")
        if range_header:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(length))
        self.end_headers()
        with target.open("rb") as handle:
            handle.seek(start)
            remaining = length
            while remaining and (chunk := handle.read(min(1024 * 256, remaining))):
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def _send_html(self, html: str) -> None:
        self._send_payload(html.encode("utf-8"), "text/html; charset=utf-8")

    def _send_json(self, payload) -> None:
        data = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        self._send_payload(data, "application/json; charset=utf-8")

    def _send_payload(self, data: bytes, content_type: str) -> None:
        etag = '"' + hashlib.sha256(data).hexdigest() + '"'
        if self.headers.get("If-None-Match") == etag:
            self.send_response(HTTPStatus.NOT_MODIFIED)
            self.send_header("ETag", etag)
            self.send_header("Cache-Control", "private, no-cache")
            self.end_headers()
            return
        compressed = "gzip" in self.headers.get("Accept-Encoding", "").lower()
        body = gzip.compress(data, compresslevel=5) if compressed and len(data) > 1024 else data
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("ETag", etag)
        self.send_header("Cache-Control", "private, no-cache")
        self.send_header("Vary", "Accept-Encoding")
        if body is not data:
            self.send_header("Content-Encoding", "gzip")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, status: HTTPStatus, message: str) -> None:
        data = json.dumps({"error": message}).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def artifact_url(path: str | None, cache_dir: Path) -> str | None:
    if not path:
        return None
    try:
        relative = Path(path).resolve().relative_to((cache_dir / "artifacts").resolve())
    except ValueError:
        return None
    return "/artifacts/" + "/".join(relative.parts)


def _parse_byte_range(value: str, size: int) -> tuple[int, int]:
    """Parse one RFC 7233 byte range for efficient movie preview streaming."""

    if size <= 0 or not value.startswith("bytes=") or "," in value:
        raise ValueError("Unsupported byte range")
    spec = value.removeprefix("bytes=").strip()
    if "-" not in spec:
        raise ValueError("Invalid byte range")
    start_text, end_text = spec.split("-", 1)
    try:
        if not start_text:
            suffix = int(end_text)
            if suffix <= 0:
                raise ValueError
            start = max(0, size - suffix)
            end = size - 1
        else:
            start = int(start_text)
            end = int(end_text) if end_text else size - 1
    except ValueError as exc:
        raise ValueError("Invalid byte range") from exc
    if start < 0 or start >= size or end < start:
        raise ValueError("Byte range is outside the artifact")
    return start, min(end, size - 1)


def _find_element(elements: list[Element], element_id: str) -> Element | None:
    return next((element for element in elements if element.element_id == element_id), None)


def _compare_payload(element: Element, compare: dict) -> dict:
    entry = (compare.get("elements") or {}).get(element.element_id) or {}
    return {
        "element_id": element.element_id,
        "model_a": compare.get("model_a_label", "current"),
        "model_b": compare.get("model_b_label", "alternate"),
        "caption_a": element.caption,
        "caption_b": entry.get("alternate_caption", ""),
        "available": bool(entry.get("alternate_caption")),
    }


def _element_summary(
    element: Element,
    cache_dir: Path,
    rating: dict[str, Any] | None = None,
) -> dict:
    semantic = element.semantic_analysis or {}
    composition = semantic.get("composition") or {}
    appearance = semantic.get("appearance") or {}
    motion = semantic.get("motion") or {}
    usability = semantic.get("usability") or {}
    action_timing = semantic.get("action_timing") or {}
    primary = element.primary()
    family = _proposed_display_family(element, primary)

    rating = rating or {
        "rating_average": 0.0,
        "rating_count": 0,
        "rating_quality": 0.0,
        "rating_distribution": {str(star): 0 for star in range(1, 6)},
        "user_rating": None,
    }
    return {
        "element_id": element.element_id,
        "name": Path(primary.original_path or primary.path).stem,
        "source_path": primary.original_path or primary.path,
        "source_format": primary.format,
        "category": element.category,
        "category_confidence": element.category_confidence,
        "caption": element.caption,
        "summary": semantic.get("summary") or element.caption,
        "search_text": semantic.get("search_text") or "",
        "primary_family": family,
        "secondary_families": element.secondary_families,
        "effect_type": semantic.get("effect_type") or "unclassified",
        "composition": composition,
        "appearance": appearance,
        "motion": motion,
        "usability": usability,
        "action_timing": action_timing,
        "analysis_status": element.analysis_status,
        "analysis_confidence": element.analysis_confidence,
        "analysis_model_version": element.analysis_model_version,
        "uncertainty": semantic.get("uncertainty") or [],
        "duration_seconds": element.duration_seconds,
        "width": element.width,
        "height": element.height,
        "fps": element.fps,
        "bit_depth": element.bit_depth,
        "color_space": element.color_space,
        "has_alpha_channel": element.has_alpha_channel,
        "alpha_non_empty": element.alpha_non_empty,
        "alpha_is_soft": element.alpha_is_soft,
        "is_sequence": element.is_sequence,
        "has_missing_frames": element.has_missing_frames,
        "poster_url": artifact_url(element.poster_path, cache_dir),
        "preview_480_url": artifact_url(element.preview_movie_480_path, cache_dir),
        "alpha_preview_url": (
            f"/api/alpha-preview/{quote(element.element_id, safe='')}"
            if element.has_alpha_channel and element.alpha_non_empty
            else None
        ),
        "near_duplicate_group_id": element.near_duplicate_group_id,
        **rating,
    }


def _proposed_display_family(element: Element, primary) -> str:
    """Supply a useful provisional shelf for records awaiting semantic analysis.

    Model-backed ``primary_family`` always wins.  For an unanalysed record, a
    conservative filename cue is preferable to an Unknown shelf and is clearly
    paired with the record's ``unanalysed`` review status in the UI.
    """

    if element.primary_family and element.primary_family != "unknown":
        return element.primary_family
    filename = Path(primary.original_path or primary.path).stem.lower()
    aliases = (
        (("snow",), "snow"),
        (("steam", "vapour", "vapor"), "smoke"),
        (("smoke", "smoky"), "smoke"),
        (("dust", "sand"), "dust"),
        (("lightning", "electric", "arc"), "electricity"),
        (("explosion", "fireball", "blast"), "explosion"),
        (("fire", "flame", "burning"), "fire"),
        (("debris", "shrapnel"), "debris"),
        (("rain",), "rain"),
        (("water", "splash", "droplet"), "water"),
        (("cloud",), "cloud"),
        (("bird",), "bird"),
    )
    for cues, family in aliases:
        if any(cue in filename for cue in cues):
            return family
    return element.category or "unknown"


def _element_detail(
    element: Element,
    cache_dir: Path,
    rating: dict[str, Any] | None = None,
) -> dict:
    data = element.to_dict()
    data.update(_element_summary(element, cache_dir, rating))
    return data


def _review_detail(element: Element, cache_dir: Path) -> dict:
    data = _element_summary(element, cache_dir)
    data["semantic_analysis"] = element.semantic_analysis
    data["human_overrides"] = element.human_overrides
    data["review_version"] = _review_version(element)
    data.update(build_review_context(element))
    return data


def _review_version(element: Element) -> str:
    return hashlib.sha256(json.dumps(element.to_dict(), sort_keys=True).encode()).hexdigest()


def _index_html(categories: list[str]) -> str:
    category_options = "\n".join(
        f'<option value="{category}">{category.replace("_", " ")}</option>' for category in categories
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>VFX Element QC</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --text: #151922;
      --muted: #647084;
      --line: #d9dee8;
      --accent: #146c94;
      --bad: #b42318;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    header {{
      position: sticky;
      top: 0;
      z-index: 2;
      display: grid;
      grid-template-columns: 1fr auto auto;
      gap: 12px;
      align-items: center;
      padding: 14px 18px;
      background: rgba(246, 247, 249, 0.96);
      border-bottom: 1px solid var(--line);
      backdrop-filter: blur(8px);
    }}
    input, select, button {{
      min-height: 36px;
      border: 1px solid var(--line);
      background: white;
      color: var(--text);
      border-radius: 6px;
      padding: 0 10px;
      font: inherit;
    }}
    button {{
      cursor: pointer;
      background: var(--accent);
      border-color: var(--accent);
      color: white;
      white-space: nowrap;
    }}
    main {{
      display: grid;
      grid-template-columns: minmax(320px, 420px) minmax(0, 1fr);
      min-height: calc(100vh - 65px);
    }}
    #list {{
      border-right: 1px solid var(--line);
      overflow: auto;
      background: #fff;
    }}
    .row {{
      display: grid;
      grid-template-columns: 96px 1fr;
      gap: 10px;
      padding: 10px;
      border-bottom: 1px solid var(--line);
      cursor: pointer;
    }}
    .row:hover, .row.active {{ background: #eef6fa; }}
    .thumb {{
      width: 96px;
      aspect-ratio: 16 / 9;
      object-fit: cover;
      background: #171a1f;
      border: 1px solid var(--line);
      border-radius: 4px;
    }}
    .row h3 {{
      margin: 0 0 3px;
      font-size: 14px;
      font-weight: 650;
    }}
    .meta {{ color: var(--muted); font-size: 12px; }}
    #detail {{
      padding: 18px;
      overflow: auto;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      margin-bottom: 14px;
      overflow: hidden;
    }}
    .panel h2 {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin: 0;
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      font-size: 14px;
      letter-spacing: 0;
    }}
    .panel .body {{ padding: 12px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    td {{ border-bottom: 1px solid #edf0f5; padding: 6px 4px; vertical-align: top; }}
    td:first-child {{ color: var(--muted); width: 210px; }}
    code {{
      background: #eef1f6;
      padding: 2px 5px;
      border-radius: 4px;
      word-break: break-all;
    }}
    pre {{
      white-space: pre-wrap;
      overflow: auto;
      background: #111827;
      color: #e5e7eb;
      border-radius: 6px;
      padding: 10px;
    }}
    .badge {{
      display: inline-block;
      margin: 2px 4px 2px 0;
      padding: 2px 6px;
      border-radius: 999px;
      background: #e9f2f6;
      color: #194b63;
      font-size: 12px;
    }}
    .search-debug {{
      display: flex;
      flex-wrap: wrap;
      gap: 4px;
      margin-top: 6px;
      color: var(--muted);
      font-size: 11px;
      line-height: 1.3;
    }}
    .debug-pill {{
      display: inline-flex;
      max-width: 100%;
      gap: 3px;
      padding: 1px 5px;
      border: 1px solid #e1e6ef;
      border-radius: 4px;
      background: #f7f9fc;
      color: #3f4a5f;
    }}
    .debug-label {{ color: var(--muted); }}
    .debug-term {{ color: var(--text); font-weight: 650; }}
    .src-ffprobe, .src-ffmpeg_alpha_probe {{ background: #d5f0d8; color: #1f6a36; }}
    .src-optical_flow {{ background: #dde6ff; color: #1f3680; }}
    .src-vlm, .src-siglip2 {{ background: #f3e0ff; color: #5e1f80; }}
    .src-human {{ background: #fff1c2; color: #6b5300; }}
    .src-filename, .src-filename_heuristic, .src-fallback {{ background: #ede5d5; color: #6b4d1a; }}
    .src-sidecar {{ background: #ffd6d6; color: #7a1f1f; }}
    .conf-dot {{
      display: inline-block;
      width: 6px;
      height: 6px;
      border-radius: 50%;
      margin-left: 4px;
      vertical-align: middle;
    }}
    .conf-high {{ background: #1f6a36; }}
    .conf-med {{ background: #c79100; }}
    .conf-low {{ background: #a32020; opacity: 0.7; }}
    .facet-row {{ display: flex; gap: 8px; align-items: baseline; flex-wrap: wrap; }}
    .facet-value {{ font-weight: 600; }}
    .compare-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
    .compare-card {{ border: 1px solid var(--line); border-radius: 6px; padding: 10px; background: #fbfcfe; }}
    .compare-card h4 {{ margin: 0 0 6px; font-size: 13px; }}
    .vote-row button {{ background: white; color: var(--text); border: 1px solid var(--line); }}
    .vote-row button.cast {{ background: var(--accent); color: white; border-color: var(--accent); }}
    video {{ width: min(720px, 100%); background: #111827; border-radius: 4px; }}
    @media (max-width: 860px) {{
      header {{ grid-template-columns: 1fr; }}
      main {{ grid-template-columns: 1fr; }}
      #list {{ border-right: 0; max-height: 45vh; }}
    }}
  </style>
</head>
<body>
  <header>
    <input id="query" placeholder="Search elements">
    <select id="category"><option value="">All categories</option>{category_options}</select>
    <button id="search">Search</button>
  </header>
  <main>
    <section id="list"></section>
    <section id="detail"><p class="meta">Select an element to inspect pipeline output.</p></section>
  </main>
  <script>
    const list = document.querySelector('#list');
    const detail = document.querySelector('#detail');
    const query = document.querySelector('#query');
    const category = document.querySelector('#category');
    let current = null;

    async function loadElements() {{
      const response = await fetch('/api/elements');
      renderList(await response.json());
    }}

    async function runSearch() {{
      const params = new URLSearchParams({{ q: query.value, count: '40' }});
      if (category.value) params.set('filter_category', category.value);
      const response = await fetch('/api/search?' + params.toString());
      renderList(await response.json());
    }}

    function renderList(items) {{
      list.innerHTML = items.map(item => `
        <article class="row ${{item.element_id === current ? 'active' : ''}}" data-id="${{item.element_id}}">
          ${{item.poster_url ? `<img class="thumb" src="${{item.poster_url}}" alt="">` : `<div class="thumb"></div>`}}
          <div>
            <h3>${{escapeHtml(item.category || 'unknown')}} <span class="meta">${{Math.round((item.category_confidence || 0) * 100)}}%</span></h3>
            <div class="meta">${{item.width || '?'}}x${{item.height || '?'}} · ${{formatDuration(item.duration_seconds)}} ${{item.score !== undefined ? `· score ${{item.score.toFixed(3)}}` : ''}}</div>
            <div>${{escapeHtml((item.caption || '').slice(0, 140))}}</div>
            ${{renderSearchExplain(item)}}
          </div>
        </article>
      `).join('');
      document.querySelectorAll('.row').forEach(row => row.addEventListener('click', () => loadDetail(row.dataset.id)));
    }}

    function renderSearchExplain(item) {{
      const explain = item.explain;
      if (!explain) return '';
      const signals = explain.signals || {{}};
      const signalHtml = ['image', 'caption', 'lexical'].map(name => {{
        const signal = signals[name] || {{}};
        const score = Number(signal.score || 0);
        if (!score) return '';
        const rank = signal.rank ? `#${{signal.rank}}` : '#-';
        return `<span class="debug-pill"><span class="debug-label">${{escapeHtml(name)}}</span><span class="debug-term">${{rank}}</span><span>${{score.toFixed(2)}}</span></span>`;
      }}).join('');
      const lexical = explain.lexical || {{}};
      const buckets = lexical.buckets || {{}};
      const bucketHtml = ['category', 'caption', 'filename', 'facets']
        .map(name => renderMatchBucket(name, buckets[name]))
        .join('');
      const facetHtml = Object.entries(lexical.facet_matches || {{}})
        .map(([name, terms]) => renderMatchBucket(name, terms))
        .join('');
      const rankHtml = explain.rank
        ? `<span class="debug-pill"><span class="debug-label">rank</span><span class="debug-term">${{explain.rank}}</span></span>`
        : '';
      const html = rankHtml + signalHtml + bucketHtml + facetHtml;
      return html ? `<div class="search-debug">${{html}}</div>` : '';
    }}

    function renderMatchBucket(name, terms) {{
      const values = Array.isArray(terms) ? terms : [];
      if (!values.length) return '';
      return `<span class="debug-pill"><span class="debug-label">${{escapeHtml(shortBucketName(name))}}</span>${{values.map(term => `<span class="debug-term">${{escapeHtml(term)}}</span>`).join(' ')}}</span>`;
    }}

    function shortBucketName(name) {{
      return {{
        category: 'cat',
        caption: 'cap',
        filename: 'file',
        facets: 'facet'
      }}[name] || name;
    }}

    async function loadDetail(id) {{
      current = id;
      const response = await fetch('/api/elements/' + encodeURIComponent(id));
      renderDetail(await response.json());
      document.querySelectorAll('.row').forEach(row => row.classList.toggle('active', row.dataset.id === id));
    }}

    async function rerun(id, stage) {{
      const response = await fetch(`/api/rerun/${{encodeURIComponent(id)}}/${{encodeURIComponent(stage)}}`, {{ method: 'POST' }});
      const data = await response.json();
      if (!response.ok || data.error) {{
        alert(data.error || 'Stage re-run failed');
        return;
      }}
      renderDetail(data);
      runSearch();
    }}

    function renderDetail(item) {{
      const html = `
        <div class="panel">
          <h2>Preview</h2>
          <div class="body">
            ${{item.preview_480_url ? `<video src="${{item.preview_480_url}}" controls loop muted></video>` : (item.poster_url ? `<img class="thumb" style="width:min(720px,100%);height:auto" src="${{item.poster_url}}" alt="">` : '')}}
            <p>${{escapeHtml(item.caption || '')}}</p>
          </div>
        </div>
        ${{panel('Stage 0', 'metadata', item.element_id, alphaPanel(item))}}
        ${{panel('Stage 0.5', 'artifacts', item.element_id, representations(item))}}
        ${{panel('Stage 1', 'classification', item.element_id, classificationPanel(item))}}
        ${{panel('Stage 2', 'caption', item.element_id, `<p>${{escapeHtml(item.caption || '')}}</p><p class="meta">${{escapeHtml(item.captioning_model_version || '')}}</p>`)}}
        ${{panel('Stage 3', 'motion', item.element_id, motionPanel(item))}}
        ${{panel('Stage 4', 'embedding', item.element_id, kv({{
          embedding_model_version: item.embedding_model_version,
          image_embed_dimensions: (item.image_embed || []).length,
          caption_embed_dimensions: (item.caption_embed || []).length
        }}))}}
        <section class="panel" id="compare-panel"><h2>Compare captions</h2><div class="body"><p class="meta">Loading…</p></div></section>
      `;
      detail.replaceChildren();
      detail.insertAdjacentHTML('afterbegin', html);
      detail.querySelectorAll('[data-rerun]').forEach(button => button.addEventListener('click', () => rerun(item.element_id, button.dataset.rerun)));
      loadCompare(item.element_id);
    }}

    function alphaPanel(item) {{
      const rows = {{
        duration_seconds: item.duration_seconds,
        resolution: `${{item.width || '?'}} x ${{item.height || '?'}}`,
        aspect_ratio: item.aspect_ratio,
        fps: item.fps,
        color_space: item.color_space,
        bit_depth: item.bit_depth,
        has_alpha_channel: item.has_alpha_channel,
        alpha_non_empty: item.alpha_non_empty,
        alpha_is_binary: item.alpha_is_binary,
        alpha_is_soft: item.alpha_is_soft,
        premult_status: item.premult_status,
        over_black_likely: item.over_black_likely,
        keyable_bg_likely: item.keyable_bg_likely
      }};
      const provenance = item.alpha_provenance || {{}};
      const cells = Object.entries(rows).map(([k, v]) => {{
        const prov = provenance[k];
        return `<tr><td>${{escapeHtml(k)}}</td><td>${{formatValue(v)}} ${{prov ? renderProvenance(prov) : ''}}</td></tr>`;
      }}).join('');
      return `<table>${{cells}}</table>`;
    }}

    function classificationPanel(item) {{
      const provCategory = (item.provenance || {{}}).category;
      const head = `
        <table>
          <tr><td>category</td><td><span class="facet-value">${{escapeHtml(item.category || '')}}</span> ${{provCategory ? renderProvenance(provCategory) : ''}}</td></tr>
          <tr><td>confidence</td><td>${{Number(item.category_confidence || 0).toFixed(2)}}</td></tr>
        </table>`;
      return head + facetTable(item.content_facets || {{}});
    }}

    function motionPanel(item) {{
      const facets = facetTable(item.motion_facets || {{}});
      const measured = `<pre>${{escapeHtml(JSON.stringify(item.measured_features || {{}}, null, 2))}}</pre>`;
      const warnings = (item.motion_warnings || []).length
        ? `<p class="meta">${{(item.motion_warnings || []).map(escapeHtml).join(' · ')}}</p>` : '';
      return facets + warnings + measured;
    }}

    function facetTable(facets) {{
      const entries = Object.entries(facets);
      if (!entries.length) return '<p class="meta">no facets</p>';
      const rows = entries.map(([key, payload]) => {{
        if (payload && typeof payload === 'object' && 'value' in payload) {{
          return `<tr><td>${{escapeHtml(key)}}</td><td><span class="facet-value">${{escapeHtml(String(payload.value))}}</span> ${{renderProvenance(payload)}}</td></tr>`;
        }}
        return `<tr><td>${{escapeHtml(key)}}</td><td>${{formatValue(payload)}}</td></tr>`;
      }}).join('');
      return `<table>${{rows}}</table>`;
    }}

    function renderProvenance(payload) {{
      if (!payload) return '';
      const src = String(payload.source || 'unknown');
      const conf = Number(payload.confidence || 0);
      const tier = conf >= 0.7 ? 'high' : conf >= 0.35 ? 'med' : 'low';
      return `<span class="badge src-${{escapeAttr(src)}}">${{escapeHtml(src)}}</span><span class="conf-dot conf-${{tier}}" title="confidence ${{conf.toFixed(2)}}"></span>`;
    }}

    function escapeAttr(value) {{
      return String(value).replace(/[^a-z0-9_-]/gi, '_');
    }}

    async function loadCompare(id) {{
      const target = document.querySelector('#compare-panel .body');
      if (!target) return;
      try {{
        const response = await fetch('/api/compare/' + encodeURIComponent(id));
        const data = await response.json();
        renderCompare(target, id, data);
      }} catch (err) {{
        target.replaceChildren();
        target.insertAdjacentHTML('afterbegin', '<p class="meta">compare unavailable</p>');
      }}
    }}

    function renderCompare(target, id, data) {{
      const html = (!data || !data.available)
        ? `<p class="meta">No alternate caption recorded for this element. Drop a JSON file alongside the library at <code>library.compare.json</code> with shape <code>{{"model_a_label":"Qwen3-VL","model_b_label":"InternVL3.5","elements":{{"&lt;id&gt;":{{"alternate_caption":"..."}}}}}}</code> to enable the A/B grader.</p>`
        : `
          <div class="compare-grid">
            <div class="compare-card"><h4>${{escapeHtml(data.model_a || 'A')}}</h4><p>${{escapeHtml(data.caption_a || '')}}</p></div>
            <div class="compare-card"><h4>${{escapeHtml(data.model_b || 'B')}}</h4><p>${{escapeHtml(data.caption_b || '')}}</p></div>
          </div>
          <div class="vote-row" style="margin-top:8px;display:flex;gap:6px">
            <button data-vote="a">A wins</button>
            <button data-vote="b">B wins</button>
            <button data-vote="tie">Tie</button>
            <button data-vote="skip">Skip</button>
          </div>
          <p class="meta" id="vote-status"></p>`;
      target.replaceChildren();
      target.insertAdjacentHTML('afterbegin', html);
      target.querySelectorAll('[data-vote]').forEach(btn => btn.addEventListener('click', () => castVote(id, btn.dataset.vote, target)));
    }}

    async function castVote(id, vote, target) {{
      const response = await fetch('/api/compare/' + encodeURIComponent(id), {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify({{ vote }})
      }});
      const data = await response.json();
      const status = target.querySelector('#vote-status');
      if (status) status.textContent = data.ok ? `recorded vote: ${{data.vote}}` : 'vote failed';
      target.querySelectorAll('[data-vote]').forEach(btn => btn.classList.toggle('cast', btn.dataset.vote === vote));
    }}

    function panel(title, stage, id, body) {{
      return `<section class="panel"><h2>${{title}} <button data-rerun="${{stage}}">Re-run</button></h2><div class="body">${{body}}</div></section>`;
    }}

    function kv(data) {{
      return `<table>${{Object.entries(data).map(([key, value]) => `<tr><td>${{escapeHtml(key)}}</td><td>${{formatValue(value)}}</td></tr>`).join('')}}</table>`;
    }}

    function representations(item) {{
      return (item.source_representations || []).map(rep => `
        <div style="margin-bottom:10px">
          <div>${{rep.is_primary ? '<span class="badge">primary</span>' : ''}}<code>${{escapeHtml(rep.path)}}</code></div>
          <div class="meta">${{escapeHtml(rep.format)}} · ${{rep.width || '?'}}x${{rep.height || '?'}} · confidence ${{Number(rep.link_confidence || 0).toFixed(2)}}</div>
          <div>
            ${{rep.linked_by_filename ? '<span class="badge">filename</span>' : ''}}
            ${{rep.linked_by_fingerprint ? '<span class="badge">fingerprint</span>' : ''}}
            ${{rep.linked_by_phash ? '<span class="badge">pHash</span>' : ''}}
          </div>
        </div>
      `).join('');
    }}

    function formatValue(value) {{
      if (value === null || value === undefined) return '<span class="meta">unknown</span>';
      if (typeof value === 'object') return `<pre>${{escapeHtml(JSON.stringify(value, null, 2))}}</pre>`;
      return escapeHtml(String(value));
    }}
    function formatDuration(value) {{
      return value ? `${{Number(value).toFixed(2)}}s` : '?s';
    }}
    function escapeHtml(value) {{
      return String(value).replace(/[&<>"']/g, ch => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[ch]));
    }}

    document.querySelector('#search').addEventListener('click', runSearch);
    query.addEventListener('keydown', event => {{ if (event.key === 'Enter') runSearch(); }});
    category.addEventListener('change', runSearch);
    loadElements();
  </script>
</body>
</html>"""
