"""Local background catalog jobs, with bounded progress and cooperative cancellation."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
from datetime import datetime, timezone
from uuid import uuid4

from .catalog_lock import CatalogBusy, catalog_writer
from .settings import models_dir
from .store import LibraryStore


PREFIX = "VFX_JOB_EVENT "
ACTIVE = {"queued", "running", "cancelling"}


class JobCancelled(Exception):
    pass


def local_folder(value: str) -> Path:
    if not isinstance(value, str) or not value.strip() or len(value) > 4096:
        raise ValueError("Enter an absolute local folder path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError("Folder path must be absolute")
    path = path.resolve(strict=True)
    if not path.is_dir():
        raise ValueError("Choose a folder, not a file")
    return path


def browse_folders(value: str | None) -> dict:
    path = local_folder(value) if value else Path.home()
    directories = []
    truncated = False
    for entry in path.iterdir():
        if entry.name.startswith("."):
            continue
        try:
            if entry.is_dir():
                directories.append({"name": entry.name, "path": str(entry.resolve())})
                if len(directories) >= 1000:
                    truncated = True
                    break
        except OSError:
            continue
    return {"path": str(path), "parent": str(path.parent) if path.parent != path else None,
            "folders": sorted(directories, key=lambda item: item["name"].casefold()),
            "truncated": truncated}


class LibraryJobs:
    def __init__(self, library: Path, cache: Path, manifest: Path | None = None,
                 artifacts_enabled: bool = True):
        self.library = library.resolve()
        self.cache = cache.resolve()
        self.manifest = (manifest or models_dir() / "manifest.json").resolve()
        self.artifacts_enabled = artifacts_enabled
        key = hashlib.sha256(str(self.library).encode()).hexdigest()[:16]
        self.directory = self.cache / "web-jobs" / key
        self._lock = threading.RLock()
        self._job = None
        self._process = None
        self._thread = None
        status = self.directory / "latest.json"
        if status.is_file():
            try:
                self._job = json.loads(status.read_text())
                if self._job.get("state") in ACTIVE:
                    self._job.update(state="interrupted", message="Server stopped. Completed elements are retained.")
            except (OSError, ValueError):
                self._job = None

    @property
    def busy(self):
        with self._lock:
            return bool(self._job and self._job["state"] in ACTIVE)

    def capabilities(self):
        error = None
        try:
            manifest = json.loads(self.manifest.read_text())
            primary = manifest.get("analysis_stack", {}).get("primary", "analysis_primary")
            entry = manifest["models"][primary]
            if entry.get("backend", "mlx_video") == "mlx_video":
                if importlib.util.find_spec("mlx_vlm") is None:
                    error = "MLX runtime unavailable. Start the server from the AI environment."
                elif not Path(entry.get("path", "")).is_dir():
                    error = "Primary model snapshot not found. Download the configured models first."
        except (OSError, ValueError, KeyError, TypeError, ImportError):
            error = "Model manifest unavailable. Set up local models before analysis."
        return {"analysis_available": error is None, "analysis_error": error}

    def snapshot(self):
        with self._lock:
            return {**self.capabilities(), "job": json.loads(json.dumps(self._job))}

    def _save(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        temporary = self.directory / f"latest-{os.getpid()}.tmp"
        temporary.write_text(json.dumps(self._job), encoding="utf-8")
        os.replace(temporary, self.directory / "latest.json")

    def start(self, payload):
        kind = payload.get("kind")
        if kind not in {"ingest", "pending", "force"}:
            raise ValueError("Choose ingest, pending or force")
        request = {"kind": kind}
        if kind == "ingest":
            folder = local_folder(payload.get("folder"))
            if folder.parent == folder:
                raise ValueError("Choose a media folder, not the filesystem root")
            fps = payload.get("default_fps", 24)
            if isinstance(fps, bool) or not isinstance(fps, (float, int)) or not math.isfinite(fps) or not 0 < fps <= 240:
                raise ValueError("Sequence frame rate must be greater than 0 and at most 240")
            analyse = payload.get("analyse", True)
            if not isinstance(analyse, bool):
                raise ValueError("analyse must be true or false")
            request.update(folder=str(folder), default_fps=fps, analyse=analyse)
        else:
            limit = payload.get("limit", 20)
            if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100000:
                raise ValueError("Batch size must be an integer from 1 to 100000")
            request["limit"] = limit
        if kind == "force":
            if payload.get("confirmed") is not True:
                raise ValueError("Confirm forced reprocessing before starting")
            scope = payload.get("scope", "selected")
            if scope not in {"selected", "all"}:
                raise ValueError("Choose selected elements or the entire catalog")
            ids = payload.get("element_ids", [])
            if not isinstance(ids, list) or len(ids) > 5000 or any(not isinstance(x, str) or len(x) > 128 for x in ids):
                raise ValueError("Invalid element selection")
            ids = list(dict.fromkeys(ids))
            existing = {x.element_id for x in LibraryStore.load_readonly(self.library)} if self.library.is_file() else set()
            if scope == "all":
                if not existing:
                    raise ValueError("Catalog is empty")
                request["limit"] = len(existing)
            elif not ids or not set(ids) <= existing:
                raise ValueError("Select existing elements to reprocess")
            else:
                request.update(element_ids=ids, limit=len(ids))
        if kind == "pending" and (not self.library.is_file() or not LibraryStore.load_readonly(self.library)):
            raise ValueError("Add elements before starting analysis")
        if kind != "ingest" or request.get("analyse"):
            capability = self.capabilities()
            if not capability["analysis_available"]:
                raise ValueError(capability["analysis_error"])
        with self._lock:
            if self.busy:
                raise CatalogBusy("A library job is already running")
            # Detect another cooperating server/CLI before reporting that this job started.
            with catalog_writer(self.library):
                pass
            job_id = uuid4().hex
            request.update(library=str(self.library), cache=str(self.cache), manifest=str(self.manifest),
                           artifacts_enabled=self.artifacts_enabled,
                           cancel_path=str(self.directory / f"{job_id}.cancel"))
            self._job = {"id": job_id, "kind": kind, "state": "queued", "phase": "queued",
                         "completed": 0, "total": None, "errors": 0, "logs": [],
                         "started_at": datetime.now(timezone.utc).isoformat()}
            self._save()
            self._thread = threading.Thread(target=self._run, args=(request,), daemon=True)
            self._thread.start()
            return self.snapshot()

    def _run(self, request):
        process = None
        try:
            environment = os.environ.copy()
            source_root = str(Path(__file__).resolve().parents[1])
            environment["PYTHONPATH"] = source_root + os.pathsep + environment.get("PYTHONPATH", "")
            process = subprocess.Popen([sys.executable, "-u", "-m", __name__, "--worker"],
                                       stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, encoding="utf-8", errors="replace", env=environment,
                                       start_new_session=os.name != "nt")
            with self._lock:
                self._process = process
                if self._job["state"] != "cancelling":
                    self._job["state"] = "running"
            process.stdin.write(json.dumps(request))
            process.stdin.close()
            while line := process.stdout.readline(8192):
                with self._lock:
                    if line.startswith(PREFIX):
                        try:
                            event = json.loads(line[len(PREFIX):])
                            allowed = {"phase", "completed", "total", "errors", "processed", "reused",
                                       "current", "backup", "message"}
                            self._job.update({key: value for key, value in event.items() if key in allowed})
                            if event.get("message"):
                                self._job["logs"] = (self._job["logs"] + [str(event["message"])[-2000:]])[-40:]
                        except (ValueError, AttributeError):
                            self._job["logs"] = (self._job["logs"] + [line.rstrip()[-2000:]])[-40:]
                    else:
                        self._job["logs"] = (self._job["logs"] + [line.rstrip()[-2000:]])[-40:]
                    self._save()
            code = process.wait()
            with self._lock:
                self._job["state"] = "cancelled" if code == 130 else "completed" if code == 0 else "failed"
                self._job["finished_at"] = datetime.now(timezone.utc).isoformat()
                self._job["exit_code"] = code
                if self._job.get("message") == "Stopping after the current element":
                    self._job["message"] = "Stopped. Completed elements are retained." if code == 130 else ""
                self._save()
        except Exception as exc:
            if process is not None and process.poll() is None:
                process.terminate()
                process.wait()
            with self._lock:
                self._job.update(state="failed", message=str(exc))
                self._save()
        finally:
            if process is not None:
                for stream in (process.stdin, process.stdout):
                    if stream is not None and not stream.closed:
                        stream.close()
            with self._lock:
                self._process = None
            Path(request["cancel_path"]).unlink(missing_ok=True)

    def cancel(self, job_id):
        with self._lock:
            if not self.busy or self._job["id"] != job_id:
                raise ValueError("No matching active job")
            (self.directory / f"{job_id}.cancel").touch()
            self._job.update(state="cancelling", message="Stopping after the current element")
            self._save()
            return self.snapshot()

    def shutdown(self):
        with self._lock:
            if self.busy:
                self.cancel(self._job["id"])
            process = self._process
        if process is not None and process.poll() is None:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                if os.name != "nt":
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                else:
                    process.terminate()
        if self._thread:
            self._thread.join(timeout=10)


def worker(request):
    cancel_path = Path(request["cancel_path"])

    def stopped():
        return cancel_path.exists()

    def emit(event):
        print(PREFIX + json.dumps(event), flush=True)

    def ingest_progress(event):
        if stopped():
            raise JobCancelled()
        emit(event)

    library, cache = Path(request["library"]), Path(request["cache"])
    try:
        with catalog_writer(library):
            if stopped():
                return 130
            store = LibraryStore(library)
            if library.is_file():
                backup = cache / "backups" / f"web-{uuid4().hex}{library.suffix}"
                backup.parent.mkdir(parents=True, exist_ok=True)
                store.backup(backup)
                emit({"backup": str(backup)})
            ids = request.get("element_ids", [])
            import_errors = 0
            if request["kind"] == "ingest":
                from .pipeline import run_ingest
                result = run_ingest(Path(request["folder"]), cache, library,
                                    default_fps=request["default_fps"],
                                    artifacts_enabled=request["artifacts_enabled"],
                                    progress=ingest_progress, writer_locked=True)
                ids = result.scanned_element_ids
                import_errors = result.failed_references
                if not request["analyse"] or not ids:
                    return 1 if import_errors else 0
            if stopped():
                return 130
            from .analysis_runner import main as analyze
            argv = ["--library", str(library), "--models-manifest", request["manifest"],
                    "--limit", str(len(ids) if ids else request.get("limit", 20))]
            if request["kind"] == "force":
                argv.append("--force")
            for element_id in ids:
                argv.extend(["--element-id", element_id])
            def analysis_progress(event):
                if "errors" in event:
                    event["errors"] += import_errors
                emit(event)
            code = analyze(argv, progress=analysis_progress, should_stop=stopped, writer_locked=True)
            return code or (1 if import_errors else 0)
    except JobCancelled:
        return 130
    except Exception as exc:
        emit({"message": f"{type(exc).__name__}: {exc}"})
        return 1


if __name__ == "__main__" and sys.argv[1:] == ["--worker"]:
    raise SystemExit(worker(json.loads(sys.stdin.read(1024 * 1024))))
