# VFX Element Tagger User Guide

This guide describes the v1 release candidate (`1.0.0rc1`). Internal development
reports use other iteration numbers; they are not public package versions. Commands run
from the repository root in an activated Python environment unless stated otherwise.

## Contents

1. [What the application does](#what-the-application-does)
2. [Platform and hardware](#platform-and-hardware)
3. [Installation](#installation)
4. [Configuration and storage](#configuration-and-storage)
5. [Download and validate models](#download-and-validate-models)
6. [Ingest footage and sequences](#ingest-footage-and-sequences)
7. [Run AI analysis](#run-ai-analysis)
8. [Browse and search](#browse-and-search)
9. [Review and correct results](#review-and-correct-results)
10. [Add assets and reprocess](#add-assets-and-reprocess)
11. [Backups, exports and migration](#backups-exports-and-migration)
12. [Advanced workflows](#advanced-workflows)
13. [Command reference](#command-reference)
14. [Troubleshooting](#troubleshooting)
15. [Accuracy, privacy and limitations](#accuracy-privacy-and-limitations)

## What the Application Does

VFX Element Tagger builds a local catalog from movies, still images and numbered image
sequences. Multiple representations, such as a master and its proxy, can belong to one
logical element. Original media remains on disk; the catalog records source locations
and generates separate browsing artifacts.

The normal workflow is:

```text
Install -> Configure storage -> Download models -> Ingest -> Analyse -> Review -> Search
```

Ingest extracts technical metadata and creates posters and preview movies. Initial
classification/captions are deterministic fallbacks, not proof that an AI model ran.
The separate gated analyzer observes frames across the clip, proposes structured VFX
descriptions, checks uncertainty, and optionally consults challenger models. Artists
can correct the results through a review queue.

The interface is a local web application. There is no packaged native installer,
hosted account, public cloud-analysis service or supported internet deployment.

## Platform and Hardware

| Platform | Catalog/ingest/browser | Default AI analyzer |
| --- | --- | --- |
| Apple Silicon Mac | Tested locally | Tested with MLX/Metal |
| Intel Mac | Expected portable core; not tested | Not supported by the tested configuration |
| Linux | Core tests pass on Python 3.11/3.12; media workflow not certified | No validated CUDA/CPU workflow |
| Windows | Core tests pass on Python 3.11/3.12; media workflow not certified | No validated default backend |
| Another device's browser | Not supported; v1 serves only on loopback | Analysis runs on the local workstation, not in the browser |

Use native ARM64 Python 3.12 for the tested Mac AI workflow. The package declares Python
3.11+, but newer Python versions and the full AI stack on 3.11 have not been certified.
MLX documents Apple Silicon, native Python and macOS 14+ for its Mac installation.
It also has Linux CUDA/CPU backends, but those upstream capabilities are not evidence
that this application's patched VLM adapters work there.
[Upstream installation requirements](https://ml-explore.github.io/mlx/build/html/install.html).

The three current analysis snapshots occupy approximately 11 GB on the development
workstation. Reserve extra space for Python packages, download staging and previews;
20-30 GB free before installing the complete stack is a planning allowance, not a
measured minimum. Source footage is additional. Memory requirements and throughput
have not been systematically benchmarked across Mac models. Start with the primary
model and one short element; do not assume a particular RAM size guarantees success.

FFmpeg and ffprobe must be on `PATH` for useful metadata and movie previews. OpenImageIO
is optional and improves EXR metadata/conversion. A current browser is required.

## Installation

### Obtain a Source Checkout

Clone [the public repository](https://github.com/kkthxs/vfx-element-tagger), or extract
a source archive from [Releases](https://github.com/kkthxs/vfx-element-tagger/releases),
into a local folder outside Dropbox/iCloud/OneDrive:

```bash
git clone https://github.com/kkthxs/vfx-element-tagger.git
cd vfx-element-tagger
```

Open a terminal in that folder. The first public v1 build is a release candidate,
not a certified stable or cross-platform AI release.

Do not move an existing catalog or model directory merely to follow these examples.
Existing installations should retain their local configuration and back up before upgrading.

### Apple Silicon: Full AI Workflow

Install native Python 3.12, Git and FFmpeg. With Homebrew already installed:

```bash
brew install ffmpeg
# Optional, for richer EXR support:
brew install openimageio
```

Check the interpreter and external tools:

```bash
python3.12 -c "import platform; print(platform.python_version(), platform.machine())"
ffmpeg -version
ffprobe -version
```

The Python architecture should be `arm64`, not `x86_64` through Rosetta. Then:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r constraints/macos-arm64-py312.lock.txt
python -m pip install -c constraints/macos-arm64-py312.lock.txt -e ".[mlx,hashing]"
python -m pip check
python -c "import mlx.core as mx; mx.eval(mx.array([1])); print('MLX compute OK')"
```

The lock file pins the full MLX dependency closure for native macOS arm64 / Python 3.12.
It is not a portable lock for other platforms or Python versions, and does not hash
Python distribution downloads. Do not silently remove pins to bypass a resolver error.
Use this full lock rather than historical partial development constraints.

### Catalog-Only Installation

No AI weights are required to scan, browse, rate or search existing catalog text:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[hashing]"
```

Install FFmpeg/ffprobe separately for your OS. On Windows, create the environment with
`py -3.12 -m venv .venv` and activate using `.venv\Scripts\Activate.ps1` in PowerShell.
Windows/Linux commands are provided for the CI-tested portable core, not as a claim of
tested AI support or complete FFmpeg/preview compatibility on those systems.

The `[models]` extra is the older PyTorch/SigLIP/text-embedding stack. It is not required
for the recommended MLX gated workflow and does not enable a new Windows/CUDA backend.

## Configuration and Storage

For a new installation only, copy the example configuration:

```bash
cp -n .vfx-tagger.example.json .vfx-tagger.json
```

On Windows use `Copy-Item .vfx-tagger.example.json .vfx-tagger.json` only if the destination
does not already exist. Edit the local JSON to suit the machine:

```json
{
  "models_dir": "../vfx-element-tagger-models",
  "cache_dir": ".cache/vfx-element-tagger",
  "library": "data/library.sqlite3"
}
```

| Setting | Purpose | Environment override | Unconfigured fallback |
| --- | --- | --- | --- |
| `models_dir` | Weights and `manifest.json` | `VFX_TAGGER_MODELS_DIR` | `~/.cache/vfx-element-tagger/models` |
| `cache_dir` | Artifact root containing `artifacts/` | `VFX_TAGGER_CACHE_DIR` | `<checkout>/.cache/vfx-element-tagger` |
| `library` | SQLite or JSON catalog | `VFX_TAGGER_LIBRARY` | `<checkout>/.cache/vfx-element-tagger/library.json` |

`VFX_TAGGER_CONFIG` selects a different JSON configuration file. Relative configured
paths resolve against the checkout root, not against the configuration file's directory.
`~` is expanded. On Windows use forward slashes in JSON paths or escape backslashes.
An environment override wins over the corresponding JSON value. Explicit command-line
paths win where that command supports them.

An important ingest exception: specifying `--cache-dir` without `--library` selects
`<that-cache>/library.json`, not the configured SQLite catalog. Supply both arguments
when creating an intentionally separate library. Passing `--models-dir` to the downloader
does not persist that choice for the analyzer; configure the same directory or pass
`--models-manifest` to subsequent model commands.

Keep a live SQLite catalog on local disk, not a synced or network filesystem. For team
access, browsers connect to one application host rather than opening its database file.
Weights can be outside the checkout. Git deliberately excludes weights, media, catalogs,
caches and `.vfx-tagger.json`; Git is not a backup of those files.

## Download and Validate Models

From the activated Mac environment:

```bash
python scripts/download_models.py
python scripts/check_models.py
```

The downloader writes a manifest under the configured model directory. The default
download is only the primary analyzer:

| Key | Snapshot | Role |
| --- | --- | --- |
| `analysis_primary` | `mlx-community/Qwen3.5-4B-MLX-4bit` | Structured native-video analyzer |
| `challenger_minicpm` | `mlx-community/MiniCPM-o-4_5-4bit` | Optional independent ordered-frame view |
| `challenger_marlin` | `NemoStation/Marlin-2B-MLX-8bit` | Optional narrative temporal evidence |
| Video embedding | Not included in the frozen v1 stack | Not integrated into search |

To add both optional challengers:

```bash
python scripts/download_models.py --with-challengers
python scripts/check_models.py
```

Marlin's repository is gated. Request access on its model page, then authenticate with
the Hugging Face CLI (`hf auth login`) in the same environment. Never put a token into
Git or an issue. A failed optional challenger is recorded in `unavailable_models`; the
other downloads can still be usable. Read the downloader output and inspect the manifest.

An explicit selection is also available:

```bash
python scripts/download_models.py --only analysis_primary challenger_minicpm
```

`check_models.py` verifies SHA-256 files and lazy loaders; it is not a full inference or
quality test. Use `--verify-only` for checksum verification without importing MLX. Start
a one-element analysis to exercise the full path. Downloads use immutable revisions
from the packaged model lock, and known snapshots are checked again before model load.
Corrupt, modified or additional weight files are refused; redownload the frozen snapshot
instead of bypassing verification.
The downloader makes one documented Marlin metadata repair without altering its weights.

After setup, normal default inference uses local snapshots and media; it does not upload
clips to a hosted model. Initial downloads need internet. For a deliberate offline run,
set `HF_HUB_OFFLINE=1` and test the complete workflow before relying on it without a network.
Custom command backends are outside that default privacy guarantee.

Read [third-party model terms](THIRD-PARTY.md) before using or redistributing snapshots.

## Ingest Footage and Sequences

```bash
python pilot.py --folder "/absolute/path/to/elements" --no-server
```

Discovery is recursive. Supported filename extensions include:

- Movies: `.mov`, `.mp4`, `.m4v`, `.mkv`, `.avi`, `.webm`, `.mpg`, `.mpeg`.
- Images: `.exr`, `.dpx`, `.tif`, `.tiff`, `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tga`.

An extension is a discovery rule, not a guarantee that the installed codecs can decode it.
Numbered files with at least two matching members form sequences, for example
`smoke.1001.exr`, `smoke.1002.exr`, or `0001.HR.exr`, `0002.HR.exr`. A lone numbered
file is a still. Sequence gaps are recorded; inspect the resulting preview, because
a missing source frame can interrupt conversion. Specify the actual sequence FPS:

```bash
python pilot.py --folder "/absolute/path/to/sequences" --default-fps 24 --no-server
```

The scan prints discovered-reference and processed/reused/preserved counts, the library
path and the processing-log location. Some malformed assets can fail without stopping
the whole scan. Inspect failures in the log and confirm that expected elements arrived.

Technical metadata comes from decoders/probes, not the VLM. Alpha softness, premultiplication,
keyability and border heuristics are limited observations, not universal compositing truth.
EXR tone mapping is for browsing; do not judge final color or deliver plates from these proxies.

### Master/Proxy Grouping

The system infers links using fingerprints, filename/geometry compatibility and perceptual
evidence. Inspect surprising merges; it is not a perfect duplicate detector. To override
grouping, place a `.proxy-chains.json` in the relevant media folder:

```json
{
  "version": 1,
  "groups": [
    {"members": ["master.mov", "preview.mp4"]}
  ],
  "separate": [["variant_a.mov", "variant_b.mov"]]
}
```

Paths are relative to that sidecar. Members must exist. Rescan after changing grouping,
using a backup/test catalog first: changing representations can change element identities.
The optional `id` field in a group is not an enforced catalog-ID override.

## Run AI Analysis

Begin with a bounded primary-only test:

```bash
python scripts/analyze_gated.py --dry-run --no-challengers --limit 1
python scripts/analyze_gated.py --no-challengers --limit 1
```

The dry run lists pending/stale IDs without inference or catalog writes, but still
requires an existing catalog and valid manifest. The default batch size is five,
chosen for diverse source types/durations rather than necessarily the newest files.

When challengers are installed, enable the normal gated workflow:

```bash
python scripts/analyze_gated.py --dry-run --limit 20
python scripts/analyze_gated.py --limit 20
```

The primary sees timestamped frames across the whole clip. Uniform anchors and measured
activity-focus samples share a bounded visual-token budget. Input diagnostics retain actual
frame indices, timestamps, dimensions and representation. MiniCPM receives ordered images
with times, not a verified native-video stream. Uncertain cases can use denser sampling,
challengers and adjudication; not every clip loads every model.

Motion measurements guide attention and challenge contradictions. They do not reliably
tell the model whether a speck is snow, a spark or dust, nor determine physical lens data.
Sparse-detail/impact specialists can add conservative material and temporal evidence.

Per-element results are saved as work completes. Ctrl+C stops a foreground analysis;
completed elements remain saved. The current in-flight element may need retrying.
There is no automatic folder watcher or automatic recurring analysis job.

New model results always enter the artist review queue, even if the internal gate passes.
`--allow-auto-accept` is an explicit experimental override, not a stable-quality guarantee.
It changes the analysis cache signature. Old catalogs may contain historical auto-accepted
or challenger-checked records; those statuses are not retroactive proof of artist review.

| Stored status | Interface label | Interpretation |
| --- | --- | --- |
| `unanalysed` | Not analysed | No current gated semantic result |
| `accepted` | Accepted | Passed the gate, or was artist-confirmed; inspect provenance |
| `escalated` | Challenger checked | Selected after escalation; not a guarantee of independent correctness |
| `needs_review` | Needs review | All new AI results require artist confirmation by default |
| `failed` | Failed | Runtime/input failure; inspect saved error and retry after fixing it |

A zero process exit code means no whole-element runtime failures. It does not mean all
candidates parsed, all items were accepted or the tags are correct. Model confidence is
uncalibrated. In the latest six-clip diagnostic run, four results still needed review.

## Browse and Search

```bash
python scripts/serve_library.py
```

Open [the library](http://127.0.0.1:8765/). Starting the server reads the configured catalog
or opens an empty one; it does not start a scan or inference until you request it.
The interface includes local folder import and background analysis controls. Start the
server from the same activated AI environment used for model setup if you want analysis;
a core-only environment can still import and browse. `--models-manifest` selects a trusted
local manifest at server startup. There is no web control for arbitrary executables or
model downloads. Browser-triggered jobs use the server's Python environment.

Browse by family/subtype, status, alpha, image sequences or missing frames. Cards show
compact descriptions and hover previews. Select a card to read the full description,
play the preview, inspect composition/motion/appearance/technical data, and copy its
source path. A compact card description is intentionally shortened; the inspector is not.
RGB/Alpha switching appears when a usable alpha matte can be generated.

Try queries such as:

```text
wide green screen man turning
fast smoke cutting the right edge
blue branching lightning
wide overhead explosion
```

Search canonicalizes some artist synonyms and ranks confirmed text/concept coverage.
It cannot find facts the analyzer missed. Filenames cannot establish visual family
evidence on analyzed assets; uncertain alternatives and generated search suggestions
cannot masquerade as confirmed observations. A broad search can help diagnose an overly
specific query, but inspect the clip before treating a hit as a match.

The browser has a basic filter set; more precise dotted semantic filters are available
through the read API. For example, with the server running:

```bash
curl --get "http://127.0.0.1:8765/api/search" \
  --data-urlencode "q=smoke" \
  --data-urlencode "count=10" \
  --data-urlencode "filter_semantic.motion.speed=fast"
```

Ratings are artist quality preferences, not semantic correctness votes. Select one to
five stars in the inspector and provide a display name. A browser-local ID identifies
the rater; another browser/profile creates another identity. Re-rating updates that
identity's existing vote. Highest-rated sorting uses a prior to avoid a single vote
dominating established assets. Ratings cannot bypass search relevance.

The current search is not a learned video-vector database. Downloading the optional
embedding model or installing `[lance]` does not enable that future feature.

## Review and Correct Results

Open [the review queue](http://127.0.0.1:8765/review) or the Review link in the header.
The queue contains `needs_review` elements, not every item in the library.

1. Watch the complete clip, including faint starts, brief flashes and late falling material.
2. Read the unresolved issues and compare the selected analysis with candidate proposals.
3. Confirm or change the offered fields; use custom values when appropriate.
4. Add a note explaining an ambiguous crop, count, backing or material decision.
5. Submit the review and check the updated library/search result.

Submitting stores field values, previous values, time and note, and marks the element
accepted. This accepts the item based on the fields submitted; it does not independently
verify every remaining field. The UI offers a bounded field subset, not a complete
annotation editor for arbitrary captions, timing or all hidden metadata.

Human corrections and review history are retained and reapplied by gated AI refreshes.
Original model candidates/uncertainty remain as provenance. Corrected fields do not
manufacture a higher model-confidence score. To inspect all provenance and the element
ID, open the detail API for a known ID:

```text
http://127.0.0.1:8765/api/elements/ELEMENT_ID
```

Source changes, regrouping and ingest rebuilds have different preservation rules;
read the warnings below before using them on reviewed assets.

## Add Assets and Reprocess

### Through the Web Interface

1. Start `python scripts/serve_library.py` from your activated environment, then open
   [the local library](http://127.0.0.1:8765/). This also works before your first import.
2. Click **Add elements**. Enter an absolute local folder path, or click **Browse**,
   navigate to the footage folder and choose **Use this folder**. Folders are on the
   machine running the server, not a remote viewer's computer.
3. Set the correct image-sequence frame rate. Leave **Analyse imported elements**
   checked to run AI after import, or uncheck it for metadata/previews only. Analysis
   is disabled when the server cannot find the required runtime or model manifest.
4. Click **Add elements** inside the dialog. The recursive scan merges into the configured
   catalog; it does not upload/copy the original footage, prune unrelated entries or
   force unchanged results. Only elements discovered in that scan are considered for
   its optional AI step, and current matching analysis signatures are reused.
5. Follow the progress, current item and error count. Expand **Job log** for details and
   the automatic pre-batch backup location. The grid refreshes as analysis checkpoints
   are saved. New predictions enter the artist review queue.

**Process pending** runs a bounded batch of pending, failed or stale results across the
catalog, not necessarily only the latest import. Its default maximum is 20. Changes to
prompts, model configuration or analysis code can make old results stale, so this can
include previously reviewed elements. Artist overrides are reapplied after new predictions.
Moving the analysis runner into the package changes the signature once for older catalogs;
this allows the installed wheel and source checkout to use the same analysis implementation.

**Force reprocess** offers selected elements or the entire catalog. Select the items and
check **Replace model predictions; retain artist overrides** before starting. This bypasses
analysis reuse, not source discovery or proxy rebuilding. It never enables automatic
acceptance or discards saved artist overrides. Entire-catalog mode deliberately processes
all catalog elements; it is not capped at the pending-batch default.

Only one web job runs at a time. Browsing remains available, but rating, review, fallback
rerun and alpha-generation writes are paused during catalog jobs. Review submissions from
an old analysis are rejected with a reload message rather than silently applied to a
changed element. The current ingest/analyzer CLI entry points share the same cooperative
writer lease. Other maintenance tools still require stopping the server and all writers.

**Stop after current item** requests cooperative cancellation; a running model pass is
not interrupted instantly. Completed AI elements stay saved. An interrupted import scan
does not publish a partial new catalog, although generated cache artifacts may remain.
Browser reloads retain job progress; the server must remain running. Server shutdown stops
its worker, and a later launch marks unfinished jobs interrupted. Retry pending items after
fixing errors; do not assume interruption implies a cleanly completed batch.

Folders, job logs, backups and catalogs remain private local runtime data. They are not
added to Git or the public source export. Browser folder import is not a cloud upload.
There is still no automatic folder watcher or recurring job scheduler.

### Add New Files Without Reanalyzing Everything

```bash
python pilot.py --folder "/absolute/path/to/elements" --no-server
python scripts/analyze_gated.py --dry-run --limit 20
python scripts/analyze_gated.py --limit 20
```

Unchanged ingest records are reused. Assets outside the scan remain in the catalog
by default, so scanning a new subfolder can add to the same configured library.
Discovery still probes/fingerprints scanned representations; "incremental" does not
mean the scan performs no work. Completed accepted/escalated/review results skip AI
only when their source/model/code/options signature matches. A prompt/model/code or
option change can make existing records stale; the dry run shows what will be selected.
Legacy results without a signature need one fresh pass. Failed analyses are eligible for retry.

### Reanalyze One Item

Get its ID from the detail API/analysis output, then:

```bash
python scripts/analyze_gated.py --force --element-id ELEMENT_ID --limit 1
```

Repeat `--element-id` for several known elements and set `--limit` accordingly. Explicit
IDs do not disable the limit. Use the same challenger/options configuration when checking
cache reuse, because changing options changes the signature.

### Deliberately Reanalyze the Catalog

Back up first. Then choose a limit at least as large as the current catalog:

```bash
python scripts/analyze_gated.py --force --dry-run --limit 100000
python scripts/analyze_gated.py --force --limit 100000
```

This refreshes semantic predictions, not discovery or proxy generation. Saved human
override fields are reapplied last, so this is not a machine-only benchmark. Evaluate
fresh AI accuracy in a separate catalog without artist overrides.

### Destructive or Rebuilding Ingest Options

`pilot.py --force-recompute` is **not** the equivalent of analyzer `--force`. It rebuilds
technical metadata and artifacts, retaining analysis, artist overrides and review history
for matching element identities. Changed source content invalidates the AI signature and
embeddings, marks retained model results for review, and discards obsolete action timing
when duration changes. Run gated analysis afterwards to obtain fresh predictions.

If master/proxy grouping changes an existing identity, ingest stops before writing the
catalog. Use an isolated catalog and reconcile reviews/ratings explicitly; neither force
nor pruning bypasses this protection. Web imports/analysis serialize their writes. Stop
the UI and other writers before running separate merge, migration, reuse or maintenance
tools. v1 is a single-workstation workflow, not a distributed job scheduler.

`--prune-missing` keeps only records discovered in the current scan. On a new-assets
subfolder this can remove the rest of the catalog. It removes catalog records, not original
footage, but can also discard associated review/rating information. Never use it as the
normal "add new assets" operation.

Re-run buttons in Pipeline & model controls are deterministic fallback tools, not VLM
jobs. Most are deliberately refused for model-backed results. Use `analyze_gated.py`
for AI refresh; an artifacts-only rerun can repair missing previews without new captions.

## Backups, Exports and Migration

SQLite is recommended for the working catalog. JSON is useful for interchange/evaluation.
Back up before bulk analysis, ingest rebuilds, pruning, merging, relocation or upgrades.

With the editable package installed, create a consistent SQLite/JSON catalog snapshot:

```bash
python -c "from pathlib import Path; from vfx_element_tagger.settings import library_path; from vfx_element_tagger.store import LibraryStore; LibraryStore(library_path()).backup(Path('backups/library-before-update.sqlite3'))"
```

Use a new destination name each time. Match the suffix to the source backend: a JSON
backup remains JSON even if given a `.sqlite3` name. For SQLite, the backup API safely
includes WAL transactions; copying just the live database file may not. Catalog backups
do not include source media or proxy artifacts. JSON ratings live in a separate
`<library>.ratings.sqlite3`; back that up independently, along with processing logs,
comparison files/votes and the local configuration when needed.

Backups refuse to overwrite the live catalog or an existing destination. SQLite snapshots
include ratings and processing events, with an integrity check before publication. To
restore, stop the UI and all workers, keep the old catalog intact, and copy the backup to
a new catalog path. Point `library` in the local configuration at that path. Verify element
counts, artist overrides, ratings and playback before resuming work. Original sources and
the matching cache must still be available; a catalog snapshot is not a media backup.

To export element payloads to a new JSON file:

```bash
python scripts/migrate_catalog.py \
  --source data/library.sqlite3 \
  --target backups/library-export.json
```

To convert an existing JSON catalog into a new SQLite catalog:

```bash
python scripts/migrate_catalog.py \
  --source backups/library-export.json \
  --target data/imported-library.sqlite3
```

Migration checks element-payload equality. It does **not** transfer ratings, processing
events or all sidecars, so an export is not a complete operational backup. Do not replace
the active catalog from an element-only export expecting those other records to survive.
Avoid `--replace` against a live server/worker. Stop them first and retain a full backup.

Paths inside catalogs/model manifests are local, often absolute. Moving just JSON/SQLite
does not relink sources or artifacts on another machine. Update/remap paths, keep the
matching cache root and test previews after any relocation. The relocation script is
an advanced maintenance tool, not a portable project bundle installer.

## Advanced Workflows

### Separate Test Catalog

Supply both library and cache paths to keep experiments isolated:

```bash
python pilot.py --folder "/absolute/path/to/test-elements" \
  --library .cache/test-run/library.json --cache-dir .cache/test-run --no-server
python scripts/analyze_gated.py --library .cache/test-run/library.json --limit 5
python scripts/serve_library.py --library .cache/test-run/library.json \
  --cache-dir .cache/test-run --port 8766
```

### Merge or Reuse Analysis

```bash
python scripts/merge_catalogs.py \
  --target data/library.sqlite3 --source .cache/test-run/library.json
python scripts/reuse_analysis.py \
  --source-library .cache/test-run/library.json \
  --target-library data/library.sqlite3 --dry-run
```

Merge normally keeps existing target records, deduplicates by ID/fingerprint and makes
a catalog backup. Artifact copying and catalog merging do not bundle original footage,
all ratings or full processing history. Inspect output and previews. `--replace-existing`
is a deliberate overwrite choice, not normal addition. Reuse transfers analysis for
matching content fingerprints; run without `--dry-run` only after inspecting the selection.

### Deterministic Refinement and Timing Audit

```bash
python scripts/refine_catalog.py --library data/library.sqlite3 --dry-run --limit 10
python scripts/audit_action_timing.py --library data/library.sqlite3
```

Refinement reuses stored model output and measurements; it does not obtain a new visual
opinion. The timing audit checks ordering and bounds, not editorial accuracy. Suggested
action ranges can be broad, especially for plates; they must not automatically trim source media.

### Custom Local Backends

Advanced manifests can use `backend: "command"` and a command list. The wrapper receives
the prompt on stdin and the resolved video path as the final argument. It must write
the requested structured JSON to stdout and diagnostics to stderr. Command arguments
can contain `{fps}`, `{max_frames}` and `{max_tokens}` placeholders.

This extension point could host a Linux/Windows model adapter, but no ready-made
supported CUDA/CPU wrapper ships here. It is not a cloud API toggle. Only trust manifests
and wrappers you have reviewed; the command runs with the user's permissions.

### Validation

```bash
python -m pip install -e ".[dev,hashing]" opencv-python
python -m unittest discover -s tests
python scripts/prepare_benchmark.py --output evaluation/heldout-new.json
python scripts/evaluate_ground_truth.py \
  --library .cache/test-run/library.json \
  --ground-truth evaluation/testing_ground_truth.json
```

The annotation queue contains pending labels, not invented ground truth. Historical
development constraints reference example assets that are not distributed. Python source
archives also exclude that historical label file pending publication review; provide
your own approved `--ground-truth` file when it is absent. Missing examples count against
the score, so the evaluator is meaningful only with matching licensed test media.
Follow [the independent benchmark protocol](../evaluation/BENCHMARK-PROTOCOL.md).

## Command Reference

Use `python SCRIPT --help` for authoritative flags. Script paths below are relative to
the checkout root; the installed `vfx-element-tagger` command is the ingest CLI, not a
single launcher for every analysis/maintenance operation.

| Command | Important controls |
| --- | --- |
| `pilot.py` / `vfx-element-tagger` | Required `--folder`; `--library`, `--cache-dir`, `--default-fps`; `--no-server`, `--no-open`, `--no-artifacts`; hazardous `--force-recompute`, `--prune-missing` |
| `scripts/serve_library.py` | Existing or new `--library`, matching `--cache-dir`, optional `--models-manifest`, `--host` (default loopback), `--port` (default 8765) |
| `scripts/download_models.py` | `--models-dir`, `--with-challengers`, `--only` model keys |
| `scripts/check_models.py` | `--models-manifest`; checksums and MLX lazy-loader checks; `--verify-only` skips model imports |
| `scripts/analyze_gated.py` | `--library`, `--models-manifest`, `--limit` (5), repeatable `--element-id`, `--selection diverse\|sequential`, `--force`, `--dry-run` |
| Analyzer tuning | `--no-challengers`, `--max-challengers` (2), `--confidence-threshold` (0.88), `--max-tokens` (1400), experimental `--allow-auto-accept`; no certified accuracy meaning for these thresholds |
| `scripts/refine_catalog.py` | Required `--library`; repeatable `--element-id`, `--limit`, `--dry-run` |
| `scripts/merge_catalogs.py` | Required `--target`, `--source`; overwrite/backup/artifact options are advanced |
| `scripts/reuse_analysis.py` | Required `--source-library`, `--target-library`; `--dry-run` |
| `scripts/migrate_catalog.py` | Required `--source`, `--target`; dangerous `--replace` |
| `scripts/audit_action_timing.py` | Required `--library`; `--json` |
| `scripts/prepare_benchmark.py` | Required new `--output`; `--library`, `--exclude-truth`, `--seed`, `--limit` |
| `scripts/evaluate_ground_truth.py` | Required `--library`; `--ground-truth`, `--json`, `--min-score`; avoid legacy all-text scoring for accuracy claims |

`--no-artifacts` is useful for metadata-only diagnostics but does not create useful
browsing/inference proxies. Some movie sources can be used directly by analysis; image
sequences generally need generated previews. Do not use it for the normal first run.

## Troubleshooting

| Symptom | Checks and action |
| --- | --- |
| `ModuleNotFoundError: mlx` or `mlx_vlm` | Activate `.venv`; check `python -m pip --version`; install the documented `[mlx,hashing]` extra with constraints using that interpreter. |
| MLX/Metal fails | Confirm Apple Silicon/native ARM64 Python and compatible macOS. Run the tiny MLX compute check outside any restricted execution sandbox. A sandbox failure does not prove a driver problem. Consult upstream diagnostics before changing system tools. |
| `torchvision` missing in the old analyzer | Prefer `analyze_gated.py`; the legacy PyTorch script requires the separate `[models]` environment and old snapshots. Do not assume installing every extra repairs an incompatible runtime. |
| Hugging Face access denied | Request gated repository access, authenticate locally, inspect `unavailable_models`; primary-only can still work. |
| Manifest not found / primary missing | Align `models_dir` or `--models-manifest` with the downloader; download `analysis_primary` in that directory. |
| No preview / black placeholder | Verify ffmpeg, source readability, processing errors, and that `cache_dir/artifacts` contains the referenced file. A wrong cache root hides existing previews. Repair artifacts after backing up. |
| EXR looks too dark/bright | Check OpenImageIO availability and source color context. Previews are tone-mapped inspection proxies, not a color-managed master. |
| Address already in use | Keep the other server or use `python scripts/serve_library.py --port 8766`; do not kill an unknown process. |
| New files do not appear | Use Add elements or CLI ingest against the intended folder/catalog; check extension rules and the job/processing log. There is no automatic watcher. |
| Analysis unavailable in the browser | Restart the server from the AI virtual environment and check its configured model manifest/snapshot. A server started from a core-only Python cannot use packages installed in another environment. |
| Saves return a conflict during a job | Wait for the job to finish; reload stale review forms. Do not run a second writer to bypass the lease. |
| Analyzer says nothing pending | A matching review/accepted result is cached. Use `--force --element-id ELEMENT_ID --limit 1` intentionally. Check your active catalog and requested ID too. |
| Existing elements unexpectedly selected | Prompts/code/options/model metadata changed, legacy signatures are absent, or an earlier result failed. Inspect `--dry-run` before a large batch. |
| Wrong material, framing or direction | Watch the source, use artist review where available, retain uncertainty and compare a forced isolated rerun. A larger confidence value is not evidence of correctness. |
| Description looks shortened | Open the inspector; card copy is clamped. Inspect the detail API/candidate raw output to distinguish UI shortening from generation/parse truncation. |
| Slow inference / out of memory | Start with one element, no challengers and no other model job. Batch `--limit` controls how many clips run, not peak memory for one clip. Preserve diagnostics before changing adapter budgets. |
| SQLite locked or missing records | Keep the DB local, stop overlapping discovery/merge jobs, inspect backups/configured paths. Do not delete WAL/SHM files from a live database. |

For an issue, include sanitized OS/architecture, Python and dependency versions, command,
traceback and reproducible public/synthetic media when permitted. Never publish a token,
private catalog, production footage or unsanitized absolute source paths.

## Accuracy, Privacy and Limitations

The current model stack remains an assistant to artist judgment. The latest diagnostic
retest processed six clips but still missed launcher/fire detail, snow direction, crop,
counts and material distinctions. Those tuned examples are not independent accuracy.
Neither `accepted` nor 100% model confidence is a guarantee.

Physical focal length, lens identity and precise camera distance cannot be reliably
recovered from arbitrary VFX plates alone. Apparent shot scale/viewpoint are visual
descriptions; verify physical claims against acquisition metadata. Likewise, action
windows are not approved editorial trims and alpha heuristics are not final keying advice.

The service has no authentication/TLS. v1 refuses non-loopback binding, rejects foreign
Host/Origin/cross-site requests, bounds JSON bodies and sends anti-framing/security headers.
These controls do not sandbox media decoders or authenticate local users. Do not port-forward
it or expose it through a tunnel. Public source publication is not public HTTP hosting.

Catalogs/logs contain paths, labels, model evidence, reviewer notes and rating identities.
Default analysis is local, but downloads contact Hugging Face and custom wrappers may
use the network. Review [security/privacy](../SECURITY.md), [model terms](THIRD-PARTY.md)
and [release readiness](RELEASE-READINESS.md) before studio deployment or publication.
