# Public v1 Release Readiness

Assessment updated: 2026-10-08. Current package version: `1.0.0rc1`.
Publication repository: https://github.com/kkthxs/vfx-element-tagger.
The public repository starts from the clean source export, not private development history.

## Recommendation

Publish the clean Apple Silicon release-candidate export after the publication checks
below. Do not tag a stable `v1.0.0` yet. The local-first architecture remains
reasonable; a universal desktop installer, CUDA port or new vector database is not
required to release an honestly scoped Mac-first application.

The product promise should be **AI-assisted cataloging with artist review**, not
accurate unattended tagging of every VFX element. The six-clip diagnostic retest still
has classification, crop, count and direction errors. Its figures are not held-out accuracy.

## Changes Prepared for Publication

- Added the MIT license already declared by package metadata and upstream patch notices.
- Aligned package metadata at `1.0.0rc1`; this candidate is not a stable `v1.0.0` release.
- Added a complete user guide, public README entry point, security policy and contribution guide.
- Froze the complete 73-package Mac/Python 3.12 runtime, upgraded vulnerable dependencies
  and reran the dependency audit: no known vulnerabilities reported on the assessment date.
- Pinned all three tested model revisions and recorded per-file SHA-256 checksums.
- Preserved artist fields and analysis across ingest refreshes; stale-source invalidation
  and ambiguous-regrouping refusal have regression coverage.
- Made artist review mandatory for new predictions; auto-accept requires an experimental flag.
- Enforced local-only binding and Host/Origin/request-body protections with HTTP negative tests.
- Made JSON saves atomic and catalog backups non-overwriting with SQLite integrity validation.
- Added a core macOS/Linux/Windows CI matrix and package build checks; hosted results are pending.
- Updated the downloader to the current Hugging Face signature and added offline regression tests.
- Added ignores for build output, backups and environment-secret files. Existing model/media/data ignores remain.
- Added a source-distribution manifest for scripts, guides, constraints and notices. Historical
  evaluation reports/data are excluded from Python source archives pending privacy/rights review;
  the existing Git history still needs separate owner approval before publication.

## Stable v1 Blockers

| Priority | Issue | Required exit condition |
| --- | --- | --- |
| Closed locally | Ingest preservation | Forced rebuild, changed-source invalidation and regrouping refusal are implemented and tested; still back up before updates. |
| Partial | Setup/model adapters | A fresh native Python environment installs the frozen runtime; real primary inference passes after security upgrades. This is not a new-Mac/OS certification or full challenger accuracy test. |
| Closed locally | Mutable model/runtime versions | Immutable revisions, SHA-256 model locks and a complete runtime pin set are supplied. Distribution hashes and clean full weight re-download are not certified. |
| Closed for local scope | HTTP request protection | Loopback-only binding, Host/Origin/cross-site guards and negative tests are implemented. No LAN/public hosting or authentication promise. |
| P1 | Analysis accuracy is not independently established and confident mistakes remain. | Use blind artist labels and per-field/retrieval metrics, including false confident accepts; establish an explicit quality/review-coverage threshold for the promised v1 scope. |
| Approved for clean source | Owner authorized public source publication; private history/media remain excluded. | Scan the exact initial commit and uploaded artifacts for paths, credentials and prohibited data. This is not permission to publish development history or article footage. |
| Closed locally | Backup/restore | SQLite payloads, artist overrides, ratings and processing events survive the restore regression. Source/cache files are separate; verify access after relocation. |
| Prepared | Public version | Metadata and changelog use `1.0.0rc1`; internal schema/iteration numbers are not public package versions. |
| Configured | Public repository, issues and private security reporting. | Use the repository's issue templates and private security advisory channel; never attach private media or unsanitized runtime data. |

Documentation makes these risks visible; it does not repair or certify them. A narrow
preview may accept some known limitations, but must state them clearly and require backups.

## Platform Scope

The default full model path is tested on Apple Silicon with MLX/Metal. Core ingest,
SQLite/JSON, review and browser code use portable Python/external tools, but Linux/Windows
support is provisional until CI and real decode/preview checks pass. Core CI does not
run AI inference or establish Metal/CUDA compatibility.

Upstream [MLX documents Linux CPU/CUDA support](https://ml-explore.github.io/mlx/build/html/install.html).
This is not a ready-made cross-platform backend for our model adapters. A future Linux
release needs tested GPU/runtime installation, timestamped input verification, memory
measurements and model-parity tests. A custom local command backend is an extension
point, not a supported end-user port. Do not describe MLX itself as universally Mac-only.

## Publication Checklist

### Before Any Public Repository

- [x] Confirm the repository name and public visibility with the owner; publish application
      source only, excluding private history, media and article assets.
- [ ] Review every tracked file and every commit, not just the working tree.
- [ ] Remove or redact private paths, asset names and studio notes where required. They
      exist in historical engineering/evaluation documents; `.gitignore` cannot hide history.
- [ ] Run a dedicated secret scan on all refs. The local targeted token-pattern check
      found no matches, but was not a comprehensive historical secret scan.
- [ ] Verify no model binaries, production media, live catalogs, caches or credentials enter the push.
- [ ] Confirm source ownership and MIT licensing; retain `NOTICE` for adapted MLX-VLM code.
- [ ] Review model cards/access terms at the exact selected revisions. Do not bundle weights.
- [ ] Keep article screenshots/test footage separate from public source and packages. The
      article assets are private local outputs authorized for that purpose, not bundled examples.
- [x] Establish the public issue and private security-reporting channels.

### Before a Release Candidate

- [ ] Complete a clean Mac setup using only the published instructions.
- [ ] Run hosted core CI and build/wheel metadata checks.
- [ ] Test ffprobe/FFmpeg ingest, image sequence gaps, alpha, EXR fallback and poster playback.
- [x] Verify incremental ingest and AI signatures, forced reruns and saved human corrections.
- [ ] Exercise canceled inference, corrupt media, unavailable challenger and missing model errors.
- [x] Test restoring SQLite payloads, artist fields, ratings and processing events; separately check media access.
- [ ] Record runtime versions, model revisions, disk space, memory and approximate per-clip time.
- [ ] Publish a known-issues section and the Mac-only tested AI scope.

### Before Stable v1

- [ ] Close the blockers above or narrow the advertised scope with an explicit release decision.
- [ ] Validate an independent artist-labeled benchmark under the frozen runtime/models.
- [ ] Define compatible catalog migrations, rollback and supported upgrade paths.
- [ ] Choose a release version, update metadata/changelog, then create its Git tag and GitHub release.
- [ ] Check the exact uploaded source/wheel artifacts for excluded data and required notices.

The publication target is `kkthxs/vfx-element-tagger`. Release-candidate publication does
not close the remaining stable-release gates or authorize publishing private history.

## Verification Evidence

The current suite has 165 passing tests in a pristine environment installed from the
final runtime lock. Coverage includes forced ingest artist/rating retention, changed-source
invalidation, regrouping refusal, model tampering, backup restore and HTTP request rejection.
These tests do not certify semantic accuracy or another OS.

A fresh dependency-free virtual environment installed that wheel, ingested a synthetic
320x240/24-fps clip, created its poster and both previews, then reused the unchanged
element on a second scan. The resulting SQLite integrity check passes. The source archive
contains scripts, guide, constraints and notices, and excludes historical evaluation
reports/data. Local documentation links resolve. This is a core packaging smoke test on
the existing Mac, not a clean-machine AI or cross-platform certification.

See [release verification](RELEASE-VERIFICATION.md),
[the held-out protocol](../evaluation/BENCHMARK-PROTOCOL.md) and
[the user guide](USER-GUIDE.md). Private diagnostic reports are deliberately excluded
from public source. Hosted CI and independent evaluation remain stable-release gates;
no stable-release approval is implied by a green unit suite.

## Safe Public Source Export

Run `python scripts/export_public_release.py --output ../vfx-element-tagger-public-v1`.
The destination must be new and outside the development repository. An allowlist and
path/credential-pattern scan run before copying. The export contains no `.git`, weights,
media, live catalog, private reports, local configuration or article screenshots. Its
publication manifest lists file hashes and exclusions. Inspect that exact tree and use
it as the starting point for a new public repository; do not push old development history.
No API keys or external hosted-analysis configuration are required or bundled. `/api/`
routes in the source are the application's own loopback HTTP endpoints.

Build only from this clean export. Then run
`python scripts/check_release_archives.py --normalize-sdist` and
`python -m twine check dist/*`. The archive check refuses private media/catalogs,
machine paths, credential patterns and unexpected binary files; normalization removes
local file-owner names/IDs from source tar headers. Repeat these checks after every build.
