# Release Candidate Verification

Local AI verification: 2026-10-07. Publication/core regression update: 2026-10-08.
Candidate: `1.0.0rc1`.

## Scope and Environment

- macOS 15.7.4, build 24G517; Apple M4 Pro, 48 GiB unified memory.
- Native arm64 CPython 3.12.10, FFmpeg 8.1.1.
- MLX 0.31.2 / MLX-VLM 0.5.0 with the full pinned runtime in
  `constraints/macos-arm64-py312.lock.txt`.
- Inference checks use an isolated environment, procedural synthetic footage and a private
  copy of authorized test footage. Production catalogs/artist decisions were not changed
  by inference verification.

## Results

- 168 unit/regression tests pass in a pristine environment installed from the final lock.
- Database regressions verify explicit connection closure for SQLite/JSON ratings,
  read-only loads and backups, and rollback/closure after a failed transactional save.
  Windows private-export exclusions and explicit model-directory settings are covered.
- [Hosted core CI](https://github.com/kkthxs/vfx-element-tagger/actions/runs/37703835212)
  passes all seven jobs: Python 3.11/3.12 on macOS, Linux and Windows, plus packaging.
  These jobs run the regression suite and CLI checks, not model inference or a complete
  OS-specific media/browser compatibility test.
- `pip check`: no broken requirements.
- `pip-audit` 2.10.1, PyPI advisory service: no known vulnerabilities reported for the
  complete updated runtime on this date. This is not a guarantee against unknown defects.
- All three installed snapshots pass full per-file SHA-256 checks against immutable
  revisions in the packaged model lock. No weights are bundled.
- All three adapters generate real output in that pristine environment from a four-frame,
  full-duration synthetic clip sample. These short, 96-token compatibility checks are not
  accuracy tests; token-limited challenger text is intentionally incomplete.
- Full primary gated inference completes after security updates, with three analysis
  candidates and mandatory `needs_review` output, no runtime failure.
- Observed primary test wall time: 88.01 seconds. Maximum RSS: 4.58 GiB. macOS peak memory
  footprint: 7.25 GiB. These are one-clip measurements, not minimum RAM requirements or a
  performance guarantee; longer/high-resolution/challenger jobs can differ substantially.
- The procedural flame-like clip was described as a distortion/energy vortex. This is
  a clear reminder that runtime success is not semantic correctness. Artist approval is
  mandatory by default; no accuracy percentage is inferred from this smoke test.
- SQLite restore regression preserves complete element payloads, artist overrides,
  ratings and processing events. Same-path/existing-destination backups are refused.
- Local HTTP tests reject foreign Host/Origin, cross-site metadata, negative body lengths,
  unsupported transfer framing, invalid JSON content types/votes and invalid search counts.
- Browser checks cover desktop and 390-pixel mobile layouts, RGB/alpha playback, search,
  artist-name entry, rating saves and review saves using a separate article catalog.
  Review fields fit the mobile viewport; the mobile inspector close control stays above
  the header. These are local checks, not a cross-browser accessibility certification.
- Browser import runs real primary/challenger analysis in a separate one-element catalog.
  The test performer is identified as a green-screen performance plate and remains subject
  to artist review. A repeated folder import reuses the element with zero AI work; Process
  Pending reports zero stale items. This is a workflow check, not independent accuracy evidence.
- Web jobs back up catalogs, serialize cooperating writers, preserve artist fields, scope
  imported analysis to scanned IDs and expose progress/errors/cancellation. Regression
  tests cover cancellation checkpoints, forced scopes, foreign-origin rejection and stale
  review conflicts. Import and force dialogs fit a 390-pixel mobile viewport.

## Not Certified

Complete media/AI workflows on other operating systems, clean new-Mac setup, a complete network re-download
of all model weights, independent held-out accuracy, minimum RAM, unattended auto-accept,
LAN/public hosting and arbitrary catalog relocation remain outside this evidence.

The article screenshot collection is a separate local output using authorized test footage;
it is excluded from the public source export and Python packages.
