# Changelog

## 1.0.0rc1

- Add browser-controlled local folder import, scoped incremental AI analysis, pending
  batches and explicitly confirmed forced reprocessing for selected/all elements.
- Add background progress, bounded logs, automatic batch backups, cooperative stopping
  and catalog writer leases; pause artist writes during jobs and reject stale reviews.
- Package the gated-analysis runner so installed wheels can launch model jobs directly.

First public-release candidate, scoped to AI-assisted cataloging on one Apple Silicon Mac.

- Preserve analysis, artist overrides and review history through forced ingest refreshes.
- Invalidate changed-source predictions and refuse ambiguous master/proxy regrouping.
- Require artist review of new AI results by default; automatic acceptance is experimental.
- Enforce local-only serving, Host/Origin checks, request bounds and anti-framing headers.
- Make JSON saves atomic and prevent destructive/overwriting backups; verify SQLite snapshots.
- Pin model revisions and check SHA-256 files before known model snapshots load.
- Freeze the complete Mac/Python 3.12 runtime and audit its dependencies.
- Add user/support/security documentation, regression tests and cross-platform core CI.
- Export public source from an allowlist without private development history or media.

Known limitations: AI confidence is uncalibrated; descriptions, materials, counts, framing,
directions and action timing need human verification. Only selected review fields are
artist-confirmed, not every sentence or timing estimate. No LAN/cloud service or supported
Windows/Linux AI backend ships in this candidate. Model weights and test footage are not bundled.
