# Contributing

Use a source checkout and Python 3.12. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,hashing]" opencv-python
python -m unittest discover -s tests
python -m compileall -q src scripts tests pilot.py
git diff --check
```

On Windows, activate with `.venv\Scripts\Activate.ps1` in PowerShell. Core CI is
configured for macOS, Linux and Windows; those jobs do not validate model inference.
Apple Silicon inference requires the separately documented MLX environment.

Keep patches scoped. Add a regression test for every behavior change. Pure unit tests
must not download weights, require private footage or assume a personal workstation path.
Use temporary directories and synthetic fixtures. Test model adapters against actual
processor inputs before making claims about video coverage or temporal accuracy.

For model/prompt changes, retain candidate diagnostics and compare on blind artist labels.
Do not report tuned development examples or self-reported confidence as general accuracy.
See [the benchmark protocol](evaluation/BENCHMARK-PROTOCOL.md).

Bug reports should include OS/architecture, Python/dependency versions, command, expected
behavior, actual behavior and a sanitized traceback. Share media only when you have the
rights to publish it. Never commit weights, media, catalogs, caches or secrets.

Contributions are under the repository's MIT license. Third-party code and model
licenses remain separate; preserve required notices.
