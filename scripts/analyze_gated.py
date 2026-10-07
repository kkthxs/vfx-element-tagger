#!/usr/bin/env python3
"""Compatibility entry point for the packaged gated-analysis runner."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vfx_element_tagger.analysis_runner import (  # noqa: E402,F401
    main, _select_elements, _apply_outcome, _technical_context, _video_path,
    _activity_video_path, _detail_video_path, _alpha_preview_path,
    _prefer_alpha_semantic_input, _diverse_subset, _selection_features,
)

if __name__ == "__main__":
    raise SystemExit(main())
