from __future__ import annotations

import json
import subprocess
import types
from pathlib import Path
from typing import Any
from .video_inputs import TimestampedVideoProcessor, patch_qwen_rope, sample_video
from .model_integrity import LOCK_PATH, verify_snapshot


class MLXVideoBackend:
    """Lazy MLX-VLM backend with full-duration anchors and source timestamps."""

    def __init__(self, model_path: Path, model_id: str | None = None):
        self.model_path = model_path.resolve()
        self.model_id = model_id or self.model_path.name
        self._stack: tuple[Any, Any, dict[str, Any]] | None = None
        self.last_input_diagnostics: dict[str, Any] = {}
        self.sampling_focus: list[float] = []

    def _load(self) -> tuple[Any, Any, dict[str, Any]]:
        if self._stack is None:
            locked = json.loads(LOCK_PATH.read_text(encoding="utf-8"))["models"]
            for key, spec in locked.items():
                if spec["repo_id"] == self.model_id:
                    verify_snapshot(key, self.model_path)
                    break
            from mlx_vlm import load  # type: ignore[import-not-found]
            from mlx_vlm.utils import load_config  # type: ignore[import-not-found]

            model, processor = load(str(self.model_path))
            config = load_config(self.model_path)
            if str(config.get("model_type") or "") == "minicpmo":
                _patch_minicpmo_quantized_vision_dtype(model)
            elif str(config.get("model_type") or "") in {"qwen3_5", "qwen3_vl", "qwen3_vl_moe"}:
                patch_qwen_rope(model)
            self._stack = model, processor, config
        return self._stack

    def analyze(
        self,
        video_path: Path,
        prompt: str,
        fps: float,
        max_frames: int,
        max_tokens: int,
    ) -> str:
        from mlx_vlm import generate  # type: ignore[import-not-found]
        from mlx_vlm.prompt_utils import apply_chat_template  # type: ignore[import-not-found]
        self.last_input_diagnostics = {}
        model, processor, config = self._load()
        model_type = str(config.get("model_type") or "")
        if model_type == "minicpmo":
            return self._analyze_minicpm_frames(
                model,
                processor,
                config,
                video_path,
                prompt,
                fps,
                max_frames,
                max_tokens,
            )
        if model_type not in {"qwen3_5", "qwen3_vl", "qwen3_vl_moe"}:
            raise ValueError(f"No verified native-video adapter for model type {model_type!r}")
        try:
            video_array, self.last_input_diagnostics = sample_video(
                video_path, fps, max_frames, self.sampling_focus
            )
            effective_fps = len(video_array) / (self.last_input_diagnostics["source_frame_count"] /
                                               self.last_input_diagnostics["source_fps"])
        except ValueError as exc:
            # Legacy catalogs can contain a one-frame MP4 preview for a still
            # source.  Analyse that frame as an image rather than failing the
            # whole element.
            if "nframes" not in str(exc) and "No frames" not in str(exc):
                raise
            frame = _first_video_frame(video_path)
            formatted = apply_chat_template(
                processor, config, prompt, num_images=1, enable_thinking=False
            )
            output = generate(
                model,
                processor,
                formatted,
                image=[frame],
                max_tokens=max_tokens,
                enable_thinking=False,
                verbose=False,
            )
            return getattr(output, "text", str(output)).strip()
        if len(video_array) == 1:
            self.last_input_diagnostics["representation"] = "single_frame_image"
            return self.analyze_image(video_array[0].transpose(1, 2, 0), prompt, max_tokens)
        formatted = apply_chat_template(
            processor,
            config,
            prompt,
            video=[str(video_path)],
            fps=effective_fps,
            enable_thinking=False,
        )
        aligned = TimestampedVideoProcessor(processor, self.last_input_diagnostics)
        vp = processor.video_processor
        original_budget = vp.max_pixels
        vp.max_pixels = len(video_array) * video_array.shape[2] * video_array.shape[3]
        try:
            output = generate(
                model,
                aligned,
                formatted,
                video=[video_array],
                fps=effective_fps,
                max_tokens=max_tokens,
                enable_thinking=False,
                verbose=False,
            )
        finally:
            vp.max_pixels = original_budget
        return getattr(output, "text", str(output)).strip()

    def _analyze_minicpm_frames(
        self,
        model,
        processor,
        config: dict[str, Any],
        video_path: Path,
        prompt: str,
        fps: float,
        max_frames: int,
        max_tokens: int,
    ) -> str:
        """Feed MiniCPM-o ordered frames through its multi-image path.

        mlx-vlm's generic video path supplies a ``videos`` argument that the
        current MiniCPM processor ignores. That previously produced fluent but
        vision-free hallucinations. MiniCPM does support multiple images, so a
        capped, full-duration frame set is the safe challenger adapter.
        """

        import numpy as np  # type: ignore[import-not-found]
        from mlx_vlm import generate  # type: ignore[import-not-found]
        from mlx_vlm.prompt_utils import apply_chat_template  # type: ignore[import-not-found]
        video_array, self.last_input_diagnostics = sample_video(
            video_path, fps, min(24, max_frames), self.sampling_focus
        )
        frames = [np.transpose(frame, (1, 2, 0)) for frame in video_array]
        frame_prompt = (
            f"{prompt}\n\nThe {len(frames)} supplied images are chronological frames sampled "
            "across the complete clip. Compare them in order to infer motion and temporal arc. "
            "Image timestamps in seconds, in the same order: "
            + json.dumps(self.last_input_diagnostics["timestamps_seconds"])
        )
        self.last_input_diagnostics["representation"] = "ordered_timestamped_images"
        formatted = apply_chat_template(
            processor,
            config,
            frame_prompt,
            num_images=len(frames),
            enable_thinking=False,
        )
        output = generate(
            model,
            processor,
            formatted,
            image=frames,
            max_tokens=max_tokens,
            enable_thinking=False,
            verbose=False,
        )
        return getattr(output, "text", str(output)).strip()

    def analyze_image(self, image, prompt: str, max_tokens: int = 900) -> str:
        """Analyse one PIL/numpy image without unloading the model stack."""

        from mlx_vlm import generate  # type: ignore[import-not-found]
        from mlx_vlm.prompt_utils import apply_chat_template  # type: ignore[import-not-found]

        model, processor, config = self._load()
        formatted = apply_chat_template(
            processor,
            config,
            prompt,
            num_images=1,
            enable_thinking=False,
        )
        output = generate(
            model,
            processor,
            formatted,
            image=[image],
            max_tokens=max_tokens,
            enable_thinking=False,
            verbose=False,
        )
        return getattr(output, "text", str(output)).strip()


def _first_video_frame(path: Path):
    import cv2  # type: ignore[import-not-found]

    capture = cv2.VideoCapture(str(path))
    try:
        ok, frame = capture.read()
        if not ok:
            raise ValueError(f"No frame could be decoded from {path}")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        capture.release()


def _patch_minicpmo_quantized_vision_dtype(model) -> None:
    """Backport the upstream MiniCPM quantized-vision dtype fix.

    mlx-vlm 0.5.0 derives the image tensor dtype from the quantized language
    embedding's packed uint32 storage. Convolution requires a floating dtype.
    Upstream now derives it from the vision patch embedding; keep this narrow
    instance patch until that release is installed.
    """

    import mlx.core as mx  # type: ignore[import-not-found]
    import numpy as np  # type: ignore[import-not-found]
    from mlx_vlm.models.minicpmo.minicpmo import (  # type: ignore[import-not-found]
        _to_mx_array,
        _to_numpy,
    )

    def get_vision_embedding(self, pixel_values, tgt_sizes):
        dtype = self.vision_tower.embeddings.patch_embedding.weight.dtype
        if pixel_values is None:
            return []
        vision_hidden_states = []
        for batch_idx, batch_pixels in enumerate(pixel_values):
            batch_tgt = tgt_sizes[batch_idx] if tgt_sizes is not None else []
            batch_tgt = _to_numpy(batch_tgt)
            if batch_tgt is None or len(batch_tgt) == 0:
                batch_tgt = np.zeros((0, 2), dtype=np.int32)
            else:
                batch_tgt = np.asarray(batch_tgt, dtype=np.int32)

            sample_embeddings = []
            for image_idx, cur_pixels in enumerate(batch_pixels):
                cur_pixels = _to_mx_array(cur_pixels, dtype=dtype)
                if cur_pixels is None or cur_pixels.ndim != 3:
                    continue
                if cur_pixels.shape[0] == 3:
                    cur_pixels = cur_pixels.transpose(1, 2, 0)
                cur_pixels = mx.expand_dims(cur_pixels, axis=0)
                if image_idx < len(batch_tgt):
                    cur_tgt = batch_tgt[image_idx]
                else:
                    seq_len = max(int(cur_pixels.shape[2] // self.config.patch_size), 1)
                    cur_tgt = np.array([1, seq_len], dtype=np.int32)
                cur_tgt = mx.array(cur_tgt, dtype=mx.int32)[None, :]
                patch_len = int((cur_tgt[0, 0] * cur_tgt[0, 1]).item())
                patch_attention_mask = mx.ones((1, 1, patch_len), dtype=mx.bool_)
                hidden = self.vision_tower(
                    cur_pixels,
                    patch_attention_mask=patch_attention_mask,
                    tgt_sizes=cur_tgt,
                )
                hidden = self.resampler(hidden, cur_tgt)
                sample_embeddings.append(hidden[0])
            vision_hidden_states.append(
                mx.stack(sample_embeddings, axis=0) if sample_embeddings else []
            )
        return vision_hidden_states

    model.get_vision_embedding = types.MethodType(get_vision_embedding, model)


class CommandVideoBackend:
    """Adapter for a local challenger wrapper.

    The command receives the prompt on stdin and the video path as its final
    argument.  This keeps model-specific stacks such as MiniCPM or a gated
    Marlin installation outside the catalog code while retaining local-only
    execution.  Stdout must contain the requested JSON response.
    """

    def __init__(self, command: list[str], model_id: str, timeout_seconds: int = 900):
        if not command:
            raise ValueError("A command backend requires a non-empty command list.")
        self.command = command
        self.model_id = model_id
        self.timeout_seconds = timeout_seconds

    def analyze(
        self,
        video_path: Path,
        prompt: str,
        fps: float,
        max_frames: int,
        max_tokens: int,
    ) -> str:
        command = [
            part.format(fps=fps, max_frames=max_frames, max_tokens=max_tokens)
            for part in self.command
        ]
        command.append(str(video_path.resolve()))
        completed = subprocess.run(
            command,
            input=prompt,
            capture_output=True,
            check=False,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=self.timeout_seconds,
        )
        if completed.returncode != 0:
            message = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(f"challenger command failed ({completed.returncode}): {message}")
        return completed.stdout.strip()


def backend_from_manifest(entry: dict[str, Any]):
    backend = str(entry.get("backend") or "mlx_video")
    model_id = str(entry.get("repo_id") or entry.get("model_id") or "local_model")
    if backend == "mlx_video":
        path = entry.get("path")
        if not path:
            raise ValueError(f"Model {model_id} has no local path in its manifest entry.")
        return MLXVideoBackend(Path(path), model_id=model_id)
    if backend == "command":
        command = entry.get("command")
        if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
            raise ValueError(f"Command model {model_id} needs a JSON string-list `command`.")
        return CommandVideoBackend(
            command,
            model_id=model_id,
            timeout_seconds=int(entry.get("timeout_seconds") or 900),
        )
    raise ValueError(f"Unsupported video backend: {backend}")


def load_analysis_stack(manifest_path: Path):
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = payload.get("models") or {}
    stack = payload.get("analysis_stack") or {}
    primary_key = stack.get("primary") or "analysis_primary"
    if primary_key not in entries:
        raise KeyError(f"Primary model `{primary_key}` is missing from {manifest_path}.")
    primary = backend_from_manifest(entries[primary_key])
    challengers = []
    for key in stack.get("challengers") or []:
        entry = entries.get(key)
        if not entry:
            continue
        challengers.append(backend_from_manifest(entry))
    return primary, challengers, payload
