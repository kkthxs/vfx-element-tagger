#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2
import numpy as np
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoProcessor, Qwen3VLForConditionalGeneration

from vfx_element_tagger.category_resolver import resolve_caption_category, resolve_initial_category
from vfx_element_tagger.models import Element
from vfx_element_tagger.stages import (
    caption_element,
    caption_embedding_text,
    classify_element,
    derive_content_facets,
    embed_element,
    enrich_search_facets,
    store_qwen_caption_tags,
)
from vfx_element_tagger.store import LibraryStore
from vfx_element_tagger.taxonomy import load_taxonomy


def main() -> int:
    parser = argparse.ArgumentParser(description="Run local AI analysis using downloaded models.")
    parser.add_argument("--library", default=".cache/vfx-element-tagger/library.json")
    parser.add_argument("--models-manifest", default=".cache/models/manifest.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--element-id", action="append", default=[], help="Process only the given element id. Can be repeated.")
    parser.add_argument("--skip-vlm", action="store_true", help="Skip Qwen3-VL captions.")
    parser.add_argument("--caption-tokens", type=int, default=256)
    parser.add_argument("--caption-frame-count", type=int, default=5, help="Ordered frames to feed Qwen-VL from start to end.")
    parser.add_argument("--no-motion-context", action="store_true", help="Do not include optical-flow context in the Qwen-VL prompt.")
    parser.add_argument("--vlm-backend", choices=("auto", "hf", "mlx"), default="auto", help="Qwen-VL runtime backend.")
    parser.add_argument("--vlm-device", choices=("auto", "cpu", "mps"), default="auto", help="Device for the HuggingFace VLM backend.")
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Re-run model analysis for selected elements even if they already have model-backed outputs.",
    )
    args = parser.parse_args()

    library_path = Path(args.library).resolve()
    manifest_path = Path(args.models_manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    store = LibraryStore(library_path)
    elements = store.load()
    if args.element_id:
        requested = set(args.element_id)
        selected_elements = [element for element in elements if element.element_id in requested]
        missing = sorted(requested - {element.element_id for element in selected_elements})
        if missing:
            raise SystemExit(f"Unknown element id(s): {', '.join(missing)}")
    elif args.limit:
        selected_elements = elements[: args.limit]
    else:
        selected_elements = elements

    if args.force_recompute:
        elements_to_process = selected_elements
    else:
        elements_to_process = [
            element
            for element in selected_elements
            if _needs_model_analysis(element, include_vlm=not args.skip_vlm)
        ]

    print(f"Loaded {len(elements)} elements from {library_path}", flush=True)
    skipped = len(selected_elements) - len(elements_to_process)
    if args.force_recompute:
        print(f"Processing {len(elements_to_process)} selected elements with local models (force recompute)", flush=True)
    else:
        print(
            f"Processing {len(elements_to_process)} selected elements needing model analysis; "
            f"skipping {skipped} already model-backed elements",
            flush=True,
        )
    if not elements_to_process:
        print("No elements need model analysis. Use --force-recompute to refresh existing model outputs.", flush=True)
        return 0

    run_siglip(elements_to_process, Path(manifest["models"]["siglip2"]["path"]))
    run_optical_flow(elements_to_process)
    if not args.skip_vlm:
        run_qwen_vl(
            elements_to_process,
            Path(manifest["models"]["qwen3_vl"]["path"]),
            max_new_tokens=args.caption_tokens,
            frame_count=args.caption_frame_count,
            include_motion_context=not args.no_motion_context,
            backend=args.vlm_backend,
            device_preference=args.vlm_device,
        )
    else:
        for element in elements_to_process:
            if not element.caption or element.captioning_model_version.startswith("deterministic"):
                caption_element(element)
    run_qwen_embeddings(elements_to_process, Path(manifest["models"]["qwen3_embedding"]["path"]))

    if store.is_sqlite:
        for element in elements_to_process:
            store.save_element(element)
    else:
        store.save(elements)
    print(f"Updated library: {library_path}", flush=True)
    return 0


def _needs_model_analysis(element: Element, include_vlm: bool = True) -> bool:
    if not _has_siglip_analysis(element):
        return True
    if not element.measured_features or not element.motion_facets:
        return True
    if include_vlm and not _has_qwen_caption(element):
        return True
    if not _has_qwen_caption_embedding(element):
        return True
    return False


def _has_siglip_analysis(element: Element) -> bool:
    version = element.embedding_model_version.lower()
    source = str(element.provenance.get("category", {}).get("source", "")).lower()
    return "siglip" in version and "siglip" in source and len(element.image_embed) == 1024


def _has_qwen_caption(element: Element) -> bool:
    return "qwen" in element.captioning_model_version.lower() and bool(element.caption.strip())


def _has_qwen_caption_embedding(element: Element) -> bool:
    version = element.embedding_model_version.lower()
    return "qwen" in version and len(element.caption_embed) == 4096


def run_siglip(elements: list[Element], model_path: Path) -> None:
    taxonomy = load_taxonomy()
    labels = [_siglip_category_prompt(category) for category in taxonomy.categories]
    print("Loading SigLIP 2", flush=True)
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    model = AutoModel.from_pretrained(model_path, local_files_only=True, dtype=torch.float32).eval()
    for index, element in enumerate(elements, start=1):
        poster = _poster_path(element)
        if not poster:
            classify_element(element)
            embed_element(element)
            continue
        image = Image.open(poster).convert("RGB")
        inputs = processor(text=labels, images=image, padding="max_length", return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            probs = outputs.logits_per_image[0].float().softmax(dim=-1)
            if hasattr(model, "get_image_features"):
                image_inputs = processor(images=image, return_tensors="pt")
                image_features = _pooled_feature_tensor(model.get_image_features(**image_inputs))
            else:
                image_features = _pooled_feature_tensor(outputs.image_embeds)
        best_index = int(probs.argmax().item())
        siglip_category = taxonomy.categories[best_index]
        siglip_confidence = float(probs[best_index].item())
        category, confidence, provenance = resolve_initial_category(
            siglip_category,
            siglip_confidence,
            poster,
        )
        element.category = category
        element.category_confidence = confidence
        element.provenance["category"] = {
            **provenance,
            "confidence": confidence,
        }
        element.image_embed = (
            torch.nn.functional.normalize(image_features.float(), dim=-1)[0].cpu().tolist()
        )
        element.embedding_model_version = "siglip2-large-patch16-384 + qwen3-embedding-8b"
        _preserve_metadata_facets(element)
        print(
            f"[siglip {index}/{len(elements)}] {element.element_id} -> "
            f"{element.category} {element.category_confidence:.2f} "
            f"(siglip {siglip_category} {siglip_confidence:.2f})",
            flush=True,
        )
    del model
    gc.collect()


def _siglip_category_prompt(category: str) -> str:
    return _SIGLIP_CATEGORY_PROMPTS.get(category, category.replace("_", " "))


def _pooled_feature_tensor(output) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    pooled = getattr(output, "pooler_output", None)
    if isinstance(pooled, torch.Tensor):
        return pooled
    last_hidden = getattr(output, "last_hidden_state", None)
    if isinstance(last_hidden, torch.Tensor):
        return last_hidden.mean(dim=1)
    if isinstance(output, (tuple, list)):
        for item in output:
            if isinstance(item, torch.Tensor):
                return item
    raise TypeError(f"Unable to extract feature tensor from {type(output).__name__}")


def run_qwen_vl(
    elements: list[Element],
    model_path: Path,
    max_new_tokens: int,
    frame_count: int = 5,
    include_motion_context: bool = True,
    backend: str = "auto",
    device_preference: str = "auto",
) -> None:
    if backend == "mlx":
        run_qwen_vl_mlx(
            elements,
            model_path,
            max_new_tokens=max_new_tokens,
            frame_count=frame_count,
            include_motion_context=include_motion_context,
        )
        return
    if backend == "auto" and _can_use_mlx(verbose=True):
        run_qwen_vl_mlx(
            elements,
            model_path,
            max_new_tokens=max_new_tokens,
            frame_count=frame_count,
            include_motion_context=include_motion_context,
            check_device=False,
        )
        return
    run_qwen_vl_hf(
        elements,
        model_path,
        max_new_tokens=max_new_tokens,
        frame_count=frame_count,
        include_motion_context=include_motion_context,
        device_preference=device_preference,
    )


def run_qwen_vl_hf(
    elements: list[Element],
    model_path: Path,
    max_new_tokens: int,
    frame_count: int,
    include_motion_context: bool,
    device_preference: str,
) -> None:
    device = _select_torch_device(device_preference)
    dtype = torch.float16 if device == "mps" else torch.bfloat16
    print(f"Loading Qwen3-VL 8B via HuggingFace on {device}", flush=True)
    if device == "cpu" and frame_count > 1:
        print(
            "[qwen-vl] CPU backend is slow with multi-frame prompts; use --caption-frame-count 1 "
            "for faster drafts or --vlm-backend mlx from a Metal-capable macOS process.",
            flush=True,
        )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=dtype,
    ).eval()
    if device != "cpu":
        model = model.to(device)
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    from qwen_vl_utils import process_vision_info

    with tempfile.TemporaryDirectory(prefix="vfx_qwen_frames_") as temp_root:
        temp_dir = Path(temp_root)
        for index, element in enumerate(elements, start=1):
            image_paths = _caption_image_paths(element, temp_dir / element.element_id, frame_count)
            if not image_paths:
                caption_element(element)
                continue
            prompt = _caption_prompt(
                element,
                frame_count=len(image_paths),
                include_motion_context=include_motion_context,
            )
            messages = [_qwen_message(prompt, image_paths)]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            if device != "cpu" and hasattr(inputs, "to"):
                inputs = inputs.to(device)
            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )
            caption = processor.batch_decode(
                generated[:, inputs.input_ids.shape[1] :],
                skip_special_tokens=True,
            )[0].strip()
            store_qwen_caption_tags(element, caption)
            _apply_caption_category_resolution(element)
            context_label = "flow-context" if include_motion_context else "visual-only"
            element.captioning_model_version = (
                f"Qwen3-VL-8B-Instruct hf-{device} {len(image_paths)}-frame {context_label} local"
            )
            print(f"[qwen-vl {index}/{len(elements)}] {element.element_id}: {caption[:90]}", flush=True)
    del model
    gc.collect()


def run_qwen_vl_mlx(
    elements: list[Element],
    model_path: Path,
    max_new_tokens: int,
    frame_count: int,
    include_motion_context: bool,
    check_device: bool = True,
) -> None:
    if check_device and not _can_use_mlx(verbose=True):
        raise RuntimeError("MLX is installed, but this process cannot access a Metal device.")
    print("Loading Qwen3-VL 8B via MLX", flush=True)
    from mlx_vlm import generate, load
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import load_config

    model, processor = load(str(model_path))
    config = load_config(model_path)
    with tempfile.TemporaryDirectory(prefix="vfx_qwen_frames_") as temp_root:
        temp_dir = Path(temp_root)
        for index, element in enumerate(elements, start=1):
            image_paths = _caption_image_paths(element, temp_dir / element.element_id, frame_count)
            if not image_paths:
                caption_element(element)
                continue
            prompt = _caption_prompt(
                element,
                frame_count=len(image_paths),
                include_motion_context=include_motion_context,
            )
            formatted_prompt = apply_chat_template(
                processor,
                config,
                prompt,
                num_images=len(image_paths),
            )
            result = generate(
                model,
                processor,
                formatted_prompt,
                image=[str(path) for path in image_paths],
                max_tokens=max_new_tokens,
                verbose=False,
            )
            caption = getattr(result, "text", str(result)).strip()
            store_qwen_caption_tags(element, caption)
            _apply_caption_category_resolution(element)
            context_label = "flow-context" if include_motion_context else "visual-only"
            element.captioning_model_version = (
                f"Qwen3-VL-8B-Instruct mlx {len(image_paths)}-frame {context_label} local"
            )
            print(f"[qwen-vl-mlx {index}/{len(elements)}] {element.element_id}: {caption[:90]}", flush=True)
    gc.collect()


def _apply_caption_category_resolution(element: Element) -> None:
    correction = resolve_caption_category(
        element.category,
        element.category_confidence,
        element.caption,
        element.content_facets,
    )
    if correction is None:
        return
    category, confidence, provenance = correction
    previous_provenance = dict(element.provenance.get("category", {}))
    element.category = category
    element.category_confidence = confidence
    element.provenance["category"] = {
        **provenance,
        "confidence": confidence,
        "previous_provenance": previous_provenance,
    }
    derive_content_facets(element)
    if element.measured_features:
        element.motion_facets = _map_motion(element, element.measured_features)
        element.motion_warnings = _motion_warnings(element.measured_features)
    enrich_search_facets(element)


def run_qwen_embeddings(elements: list[Element], model_path: Path) -> None:
    print("Loading Qwen3-Embedding 8B", flush=True)
    model = SentenceTransformer(str(model_path), device="cpu")
    for element in elements:
        enrich_search_facets(element)
    captions = [caption_embedding_text(element) for element in elements]
    vectors = model.encode(captions, normalize_embeddings=True, show_progress_bar=True)
    for element, vector in zip(elements, vectors):
        element.caption_embed = vector.astype(float).tolist()
        element.embedding_model_version = "siglip2-large-patch16-384 + qwen3-embedding-8b"
    del model
    gc.collect()


def run_optical_flow(elements: list[Element]) -> None:
    for index, element in enumerate(elements, start=1):
        video_path = element.preview_movie_480_path or element.preview_movie_1080_path
        if not video_path:
            continue
        features = _measure_flow(Path(video_path))
        element.measured_features = features
        element.motion_facets = _map_motion(element, features)
        element.motion_warnings = _motion_warnings(features)
        print(f"[flow {index}/{len(elements)}] {element.element_id}", flush=True)


def _measure_flow(path: Path, max_pairs: int = 24) -> dict:
    capture = cv2.VideoCapture(str(path))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total < 2:
        capture.release()
        return {"sampled_pairs": 0}
    stride = max(1, total // max_pairs)
    previous = None
    first_gray = None
    last_gray = None
    magnitudes = []
    angles = []
    divergences = []
    curls = []
    global_magnitudes = []
    global_angles = []
    local_magnitudes = []
    luminance_means = []
    foreground_areas = []
    frame_index = 0
    sampled = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index % stride:
            frame_index += 1
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (320, 180), interpolation=cv2.INTER_AREA)
        if first_gray is None:
            first_gray = gray
        last_gray = gray
        luminance_means.append(float(np.mean(gray)))
        foreground_areas.append(float((gray > 20).sum()) / float(gray.size))
        if previous is not None:
            flow = cv2.calcOpticalFlowFarneback(
                previous,
                gray,
                None,
                0.5,
                3,
                15,
                3,
                5,
                1.2,
                0,
            )
            fx = flow[..., 0]
            fy = flow[..., 1]
            mean_fx = float(np.mean(fx))
            mean_fy = float(np.mean(fy))
            global_mag = float(math.hypot(mean_fx, mean_fy))
            global_angle = float(np.rad2deg(math.atan2(mean_fy, mean_fx)))
            local_fx = fx - mean_fx
            local_fy = fy - mean_fy
            mag, angle = cv2.cartToPolar(fx, fy, angleInDegrees=True)
            local_mag, _local_angle = cv2.cartToPolar(local_fx, local_fy, angleInDegrees=True)
            magnitudes.append(float(np.median(mag)))
            angles.append(float(np.rad2deg(math.atan2(float(np.mean(fy)), float(np.mean(fx))))))
            local_magnitudes.append(float(np.median(local_mag)))
            global_magnitudes.append(global_mag)
            global_angles.append(global_angle)
            div = np.gradient(local_fx, axis=1) + np.gradient(local_fy, axis=0)
            curl = np.gradient(local_fy, axis=1) - np.gradient(local_fx, axis=0)
            divergences.append(float(np.mean(div)))
            curls.append(float(np.mean(np.abs(curl))))
            sampled += 1
        previous = gray
        frame_index += 1
    capture.release()
    if not magnitudes:
        return {"sampled_pairs": 0}
    start_end_visual_distance = 0.0
    if first_gray is not None and last_gray is not None:
        start_end_visual_distance = float(np.mean(cv2.absdiff(first_gray, last_gray))) / 255.0
    expansion_ratio = 1.0
    if foreground_areas and foreground_areas[0] > 0:
        expansion_ratio = float(foreground_areas[-1] / max(1e-6, foreground_areas[0]))
    temporal_acceleration = 0.0
    if len(magnitudes) > 2:
        temporal_acceleration = float(np.median(np.diff(magnitudes)))
    median_global = float(np.median(global_magnitudes)) if global_magnitudes else 0.0
    median_local = float(np.median(local_magnitudes)) if local_magnitudes else 0.0
    return {
        "sampled_pairs": sampled,
        "dominant_flow_angle": float(np.median(angles)),
        "median_flow_magnitude": float(np.median(magnitudes)),
        "magnitude_variance": float(np.var(magnitudes)),
        "flow_divergence_mean": float(np.mean(divergences)),
        "flow_curl_mean": float(np.mean(curls)),
        "global_translation_angle": float(np.median(global_angles)) if global_angles else None,
        "global_translation_magnitude": median_global,
        "local_flow_magnitude": median_local,
        "global_motion_ratio": float(median_global / max(1e-6, median_global + median_local)),
        "brightness_variance": float(np.var(luminance_means)),
        "mean_luminance_start": float(luminance_means[0]) if luminance_means else None,
        "mean_luminance_end": float(luminance_means[-1]) if luminance_means else None,
        "foreground_area_start": float(foreground_areas[0]) if foreground_areas else None,
        "foreground_area_end": float(foreground_areas[-1]) if foreground_areas else None,
        "expansion_ratio": expansion_ratio,
        "start_end_visual_distance": start_end_visual_distance,
        "temporal_acceleration": temporal_acceleration,
        "implementation": "opencv-farneback-0.1.5",
    }


def _map_motion(element: Element, features: dict) -> dict:
    mapper = {
        "smoke": _map_smoke,
        "fire": _map_fire,
        "explosion": _map_explosion,
        "dust": _map_dust,
        "debris": _map_debris,
        "water": _map_water,
        "atmosphere": _map_atmosphere,
        "lens_effect": _map_lens_effect,
        "practical_light": _map_practical_light,
        "plate": _map_plate,
        "impact": _map_impact,
        "sparks": _map_sparks,
    }.get(element.category, _map_generic)
    return mapper(features)


def _base_flow_facets(features: dict) -> tuple[dict, float, str, str]:
    magnitude = float(features.get("median_flow_magnitude") or 0.0)
    angle = float(features.get("dominant_flow_angle") or 0.0)
    speed = _speed_label(magnitude)
    direction = _direction_from_angle(angle, magnitude)
    confidence = _confidence(features)
    return (
        {
            "speed": _facet(speed, confidence),
            "direction": _facet(direction, confidence),
        },
        confidence,
        speed,
        direction,
    )


def _map_smoke(features: dict) -> dict:
    facets, confidence, _speed, direction = _base_flow_facets(features)
    divergence = float(features.get("flow_divergence_mean") or 0.0)
    curl = float(features.get("flow_curl_mean") or 0.0)
    expansion = float(features.get("expansion_ratio") or 1.0)
    if expansion > 1.25 or divergence > 0.01:
        character = "billowing"
    elif direction == "up":
        character = "rising"
    elif curl > 0.08:
        character = "rolling"
    elif expansion < 0.85:
        character = "dissipating"
    else:
        character = "drifting"
    facets["character"] = _facet(character, confidence)
    facets["density_evolution"] = _facet(_evolution_from_expansion(expansion), confidence)
    facets["loopable"] = _facet(_loopability(features), confidence * 0.8)
    return facets


def _map_fire(features: dict) -> dict:
    facets, confidence, speed, _direction = _base_flow_facets(features)
    flicker = float(features.get("brightness_variance") or 0.0)
    acceleration = float(features.get("temporal_acceleration") or 0.0)
    character = "flickering" if flicker > 8 else "surging" if acceleration > 0.05 else "sustained"
    facets["character"] = _facet(character, confidence)
    facets["wind_influence"] = _facet("strong" if speed == "fast" else "gentle" if speed == "medium" else "none", confidence * 0.75)
    facets["intensity_arc"] = _facet(_brightness_arc(features), confidence)
    facets["containment"] = _facet("column" if _direction_from_angle(float(features.get("dominant_flow_angle") or 0.0), float(features.get("median_flow_magnitude") or 0.0)) == "up" else "contained", confidence * 0.75)
    return facets


def _map_explosion(features: dict) -> dict:
    facets, confidence, speed, _direction = _base_flow_facets(features)
    expansion = float(features.get("expansion_ratio") or 1.0)
    facets["character"] = _facet("explosive_onset" if speed == "fast" or expansion > 1.35 else "surging", confidence)
    facets["intensity_arc"] = _facet(_brightness_arc(features), confidence)
    facets["direction"] = _facet("expanding" if expansion > 1.2 else facets["direction"]["value"], confidence)
    return facets


def _map_dust(features: dict) -> dict:
    facets, confidence, _speed, direction = _base_flow_facets(features)
    curl = float(features.get("flow_curl_mean") or 0.0)
    expansion = float(features.get("expansion_ratio") or 1.0)
    character = "swirling" if curl > 0.08 else "settling" if direction == "down" else "kicked_up" if direction == "up" else "drifting"
    facets["character"] = _facet(character, confidence)
    facets["direction"] = _facet("radial" if expansion > 1.25 else direction, confidence)
    return facets


def _map_debris(features: dict) -> dict:
    facets, confidence, speed, direction = _base_flow_facets(features)
    curl = float(features.get("flow_curl_mean") or 0.0)
    trajectory = "falling_gravity" if direction == "down" else "exploding_outward" if facets["direction"]["value"] == "expanding" else "tumbling" if curl > 0.06 else "linear"
    facets["trajectory"] = _facet(trajectory, confidence)
    facets["gravity_feel"] = _facet("heavy_impact" if speed == "fast" and direction == "down" else "weighted" if direction == "down" else "floating", confidence)
    facets["quantity_arc"] = _facet(_evolution_from_expansion(float(features.get("expansion_ratio") or 1.0)), confidence)
    return facets


def _map_water(features: dict) -> dict:
    facets, confidence, _speed, direction = _base_flow_facets(features)
    expansion = float(features.get("expansion_ratio") or 1.0)
    water_direction = {
        "up": "vertical_up",
        "down": "vertical_down",
        "left": "lateral",
        "right": "lateral",
    }.get(direction, "radial" if expansion > 1.2 else "lateral")
    facets["direction"] = _facet(water_direction, confidence)
    facets["gravity_feel"] = _facet("atomised" if float(features.get("flow_curl_mean") or 0.0) > 0.08 else "gravity_driven" if water_direction == "vertical_down" else "sheet", confidence)
    facets["loopable"] = _facet(_loopability(features), confidence * 0.8)
    return facets


def _map_atmosphere(features: dict) -> dict:
    facets, confidence, speed, direction = _base_flow_facets(features)
    character = "static" if speed == "still" else "settling" if direction == "down" else "advecting" if direction in {"left", "right"} else "drifting"
    return {"character": _facet(character, confidence), "speed": _facet("still" if speed == "still" else speed, confidence)}


def _map_lens_effect(features: dict) -> dict:
    _facets, confidence, _speed, _direction = _base_flow_facets(features)
    flicker = float(features.get("brightness_variance") or 0.0)
    global_motion = float(features.get("global_translation_magnitude") or 0.0)
    character = "flickering" if flicker > 8 else "sweeping" if global_motion > 0.4 else "pulsing" if flicker > 2 else "static"
    return {"character": _facet(character, confidence)}


def _map_practical_light(features: dict) -> dict:
    _facets, confidence, speed, _direction = _base_flow_facets(features)
    flicker = float(features.get("brightness_variance") or 0.0)
    behaviour = "strobing" if flicker > 20 else "flickering" if flicker > 5 else "sweeping" if speed == "fast" else "static"
    return {"behaviour": _facet(behaviour, confidence), "change_profile": _facet(_brightness_arc(features), confidence)}


def _map_plate(features: dict) -> dict:
    _facets, confidence, speed, direction = _base_flow_facets(features)
    ratio = float(features.get("global_motion_ratio") or 0.0)
    if speed == "still":
        camera_motion = "locked_off"
    elif ratio > 0.7:
        camera_motion = "pan" if direction in {"left", "right"} else "tilt"
    else:
        camera_motion = "handheld" if float(features.get("magnitude_variance") or 0.0) > 0.3 else "slow_drift"
    parallax = "high" if ratio < 0.35 and speed != "still" else "medium" if ratio < 0.7 and speed != "still" else "none"
    return {"camera_motion": _facet(camera_motion, confidence), "parallax": _facet(parallax, confidence), "speed": _facet("static" if speed == "still" else "moderate" if speed == "medium" else speed, confidence)}


def _map_impact(features: dict) -> dict:
    facets, confidence, speed, direction = _base_flow_facets(features)
    expansion = float(features.get("expansion_ratio") or 1.0)
    facets["onset"] = _facet("instant" if speed == "fast" else "fast_buildup" if speed == "medium" else "slow_buildup", confidence)
    facets["dissipation"] = _facet("smoke_trail" if expansion < 0.9 else "fast" if speed == "fast" else "medium", confidence)
    facets["debris_pattern"] = _facet("vertical_column" if direction == "up" else "radial" if expansion > 1.2 else "forward_cone", confidence)
    return facets


def _map_sparks(features: dict) -> dict:
    facets, confidence, _speed, direction = _base_flow_facets(features)
    trajectory = "falling" if direction == "down" else "arcing" if float(features.get("flow_curl_mean") or 0.0) > 0.06 else "shower" if direction == "up" else "scattering"
    facets["trajectory"] = _facet(trajectory, confidence)
    facets["quantity_arc"] = _facet("burst" if float(features.get("start_end_visual_distance") or 0.0) > 0.2 else _evolution_from_expansion(float(features.get("expansion_ratio") or 1.0)), confidence)
    return facets


def _map_generic(features: dict) -> dict:
    facets, _confidence, _speed, _direction = _base_flow_facets(features)
    return facets


def _facet(value: str, confidence: float) -> dict:
    return {"value": value, "source": "optical_flow", "confidence": _clamp_conf(confidence)}


def _speed_label(magnitude: float) -> str:
    if magnitude < 0.05:
        return "still"
    if magnitude < 0.35:
        return "slow"
    if magnitude < 1.5:
        return "medium"
    return "fast"


def _confidence(features: dict) -> float:
    magnitude = float(features.get("median_flow_magnitude") or 0.0)
    sampled = float(features.get("sampled_pairs") or 0.0)
    confidence = 0.25 + min(0.5, magnitude / 3.0) + min(0.15, sampled / 160.0)
    if float(features.get("global_motion_ratio") or 0.0) > 0.8:
        confidence *= 0.8
    return _clamp_conf(confidence)


def _clamp_conf(value: float) -> float:
    return min(0.9, max(0.25, float(value)))


def _evolution_from_expansion(expansion: float) -> str:
    if expansion > 1.15:
        return "building"
    if expansion < 0.85:
        return "dissipating"
    return "sustained"


def _brightness_arc(features: dict) -> str:
    start = features.get("mean_luminance_start")
    end = features.get("mean_luminance_end")
    if start is None or end is None:
        return "steady"
    delta = float(end) - float(start)
    if delta > 4:
        return "building"
    if delta < -4:
        return "dissipating"
    return "sustained"


def _loopability(features: dict) -> str:
    distance = float(features.get("start_end_visual_distance") or 0.0)
    if distance < 0.04:
        return "looped"
    if distance > 0.16:
        return "one_shot"
    return "ambiguous"


def _motion_warnings(features: dict) -> list[str]:
    warnings: list[str] = []
    if not features.get("sampled_pairs", 0):
        warnings.append("optical flow unavailable")
    if float(features.get("global_motion_ratio") or 0.0) > 0.8:
        warnings.append("global camera/plate motion dominates local motion")
    if float(features.get("brightness_variance") or 0.0) > 30:
        warnings.append("high flicker may bias optical flow")
    if float(features.get("median_flow_magnitude") or 0.0) < 0.05:
        warnings.append("very low motion signal")
    return warnings


def _direction_from_angle(angle: float, magnitude: float) -> str:
    if magnitude < 0.05:
        return "no_dominant"
    if -45 <= angle < 45:
        return "right"
    if 45 <= angle < 135:
        return "down"
    if angle >= 135 or angle < -135:
        return "left"
    return "up"


def _can_use_mlx(verbose: bool = False) -> bool:
    try:
        import importlib.util

        if importlib.util.find_spec("mlx_vlm") is None:
            if verbose:
                print("[mlx] unavailable: mlx-vlm is not installed", flush=True)
            return False
        probe = subprocess.run(
            [
                sys.executable,
                "-c",
                "import mlx.core as mx; mx.eval(mx.array([1])); import mlx_vlm",
            ],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        if verbose:
            print(f"[mlx] unavailable: {exc}", flush=True)
        return False
    if probe.returncode == 0:
        return True
    lines = (probe.stderr or probe.stdout).strip().splitlines()
    if verbose and lines:
        print(f"[mlx] unavailable: {lines[-1]}", flush=True)
    return False


def _select_torch_device(preference: str) -> str:
    if preference == "cpu":
        return "cpu"
    if preference == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("Requested --vlm-device mps, but PyTorch cannot access MPS/Metal.")
        return "mps"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _qwen_message(prompt: str, image_paths: list[Path]) -> dict:
    return {
        "role": "user",
        "content": [
            *({"type": "image", "image": str(path)} for path in image_paths),
            {"type": "text", "text": prompt},
        ],
    }


def _caption_image_paths(element: Element, output_dir: Path, frame_count: int) -> list[Path]:
    frame_count = max(1, frame_count)
    video_path = element.preview_movie_480_path or element.preview_movie_1080_path
    if video_path and frame_count > 1:
        sampled = _sample_video_frames(Path(video_path), output_dir, frame_count)
        if sampled:
            return sampled
    poster = _poster_path(element)
    return [poster] if poster else []


def _sample_video_frames(path: Path, output_dir: Path, frame_count: int) -> list[Path]:
    if not path.exists():
        return []
    capture = cv2.VideoCapture(str(path))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        capture.release()
        return []
    if frame_count == 1:
        indices = [total // 2]
    else:
        indices = sorted(
            {
                int(round(position * (total - 1) / max(1, frame_count - 1)))
                for position in range(frame_count)
            }
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_paths: list[Path] = []
    for order, frame_index in enumerate(indices, start=1):
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            continue
        out = output_dir / f"frame_{order:02d}.jpg"
        if cv2.imwrite(str(out), frame):
            frame_paths.append(out)
    capture.release()
    return frame_paths


def _caption_prompt(
    element: Element,
    frame_count: int = 1,
    include_motion_context: bool = True,
) -> str:
    category = element.category.replace("_", " ")
    axes = _CAPTION_AXES.get(element.category, _CAPTION_AXES["unknown"])
    temporal_context = (
        f"You are given {frame_count} ordered sampled frame(s) from the element, "
        "from start to end. Describe visible temporal change across those frames when present. "
    )
    filmic_context = (
        "Also describe framing distance, subject placement, edge contact or cropping, "
        "implied lens/framing feel, camera angle, depth cues, and whether it reads as "
        "foreground, midground, background, insert, wide, medium, or close-up. "
    )
    motion_context = _motion_context_prompt(element) if include_motion_context else ""
    return (
        "You are describing a VFX element for a compositor searching a local element library. "
        f"The tentative category is {category} with confidence {element.category_confidence:.2f}. "
        "Use that category as a routing hint, not ground truth; if the ordered frames contradict it, "
        "describe the actual visible content and compositing use instead of inventing category-specific details. "
        f"{temporal_context}"
        f"{motion_context}"
        f"In one concise paragraph, describe: {axes}. {filmic_context}"
        "Use VFX/compositing vocabulary. Do not mention that this is an image or poster frame. "
        "End with one line starting TAGS: followed by compact JSON using keys "
        "subject, backing, shot_scale, frame_role, edge_contact, prop, action, "
        "effect_reference, camera_motion, keyability, motion_summary, confidence. "
        "Use short snake_case values, empty strings "
        "for unknown fields, and confidence from 0 to 1."
    )


def _motion_context_prompt(element: Element) -> str:
    if not element.motion_facets and not element.measured_features:
        return ""
    facet_parts = []
    for key, payload in sorted(element.motion_facets.items()):
        if isinstance(payload, dict) and "value" in payload:
            confidence = payload.get("confidence")
            if isinstance(confidence, (float, int)):
                facet_parts.append(f"{key}={payload['value']} ({float(confidence):.2f})")
            else:
                facet_parts.append(f"{key}={payload['value']}")
        else:
            facet_parts.append(f"{key}={payload}")
    feature_keys = (
        "sampled_pairs",
        "median_flow_magnitude",
        "global_translation_magnitude",
        "global_motion_ratio",
        "brightness_variance",
        "expansion_ratio",
        "start_end_visual_distance",
        "temporal_acceleration",
    )
    feature_parts = []
    for key in feature_keys:
        value = element.measured_features.get(key)
        if isinstance(value, float):
            feature_parts.append(f"{key}={value:.3f}")
        elif value is not None:
            feature_parts.append(f"{key}={value}")
    warning_text = ", ".join(element.motion_warnings) if element.motion_warnings else "none"
    return (
        "Measured optical-flow context follows; use it as guidance with its confidence, "
        "and prefer the ordered frames when the measurements look noisy. "
        f"Motion facets: {', '.join(facet_parts) or 'none'}. "
        f"Measured features: {', '.join(feature_parts) or 'none'}. "
        f"Warnings: {warning_text}. "
    )


_CAPTION_AXES = {
    "smoke": "density, colour, temperature, motion character, direction, speed, density evolution, lighting, and foreground/midground/background role",
    "fire": "flame type, colour temperature, scale, flicker or surge behaviour, wind influence, intensity arc, containment, and self-lighting",
    "explosion": "blast scale, onset, expanding fire/smoke/debris components, directionality, dissipation, lighting, and comp usefulness",
    "dust": "density, particle scale, kicked-up/settling/swirl character, direction, speed, lighting, and scale role",
    "debris": "material, scale, quantity, trajectory, gravity feel, speed, quantity arc, and lighting context",
    "water": "water type, scale, splash/spray/sheet structure, direction, gravity feel, loopability, transparency, and lighting",
    "atmosphere": "haze/fog/mist type, density, motion character, speed, lighting, depth role, and whether it reads as foreground, midground, or background",
    "lens_effect": "flare/leak/glint/dirt type, intensity, sweep/flicker/pulse behaviour, colour, backing, and comp use",
    "practical_light": "light type, colour, flicker/strobe/sweep behaviour, intensity change profile, spill, backing, and comp use",
    "plate": "visible subject, green-screen or blue-screen backing when present, keyability, held prop or weapon, smoke/fire/muzzle/backblast reference value, camera motion, parallax, performance/action, framing, lighting, and whether it is a useful foreground, midground, background, or insert plate",
    "impact": "impact type, onset, debris/smoke/liquid pattern, dissipation, direction, scale, lighting, and comp use",
    "sparks": "spark source, trajectory, quantity arc, density, speed, colour, backing, and comp use",
    "unknown": "visible content, material or density, backing, lighting, apparent scale, implied motion, and comp use",
}


_SIGLIP_CATEGORY_PROMPTS = {
    "smoke": "smoke plume, vapor, drifting grey or white volumetric smoke VFX element",
    "fire": "fire, flame, burning, ember, ignition, torch, flame VFX element",
    "explosion": "explosion, blast, fireball, expanding burst, pyrotechnic VFX element",
    "dust": "dust cloud, dirt puff, powder burst, fine particulate VFX element",
    "debris": "debris, fragments, rubble, broken pieces, falling material VFX element",
    "water": "water splash, spray, droplets, wave, liquid VFX element",
    "atmosphere": "haze, fog, mist, atmospheric depth, heat shimmer VFX element",
    "lens_effect": "lens flare, light leak, glint, bokeh, lens dirt, optical lens effect",
    "practical_light": "practical light, muzzle flash, lightning, neon, headlight, torch light effect",
    "plate": "live-action plate, green screen or blue screen subject, actor, person, vehicle, crowd, prop or weapon performance, environment footage for compositing",
    "impact": "impact hit, bullet hit, ground hit, blood spray, contact burst VFX element",
    "sparks": "sparks, welding, grinding, electrical spark shower, pyrotechnic sparks VFX element",
}


def _poster_path(element: Element) -> Path | None:
    if element.poster_path and Path(element.poster_path).exists():
        return Path(element.poster_path)
    return None


def _preserve_metadata_facets(element: Element) -> None:
    existing = dict(element.content_facets)
    derive_content_facets(element)
    # Preserve deterministic metadata/usability facets from ingest, but keep
    # model-derived category and category-specific facets from SigLIP routing.
    for key in ("resolution_tier", "frame_rate_tier", "alpha_status", "bit_depth_tier", "color_space"):
        if key in existing:
            element.content_facets[key] = existing[key]


if __name__ == "__main__":
    raise SystemExit(main())
