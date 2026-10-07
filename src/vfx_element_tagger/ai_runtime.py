from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from .settings import models_dir


DEFAULT_MANIFEST = models_dir() / "manifest.json"


def load_model_manifest(path: Path = DEFAULT_MANIFEST) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def model_path(key: str, manifest_path: Path = DEFAULT_MANIFEST) -> Path | None:
    manifest = load_model_manifest(manifest_path)
    value = manifest.get("models", {}).get(key, {}).get("path")
    if not value:
        return None
    path = Path(value)
    return path if path.exists() else None


@lru_cache(maxsize=1)
def qwen_embedding_model():
    from sentence_transformers import SentenceTransformer

    path = model_path("qwen3_embedding")
    if not path:
        raise RuntimeError("Qwen3 embedding model is not downloaded.")
    return SentenceTransformer(str(path), device="cpu")


@lru_cache(maxsize=1)
def siglip_text_stack():
    import torch
    from transformers import AutoModel, AutoProcessor

    path = model_path("siglip2")
    if not path:
        raise RuntimeError("SigLIP 2 model is not downloaded.")
    processor = AutoProcessor.from_pretrained(path, local_files_only=True)
    model = AutoModel.from_pretrained(path, local_files_only=True, dtype=torch.float32).eval()
    return processor, model


def qwen_query_embedding(text: str) -> list[float]:
    model = qwen_embedding_model()
    vector = model.encode([text], normalize_embeddings=True, show_progress_bar=False)[0]
    return vector.astype(float).tolist()


def siglip_query_embedding(text: str) -> list[float]:
    import torch

    processor, model = siglip_text_stack()
    inputs = processor(text=[text], padding="max_length", return_tensors="pt")
    with torch.no_grad():
        if hasattr(model, "get_text_features"):
            features = model.get_text_features(**inputs)
        else:
            outputs = model(**inputs)
            features = outputs.text_embeds
    features = torch.nn.functional.normalize(features.float(), dim=-1)
    return features[0].cpu().tolist()
