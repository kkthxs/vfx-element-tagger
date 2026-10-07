from __future__ import annotations

import importlib.resources
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Taxonomy:
    raw_yaml: str
    categories: list[str]
    category_schemas: dict[str, dict[str, dict[str, list[str]]]] = field(default_factory=dict)
    usability: dict[str, list[str]] = field(default_factory=dict)

    def motion_facets_for(self, category: str) -> dict[str, list[str]]:
        return self.category_schemas.get(category, {}).get("motion", {})

    def content_facets_for(self, category: str) -> dict[str, list[str]]:
        return self.category_schemas.get(category, {}).get("content", {})

    def context_facets_for(self, category: str) -> dict[str, list[str]]:
        return self.category_schemas.get(category, {}).get("context", {})


def load_taxonomy() -> Taxonomy:
    raw = (
        importlib.resources.files("vfx_element_tagger")
        .joinpath("data/taxonomy.yaml")
        .read_text(encoding="utf-8")
    )
    parsed = _parse_yaml(raw)
    categories = [str(item) for item in parsed.get("category", []) if isinstance(item, str)]
    usability = _coerce_facet_map(parsed.get("usability", {}))
    schemas: dict[str, dict[str, dict[str, list[str]]]] = {}
    for name in categories:
        block = parsed.get(name)
        if not isinstance(block, dict):
            continue
        schemas[name] = {
            section: _coerce_facet_map(block.get(section, {}))
            for section in ("content", "motion", "context")
            if isinstance(block.get(section), dict)
        }
    return Taxonomy(raw_yaml=raw, categories=categories, category_schemas=schemas, usability=usability)


def _parse_yaml(raw: str) -> dict[str, object]:
    """Tiny YAML reader for the shape used in data/taxonomy.yaml.

    Handles two-space-indented maps, dash-prefixed list items nested
    under a key, and inline `[a, b, c]` flow lists. Deliberately does
    not handle anchors, multi-line strings, or anything fancier.
    """
    root: dict[str, object] = {}
    # Each frame: (indent, container, parent_dict_or_None, parent_key_or_None)
    stack: list[tuple[int, object, object | None, str | None]] = [(-1, root, None, None)]

    for raw_line in raw.splitlines():
        line = raw_line.rstrip()
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        body = line.strip()

        while len(stack) > 1 and indent <= stack[-1][0]:
            stack.pop()
        parent_indent, container, parent_dict, parent_key = stack[-1]

        if body.startswith("- "):
            value = _coerce_scalar(body[2:].strip())
            if not isinstance(container, list):
                if parent_dict is None or parent_key is None:
                    continue
                new_list: list[object] = []
                parent_dict[parent_key] = new_list
                stack.pop()
                stack.append((parent_indent, new_list, parent_dict, parent_key))
                container = new_list
            container.append(value)
            continue

        if ":" not in body or not isinstance(container, dict):
            continue
        key, _, rest = body.partition(":")
        key = key.strip()
        rest = rest.strip()
        if rest == "":
            child: dict[str, object] = {}
            container[key] = child
            stack.append((indent, child, container, key))
        elif rest.startswith("[") and rest.endswith("]"):
            container[key] = [
                _coerce_scalar(item.strip())
                for item in rest[1:-1].split(",")
                if item.strip()
            ]
        else:
            container[key] = _coerce_scalar(rest)
    return root


def _coerce_scalar(token: str) -> object:
    token = token.strip()
    if len(token) >= 2 and token[0] in ('"', "'") and token[-1] == token[0]:
        return token[1:-1]
    if token.isdigit():
        return int(token)
    return token


def _coerce_facet_map(block: object) -> dict[str, list[str]]:
    if not isinstance(block, dict):
        return {}
    result: dict[str, list[str]] = {}
    for key, value in block.items():
        if isinstance(value, list):
            result[str(key)] = [str(item) for item in value]
    return result


CATEGORY_KEYWORDS: dict[str, set[str]] = {
    "smoke": {"smoke", "smk", "smokey", "billow", "plume", "vapor", "vapour"},
    "fire": {"fire", "flame", "burn", "ember", "torch", "ignition"},
    "explosion": {"explosion", "explode", "blast", "detonation", "pyro", "boom"},
    "dust": {"dust", "dusty", "sand", "powder", "dirt"},
    "debris": {"debris", "rubble", "rock", "glass", "shatter", "wood", "metal"},
    "water": {"water", "splash", "spray", "wave", "droplet", "mist", "rain"},
    "atmosphere": {"atmos", "atmosphere", "haze", "fog", "mist", "cloud"},
    "lens_effect": {"flare", "lens", "leak", "glint", "bokeh", "scratch", "dirt"},
    "practical_light": {"muzzle", "lightning", "neon", "headlight", "strobe"},
    "plate": {"plate", "sky", "street", "crowd", "vehicle", "interior", "landscape"},
    "impact": {"impact", "hit", "bullet", "blood", "groundhit"},
    "sparks": {"spark", "sparks", "welding", "grinding", "ricochet"},
}
