#!/usr/bin/env python3
"""Freeze and validate only the dependency closure of the installed MLX runtime."""
from importlib import metadata
from pathlib import Path
import argparse

from packaging.requirements import Requirement


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    todo = [("mlx-vlm", "")]
    visited, versions = set(), {}
    while todo:
        name, extra = todo.pop()
        key = (name.lower().replace("_", "-"), extra)
        if key in visited:
            continue
        visited.add(key)
        distribution = metadata.distribution(name)
        versions[distribution.metadata["Name"]] = distribution.version
        for requirement in distribution.requires or []:
            parsed = Requirement(requirement)
            if parsed.marker and not parsed.marker.evaluate({"extra": extra}):
                continue
            installed = metadata.version(parsed.name)
            if parsed.specifier and installed not in parsed.specifier:
                raise ValueError(f"{parsed.name}=={installed} does not satisfy {parsed.specifier}")
            todo.extend((parsed.name, value) for value in parsed.extras or [""])
    header = "# Complete dependency closure: native macOS arm64 / CPython 3.12.\n# Version pins, not distribution hashes; audit and retest before updates.\n"
    args.output.write_text(header + "\n".join(f"{key}=={value}" for key, value in sorted(versions.items())) + "\n")
    print(f"Frozen {len(versions)} dependencies")


if __name__ == "__main__":
    main()
