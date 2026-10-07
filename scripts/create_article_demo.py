#!/usr/bin/env python3
"""Generate original procedural demo footage and explicitly curated UI fixtures.

No production media, catalog, model predictions or personal paths are copied.
This is a UI demonstration, not an analysis accuracy benchmark.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vfx_element_tagger.models import Element, SourceRepresentation
from vfx_element_tagger.store import LibraryStore

W, H, FPS, FRAMES = 640, 360, 24, 96
ITEMS = [
    ("smoke", "billowing_smoke_plume", "Silver Smoke Plume", "rising", (176, 191, 210)),
    ("fire", "rolling_flame", "Amber Flame Roll", "rising", (255, 104, 22)),
    ("electricity", "branching_lightning_strike", "Branching Lightning", "downward", (115, 165, 255)),
    ("spark", "spark_burst", "Radial Spark Burst", "outward", (255, 181, 72)),
    ("smoke", "ground_smoke", "Low Ground Smoke", "left_to_right", (133, 162, 179)),
    ("energy", "energy_ring", "Cyan Energy Ring", "expanding", (44, 233, 227)),
    ("dust", "ground_dust_hit", "Ground Dust Impact", "outward", (198, 166, 112)),
    ("snow", "falling_snow", "Drifting Snow Field", "downward", (224, 237, 255)),
    ("fire", "flame_column", "Vertical Flame Column", "rising", (255, 140, 40)),
    ("smoke", "wispy_smoke", "Wispy Smoke Curl", "rising", (145, 159, 200)),
    ("particle", "floating_particles", "Floating Particle Field", "upward", (175, 226, 208)),
    ("electricity", "electric_arc", "Horizontal Electric Arc", "left_to_right", (183, 142, 255)),
]


def frame(kind, index, t, colour):
    rng = np.random.default_rng(300 + index)
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    x, y = x / W, y / H
    if kind in {"smoke", "fire", "dust"}:
        field = np.zeros((H, W), dtype=np.float32)
        for n in range(24):
            phase = (n / 24 + t * .16) % 1
            cx = .5 + .15 * np.sin(n * 2.3 + phase * 4) * phase
            cy = .86 - phase * .65
            sx, sy = .025 + phase * .10, .025 + phase * .10
            if index == 4:
                cx, cy, sx, sy = .12 + phase * .75, .74 + .045 * np.sin(n), .08, .055
            if kind == "dust":
                cx, cy, sx, sy = .5 + np.sin(n) * .32 * phase, .8 - phase * .27, .07, .045
            field += np.exp(-((x-cx)**2 / sx**2 + (y-cy)**2 / sy**2)) * (.8 - phase*.5)
        turbulence = (np.sin(59*x + 13*np.sin(19*y+t)) * np.sin(47*y-t*3)
                      + np.sin(123*x+57*y+t*4) * .3)
        field = np.clip(field * (.58 + turbulence*.17), 0, 1)
        if kind == "fire":
            field = field ** 1.7
        rgb = np.clip(field[..., None] * np.array(colour), 0, 255).astype(np.uint8)
        if kind == "fire":
            rgb[..., 1] = np.clip(field ** 2 * 218, 0, 255)
            rgb[..., 2] = np.clip(field ** 4 * 117, 0, 255)
        image = Image.fromarray(rgb)
    else:
        image = Image.new("RGB", (W, H))
        draw = ImageDraw.Draw(image)
        if kind == "electricity":
            def bolt(points, width):
                draw.line(points, fill=colour, width=width)
                draw.line(points, fill=(230, 241, 255), width=1)
            points = [(W*.5 + rng.normal(0, 14), j) for j in range(18, H-20, 16)]
            if index == 11:
                points = [(j, H*.5 + rng.normal(0, 15)) for j in range(28, W-20, 20)]
            bolt(points, 4)
            for n in (5, 9, 13):
                a, b = points[n]
                branch = [(a + k*8, b + k*4 + rng.normal(0, 10)) for k in range(9)]
                bolt(branch, 2)
        elif kind == "energy":
            radius = 90 + 13*np.sin(t*3)
            for j in range(600):
                angle = j / 600 * np.pi * 2
                r = radius + 6*np.sin(angle*13+t*5)
                a, b = W*.5+r*np.cos(angle), H*.5+r*.7*np.sin(angle)
                draw.ellipse((a-2, b-2, a+2, b+2), fill=colour)
        else:
            for n in range(140):
                if kind == "spark":
                    angle, speed = rng.uniform(0, 2*np.pi), rng.uniform(25, 150)
                    phase = .5 + .18*np.sin(t*2)
                    a, b = W*.5+np.cos(angle)*speed*phase*2, H*.55+np.sin(angle)*speed*phase
                    draw.line((a, b, a+np.cos(angle)*9, b+np.sin(angle)*7), fill=colour, width=2)
                else:
                    a, b = rng.uniform(0, W), (rng.uniform(0, H)+t*(24 if kind=="snow" else -15))%H
                    radius = rng.uniform(.7, 2.1)
                    draw.ellipse((a-radius, b-radius, a+radius, b+radius), fill=colour)
        glow = image.filter(ImageFilter.GaussianBlur(5))
        image = Image.fromarray(np.clip(np.asarray(image, dtype=np.float32) +
                                        np.asarray(glow, dtype=np.float32)*1.8, 0, 255).astype(np.uint8))
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    catalog = root / "library.sqlite3"
    if catalog.exists():
        parser.error("Demo catalog already exists; choose a new output directory")
    elements = []
    for index, (family, subtype, name, direction, colour) in enumerate(ITEMS):
        slug = name.lower().replace(" ", "-")
        directory = root / "artifacts" / slug
        directory.mkdir(parents=True, exist_ok=True)
        movie, poster = directory / "preview.mp4", directory / "poster.jpg"
        primary = directory / "source.mov" if index == 0 else movie
        command = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                   "-s", f"{W}x{H}", "-r", str(FPS), "-i", "pipe:0", "-an", "-c:v", "libx264",
                   "-pix_fmt", "yuv420p", "-crf", "17", "-movflags", "+faststart", str(movie)]
        process = subprocess.Popen(command, stdin=subprocess.PIPE)
        for number in range(FRAMES):
            image = frame(family, index, number/FPS, colour)
            if number == 36:
                image.save(poster, quality=95)
            process.stdin.write(image.tobytes())
        process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError("Demo preview encoding failed")
        if index == 0:
            rgba = np.asarray(frame(family, index, 1.5, colour))
            alpha = np.max(rgba, axis=2).astype(np.uint8)
            source_image = directory / "alpha-source.png"
            Image.fromarray(np.dstack((rgba, alpha))).save(source_image)
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-loop", "1", "-i", str(source_image),
                            "-t", "4", "-r", str(FPS), "-c:v", "qtrle", str(primary)], check=True)
        rep = SourceRepresentation(slug, str(primary), f"/demo/elements/{slug}.mov", "mov",
                                   width=W, height=H, fps=FPS, duration_seconds=4, bit_depth=8)
        analysis = {
            "primary_family": family, "secondary_families": [], "effect_type": subtype,
            "summary": f"A {name.lower()} with {direction.replace('_', ' ')} motion, isolated against black.",
            "search_text": f"{name} {family} {subtype} wide isolated black {direction} compositing element",
            "composition": {"viewpoint": "front", "shot_scale": "wide", "origin": "lower_centre",
                            "direction": direction, "edge_contact": ["none"], "frame_coverage": "medium",
                            "spatial_distribution": "central", "instance_count": "single"},
            "motion": {"speed": "slow" if family=="smoke" else "fast", "direction": direction,
                       "character": ["billowing", "turbulent"] if family=="smoke" else ["flowing"],
                       "temporal_arc": "sustained", "depth_motion": "none", "event_count": "single"},
            "appearance": {"colour": ["silver"] if family=="smoke" else ["warm" if family=="fire" else "cool"],
                           "backing": "transparent" if index==0 else "black", "density": "medium",
                           "texture": ["billowing"] if family=="smoke" else ["fine"]},
            "usability": {"recommended_use": ["compositing", "atmosphere" if family=="smoke" else "accent"],
                          "cleanup_notes": "Synthetic demonstration footage; not a production plate."},
            "action_timing": {"available": True, "visible_start_seconds": 0, "visible_end_seconds": 4,
                              "optimal_start_seconds": .5, "optimal_end_seconds": 3.5, "peak_seconds": 1.5,
                              "source_fps": FPS, "pattern": "sustained", "review_required": True},
            "uncertainty": [],
        }
        item = Element(slug, [rep], slug, duration_seconds=4, width=W, height=H, fps=FPS,
                       bit_depth=8, aspect_ratio=W/H, has_alpha_channel=index==0, alpha_non_empty=index==0,
                       alpha_is_soft=index==0, over_black_likely=index!=0, poster_path=str(poster),
                       preview_movie_480_path=str(movie), preview_movie_1080_path=str(movie),
                       category=family, category_confidence=.91, primary_family=family,
                       caption=analysis["summary"], semantic_analysis=analysis, analysis_confidence=.91,
                       analysis_status="needs_review" if index in (2, 6) else "accepted",
                       analysis_model_version="synthetic_demo_fixture", analysis_schema_version="vfx-semantic-0.5",
                       provenance={"semantic_analysis": {"source": "synthetic_demo_fixture"}})
        if item.analysis_status == "needs_review":
            alternate = deepcopy(analysis)
            alternate["composition"]["edge_contact"] = ["top"]
            item.analysis_candidates = [
                {"model": "Demo primary proposal", "role": "primary", "confidence": .91, "result": analysis},
                {"model": "Demo challenger proposal", "role": "challenger", "confidence": .8, "result": alternate},
            ]
            item.analysis_escalation = {"candidate_disagreement": True, "human_review_required": True,
                                        "final_assessment": {"issues": ["composition.edge_contact uncertain"]}}
        elements.append(item)
        print(f"Created {name}", flush=True)
    store = LibraryStore(catalog)
    store.save(elements)
    for index, item in enumerate(elements):
        if index % 3 == 0:
            store.rate_element(item.element_id, "demo-artist", 5 if index == 0 else 4, "Demo Artist")
    (root / "DEMO.json").write_text(json.dumps({"media": "Original procedural synthetic assets",
        "labels": "Curated fixtures, not AI predictions", "elements": len(elements)}, indent=2))
    print(f"Serve with: python scripts/serve_library.py --demo --port 8766 --library {catalog} --cache-dir {root}")


if __name__ == "__main__":
    main()
