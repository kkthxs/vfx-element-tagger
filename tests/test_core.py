from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vfx_element_tagger.category_resolver import resolve_initial_category
from vfx_element_tagger.discovery import discover_media
from vfx_element_tagger.models import Element, SourceRepresentation
from vfx_element_tagger.pipeline import _should_reuse_existing_element, rerun_stage
from vfx_element_tagger.proxy import build_elements_from_representations
from vfx_element_tagger.search import _concept_tokens, _query_tokens, _vector_score, search_elements
from vfx_element_tagger.stages import (
    caption_element,
    classify_element,
    derive_motion,
    embed_element,
    enrich_search_facets,
    facet_value,
    parse_qwen_tags,
    store_qwen_caption_tags,
)
from vfx_element_tagger.util import normalize_root, run_json


class CoreTests(unittest.TestCase):
    def test_subprocess_json_replaces_invalid_metadata_bytes(self) -> None:
        data, error = run_json(
            [
                sys.executable,
                "-c",
                "import sys; sys.stdout.buffer.write(b'{\"title\":\"\\xa9\"}')",
            ]
        )

        self.assertIsNone(error)
        self.assertEqual(data, {"title": "�"})

    def test_discovers_image_sequence_as_single_reference(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for frame in range(1001, 1004):
                (root / f"heavy_smoke.{frame}.png").write_bytes(b"fake")

            references = discover_media(root)

        self.assertEqual(len(references), 1)
        self.assertEqual(references[0].kind, "sequence")
        self.assertEqual(references[0].first_frame, 1001)
        self.assertEqual(references[0].last_frame, 1003)
        self.assertEqual(references[0].frame_padding, 4)

    def test_discovers_numeric_only_compound_suffix_sequence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "Ground Burst" / "v002"
            root.mkdir(parents=True)
            for frame in range(1, 5):
                (root / f"{frame:04d}.HR.exr").write_bytes(b"fake")

            references = discover_media(Path(tmp))

        self.assertEqual(len(references), 1)
        self.assertEqual(references[0].kind, "sequence")
        self.assertEqual(references[0].path.name, "%04d.HR.exr")
        self.assertEqual(references[0].format_hint, "exr_sequence")
        self.assertTrue(references[0].filename_root)

    def test_sequence_pattern_preserves_separator(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for frame in range(1, 4):
                (root / f"snow_pass-{frame:04d}.exr").write_bytes(b"fake")

            references = discover_media(root)

        self.assertEqual(references[0].path.name, "snow_pass-%04d.exr")

    def test_phash_alone_does_not_merge_unrelated_assets(self) -> None:
        first = _representation("vendor_a_smk.mov", phash="same")
        second = _representation("unrelated_preview.mp4", phash="same")

        elements = build_elements_from_representations([first, second])

        self.assertEqual(len(elements), 2)

    def test_near_phash_alone_does_not_merge_unrelated_assets(self) -> None:
        first = _representation("vendor_a_smk.mov", phash="phash_0000000000000000")
        second = _representation("unrelated_preview.mp4", phash="phash_0000000000000001")

        elements = build_elements_from_representations([first, second])

        self.assertEqual(len(elements), 2)

    def test_exact_fingerprint_links_proxy_chain(self) -> None:
        first = _representation("master.mov")
        second = _representation("preview.mp4")
        second.content_fingerprint = first.content_fingerprint

        elements = build_elements_from_representations([first, second])

        self.assertEqual(len(elements), 1)
        self.assertTrue(
            all(rep.linked_by_fingerprint for rep in elements[0].source_representations)
        )

    def test_numbered_asset_variants_keep_distinct_filename_roots(self) -> None:
        self.assertEqual(
            normalize_root("SteamJet-009.mov", preserve_identifiers=True),
            "steamjet_009",
        )
        self.assertEqual(
            normalize_root("SteamJet-016.mov", preserve_identifiers=True),
            "steamjet_016",
        )

    def test_uninformative_black_phash_does_not_merge_distinct_sequences(self) -> None:
        first = _representation("aerial_explosion.mov", phash="phash_0000000000000000")
        second = _representation("side_burst.mov", phash="phash_0000000000000000")

        elements = build_elements_from_representations([first, second])

        self.assertEqual(len(elements), 2)

    def test_search_respects_category_filter(self) -> None:
        smoke = build_elements_from_representations([_representation("heavy_smoke_slow.mov")])[0]
        fire = build_elements_from_representations([_representation("fire_fast.mov")])[0]
        for element in (smoke, fire):
            classify_element(element)
            derive_motion(element)
            caption_element(element)
            embed_element(element)

        hits = search_elements([smoke, fire], "heavy smoke", filters={"category": "smoke"})

        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].element.category, "smoke")

    def test_search_uses_structured_semantics_and_dotted_filters(self) -> None:
        explosion = build_elements_from_representations([_representation("asset_a.mov")])[0]
        smoke = build_elements_from_representations([_representation("asset_b.mov")])[0]
        explosion.primary_family = "explosion"
        explosion.semantic_analysis = {
            "summary": "A wide orange blast viewed from above.",
            "effect_type": "aerial_explosion",
            "composition": {"viewpoint": "top_down"},
            "motion": {"speed": "fast"},
        }
        smoke.primary_family = "smoke"
        smoke.semantic_analysis = {
            "summary": "A slow grey smoke plume.",
            "effect_type": "rising_smoke",
            "composition": {"viewpoint": "front"},
            "motion": {"speed": "slow"},
        }

        hits = search_elements(
            [smoke, explosion],
            "wide top down orange explosion",
            filters={"semantic.motion.speed": "fast"},
        )

        self.assertEqual([hit.element.element_id for hit in hits], [explosion.element_id])
        self.assertIn("top", hits[0].explain["lexical"]["buckets"]["semantic"])

    def test_search_explain_reports_lexical_match_buckets(self) -> None:
        element = build_elements_from_representations([_representation("heavy_smoke_slow.mov")])[0]
        element.category = "smoke"
        element.caption = "Heavy grey smoke rises across frame."
        element.content_facets = {"density": {"value": "heavy", "source": "test", "confidence": 1.0}}
        element.motion_facets = {
            "speed": {"value": "slow", "source": "test", "confidence": 1.0},
            "character": {"value": "rising", "source": "test", "confidence": 1.0},
        }

        hits = search_elements([element], "heavy smoke slow rising")

        explain = hits[0].explain
        lexical = explain["lexical"]
        self.assertEqual(explain["rank"], 1)
        self.assertEqual(explain["signals"]["lexical"]["rank"], 1)
        self.assertEqual(lexical["score"], 1.0)
        self.assertEqual(lexical["matched_terms"], ["heavy", "rising", "slow", "smoke"])
        self.assertEqual(lexical["buckets"]["category"], ["smoke"])
        self.assertEqual(lexical["buckets"]["caption"], ["heavy", "smoke"])
        self.assertEqual(lexical["buckets"]["filename"], ["heavy", "slow", "smoke"])
        self.assertEqual(lexical["buckets"]["facets"], ["heavy", "rising", "slow"])
        self.assertEqual(lexical["facet_matches"]["content.density"], ["heavy"])
        self.assertEqual(lexical["facet_matches"]["motion.speed"], ["slow"])
        self.assertEqual(lexical["facet_matches"]["motion.character"], ["rising"])

    def test_search_omits_elements_without_a_meaningful_match(self) -> None:
        explosion = build_elements_from_representations([_representation("explosion.mov")])[0]
        explosion.primary_family = "explosion"
        explosion.semantic_analysis = {
            "summary": "A bright orange blast expands against black.",
            "effect_type": "aerial_explosion",
        }
        smoke = build_elements_from_representations([_representation("smoke.mov")])[0]
        smoke.primary_family = "smoke"
        smoke.semantic_analysis = {
            "summary": "A grey smoke plume rises slowly.",
            "effect_type": "rising_smoke",
        }

        hits = search_elements([explosion, smoke], "blood splatter")

        self.assertEqual(hits, [])

    def test_search_understands_artist_synonyms_and_structured_phrases(self) -> None:
        plate = build_elements_from_representations([_representation("plate.mov")])[0]
        plate.primary_family = "plate"
        plate.semantic_analysis = {
            "summary": "A male performer walks and gestures in a studio.",
            "effect_type": "greenscreen_performance_plate",
            "appearance": {"backing": "green_screen"},
            "motion": {"character": ["walking", "gesturing"]},
        }
        explosion = build_elements_from_representations([_representation("blast.mov")])[0]
        explosion.primary_family = "explosion"
        explosion.semantic_analysis = {
            "summary": "A wide explosion viewed from above.",
            "effect_type": "aerial_explosion",
            "composition": {"viewpoint": "top_down", "shot_scale": "wide"},
        }

        plate_hits = search_elements([explosion, plate], "actor walking on chroma key")
        explosion_hits = search_elements([plate, explosion], "wide overhead blast")

        self.assertEqual([hit.element.element_id for hit in plate_hits], [plate.element_id])
        self.assertEqual(
            [hit.element.element_id for hit in explosion_hits],
            [explosion.element_id],
        )
        self.assertIn(
            "greenscreen",
            plate_hits[0].explain["semantic_relevance"]["matched_concepts"],
        )

    def test_search_respects_modifiers_in_two_concept_queries(self) -> None:
        ground_fire = build_elements_from_representations([_representation("ground_fire.mov")])[0]
        ground_fire.primary_family = "fire"
        ground_fire.semantic_analysis = {
            "summary": "A low ground-level flame burns in place.",
            "effect_type": "ground_fire",
        }
        fireball = build_elements_from_representations([_representation("fireball.mov")])[0]
        fireball.primary_family = "fire"
        fireball.semantic_analysis = {
            "summary": "A fireball flies through the centre.",
            "effect_type": "projectile_fireball",
        }

        hits = search_elements([fireball, ground_fire], "ground fire")

        self.assertEqual([hit.element.element_id for hit in hits], [ground_fire.element_id])

    def test_vector_score_requires_matching_dimensions(self) -> None:
        self.assertEqual(_vector_score([1.0, 0.0], [1.0]), 0.0)
        self.assertEqual(_vector_score([1.0, 0.0], [1.0, 0.0]), 1.0)

    def test_query_tokens_drop_search_stopwords(self) -> None:
        self.assertEqual(_query_tokens("wide green screen element of a man"), {"wide", "green", "screen", "man"})

    def test_search_normalizes_explosion_word_forms(self) -> None:
        self.assertEqual(
            _concept_tokens(
                "explosion explosions explode explodes exploded exploding detonation detonating"
            ),
            {"explosion"},
        )

    def test_explosion_query_includes_exploding_integrated_plate(self) -> None:
        aerial = build_elements_from_representations([_representation("aerial.mov")])[0]
        aerial.primary_family = "explosion"
        aerial.semantic_analysis = {
            "summary": "A large fireball expands into a smoke cloud.",
            "effect_type": "aerial_explosion",
        }
        plate = build_elements_from_representations([_representation("helicopter.mov")])[0]
        plate.primary_family = "plate"
        plate.semantic_analysis = {
            "summary": "A helicopter flies right and then explodes into burning debris.",
            "effect_type": "action_plate",
        }

        expected = {aerial.element_id, plate.element_id}
        for query in ("explosion", "explode", "explodes", "exploded", "exploding", "detonation"):
            with self.subTest(query=query):
                self.assertEqual(
                    {hit.element.element_id for hit in search_elements([aerial, plate], query)},
                    expected,
                )

    def test_family_gate_ignores_broad_generated_search_suggestions(self) -> None:
        smoke = build_elements_from_representations([_representation("smoke.mov")])[0]
        smoke.primary_family = "smoke"
        smoke.semantic_analysis = {
            "summary": "A dark smoke plume rises into a mushroom-shaped cloud.",
            "search_text": "smoke plume, smoke burst, smoke explosion, aftermath smoke",
            "effect_type": "rising_smoke_plume",
        }

        self.assertEqual(search_elements([smoke], "explosion"), [])
        self.assertEqual(search_elements([smoke], "exploding"), [])

    def test_model_backed_rerun_does_not_downgrade_to_fallback(self) -> None:
        element = build_elements_from_representations([_representation("heavy_smoke_slow.mov")])[0]
        element.captioning_model_version = "Qwen3-VL-8B-Instruct poster-frame local"
        element.embedding_model_version = "siglip2-large-patch16-384 + qwen3-embedding-8b"
        element.image_embed = [0.0] * 1024

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "deterministic fallback"):
                rerun_stage(element, "caption", Path(tmp))

    def test_incremental_ingest_reuses_unchanged_element(self) -> None:
        existing = build_elements_from_representations([_representation("heavy_smoke_slow.mov")])[0]
        candidate = build_elements_from_representations([_representation("heavy_smoke_slow.mov")])[0]

        self.assertTrue(_should_reuse_existing_element(existing, candidate))

        candidate.source_representations[0].content_fingerprint = "changed"

        self.assertFalse(_should_reuse_existing_element(existing, candidate))

    def test_low_confidence_key_screen_routes_to_plate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "greenscreen.jpg"
            pixels = np.zeros((90, 160, 3), dtype=np.uint8)
            pixels[:, :] = [20, 190, 30]
            Image.fromarray(pixels).save(path)

            category, confidence, provenance = resolve_initial_category(
                "lens_effect",
                0.43,
                path,
            )

        self.assertEqual(category, "plate")
        self.assertGreaterEqual(confidence, 0.6)
        self.assertEqual(provenance["siglip_top_category"], "lens_effect")

    def test_caption_enrichment_adds_vfx_search_facets(self) -> None:
        element = build_elements_from_representations([_representation("greenscreen_man.mov")])[0]
        element.category = "plate"
        element.content_facets = {"subject": {"value": "unknown", "source": "test", "confidence": 0.1}}
        element.caption = (
            "A male performer turns in a medium-wide green-screen studio. "
            "The uniform green backdrop is fully keyable with no edge contact, "
            "and the performer reads as a foreground element."
        )

        enrich_search_facets(element)

        self.assertEqual(facet_value(element.content_facets["subject"]), "person")
        self.assertEqual(facet_value(element.content_facets["backing"]), "green_screen")
        self.assertEqual(facet_value(element.content_facets["shot_scale"]), "medium_wide")
        self.assertIn("man", facet_value(element.content_facets["search_terms"]))
        self.assertIn("rotating", facet_value(element.content_facets["search_terms"]))

    def test_qwen_tag_parser_prefers_structured_payload(self) -> None:
        raw = (
            "A male performer turns in a medium-wide green-screen studio.\n"
            'TAGS: {"subject":"male performer","backing":"green screen",'
            '"shot_scale":"medium wide","frame_role":"foreground",'
            '"edge_contact":"no edge contact","action":["turning","gesturing"],'
            '"camera_motion":"locked off","keyability":"clean key",'
            '"motion_summary":"turning gesture with static camera","confidence":0.86}'
        )

        caption, tags = parse_qwen_tags(raw)

        self.assertEqual(caption, "A male performer turns in a medium-wide green-screen studio.")
        self.assertEqual(tags["subject"], "person")
        self.assertEqual(tags["backing"], "green_screen")
        self.assertEqual(tags["shot_scale"], "medium_wide")
        self.assertEqual(tags["frame_role"], "foreground")
        self.assertEqual(tags["edge_contact"], "clean_margin")
        self.assertEqual(tags["action"], "turning gesturing")
        self.assertEqual(tags["camera_motion"], "locked_off")
        self.assertEqual(tags["keyability"], "keyable")
        self.assertEqual(tags["confidence"], 0.86)

    def test_qwen_tag_storage_uses_deterministic_fallback_and_round_trips(self) -> None:
        element = build_elements_from_representations([_representation("greenscreen_man.mov")])[0]
        element.category = "plate"
        element.motion_facets = {
            "camera_motion": {"value": "pan", "source": "test", "confidence": 0.6},
            "speed": {"value": "slow", "source": "test", "confidence": 0.6},
        }
        element.content_facets = {"subject": {"value": "unknown", "source": "test", "confidence": 0.1}}

        store_qwen_caption_tags(
            element,
            "A male performer walks across a medium wide green-screen backing. "
            "The subject is foreground, fully contained with no edge contact, and keyable.",
        )

        self.assertEqual(element.qwen_tags["subject"], "person")
        self.assertEqual(element.qwen_tags["backing"], "green_screen")
        self.assertEqual(element.qwen_tags["shot_scale"], "medium_wide")
        self.assertEqual(element.qwen_tags["frame_role"], "foreground")
        self.assertEqual(element.qwen_tags["edge_contact"], "clean_margin")
        self.assertEqual(element.qwen_tags["action"], "walking")
        self.assertEqual(element.qwen_tags["camera_motion"], "pan")
        self.assertIn("slow", element.qwen_tags["motion_summary"])
        self.assertEqual(facet_value(element.content_facets["backing"]), "green_screen")
        self.assertEqual(facet_value(element.content_facets["keyability"]), "keyable")

        round_tripped = Element.from_dict(element.to_dict())

        self.assertEqual(round_tripped.qwen_tags, element.qwen_tags)

    def test_v02_analysis_fields_round_trip(self) -> None:
        element = build_elements_from_representations([_representation("lightning.mov")])[0]
        element.primary_family = "electricity"
        element.secondary_families = ["light"]
        element.semantic_analysis = {"effect_type": "branching_lightning_strike"}
        element.analysis_status = "escalated"
        element.analysis_confidence = 0.88
        element.source_representations[0].technical_metadata = {
            "channel_names": ["R", "G", "B", "A"]
        }

        restored = Element.from_dict(element.to_dict())

        self.assertEqual(restored.primary_family, "electricity")
        self.assertEqual(restored.secondary_families, ["light"])
        self.assertEqual(restored.analysis_status, "escalated")
        self.assertEqual(
            restored.source_representations[0].technical_metadata["channel_names"],
            ["R", "G", "B", "A"],
        )

    def test_weapon_plate_enrichment_preserves_smoke_fire_reference_terms(self) -> None:
        element = build_elements_from_representations([_representation("blue_screen_rpg_operator.mov")])[0]
        element.category = "plate"
        element.content_facets = {"subject": {"value": "unknown", "source": "test", "confidence": 0.1}}

        store_qwen_caption_tags(
            element,
            "A wide blue screen plate of an operator firing an RPG rocket launcher. "
            "The shot is useful as smoke, fire, muzzle flash, and backblast reference.",
        )

        self.assertEqual(element.qwen_tags["subject"], "person")
        self.assertEqual(element.qwen_tags["backing"], "blue_screen")
        self.assertEqual(element.qwen_tags["prop"], "rocket_launcher")
        self.assertIn("firing_weapon", element.qwen_tags["action"])
        self.assertIn("smoke", element.qwen_tags["effect_reference"])
        self.assertIn("fire", element.qwen_tags["effect_reference"])
        self.assertIn("backblast", element.qwen_tags["effect_reference"])
        self.assertEqual(facet_value(element.content_facets["prop"]), "rocket_launcher")
        self.assertIn("rpg", facet_value(element.content_facets["search_terms"]))
        self.assertIn("muzzle_flash", facet_value(element.content_facets["search_terms"]))


def _representation(name: str, phash: str | None = None) -> SourceRepresentation:
    return SourceRepresentation(
        representation_id=f"rep_{name}",
        path=f"/tmp/{name}",
        original_path=f"/tmp/{name}",
        format="mov",
        codec="h264",
        width=1920,
        height=1080,
        fps=24.0,
        duration_seconds=2.0,
        bit_depth=8,
        file_size_bytes=1024,
        filename_root=Path(name).stem,
        content_fingerprint=f"fingerprint_{name}",
        perceptual_hash=phash,
    )


if __name__ == "__main__":
    unittest.main()
