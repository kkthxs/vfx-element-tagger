from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vfx_element_tagger.models import Element, SourceRepresentation
from vfx_element_tagger.search import search_elements


class SearchReliabilityTests(unittest.TestCase):
    def element(self, name="asset.mov"):
        return Element("id", [SourceRepresentation("rep", name, name, "mov")], "rep",
                       primary_family="plate", category="plate")

    def test_filename_cannot_validate_wrong_semantic_family(self):
        element = self.element("falling_snow.mov")
        element.semantic_analysis = {"summary": "A static starfield.", "primary_family": "environment"}
        self.assertEqual(search_elements([element], "falling snow"), [])

    def test_uncertainty_suggestions_and_negation_do_not_create_fire(self):
        element = self.element()
        element.semantic_analysis = {"summary": "A man holds a launcher. No fire is visible.",
                                     "uncertainty": ["Possibly fire and smoke"],
                                     "search_text": "fire smoke explosion",
                                     "usability": {"best_uses": ["fire smoke reference"]}}
        self.assertEqual(search_elements([element], "fire smoke"), [])

    def test_old_facets_cannot_rescue_missing_rotation(self):
        element = self.element()
        element.semantic_analysis = {"summary": "A man stands in front of a green screen."}
        element.motion_facets = {"character": {"value": "rotating", "source": "old", "confidence": .9}}
        self.assertEqual(search_elements([element], "rotating"), [])

    def test_old_caption_and_category_do_not_override_selected_analysis(self):
        element = self.element()
        element.category = "fire"
        element.caption = "A fire explosion."
        element.semantic_analysis = {"primary_family": "plate", "summary": "A man stands still."}
        self.assertEqual(search_elements([element], "fire explosion"), [])

    def test_affirmed_material_in_plate_is_searchable_without_filename(self):
        element = self.element()
        element.semantic_analysis = {"summary": "A man fires an RPG, producing orange fire and white smoke.",
                                     "primary_family": "plate"}
        hits = search_elements([element], "man RPG fire smoke")
        self.assertEqual(len(hits), 1)
        self.assertFalse(hits[0].explain["semantic_relevance"]["filename_is_visual_evidence"])
