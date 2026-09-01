from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "run_apimart_minimax_h3_sequential.py"
sys.path.insert(0, str(SCRIPT_PATH.parent))
SPEC = importlib.util.spec_from_file_location("run_apimart_minimax_h3_sequential_geometry", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


class H3CanvasGeometryTests(unittest.TestCase):
    def test_near_square_source_is_letterboxed_without_stretching(self) -> None:
        geometry = module.compute_canvas_geometry(1016, 1018)
        self.assertEqual(geometry.content_width, 766)
        self.assertEqual(geometry.content_height, 768)
        self.assertEqual(geometry.offset_x, 289)
        self.assertEqual(geometry.offset_y, 0)
        self.assertEqual(geometry.as_dict()["mode"], "letterbox_then_crop_v1")

    def test_wide_source_gets_top_and_bottom_padding(self) -> None:
        geometry = module.compute_canvas_geometry(1920, 1080)
        self.assertEqual((geometry.content_width, geometry.content_height), (1344, 756))
        self.assertEqual((geometry.offset_x, geometry.offset_y), (0, 6))

    def test_native_h3_geometry_has_no_padding(self) -> None:
        geometry = module.compute_canvas_geometry(1344, 768)
        self.assertEqual((geometry.content_width, geometry.content_height), (1344, 768))
        self.assertEqual((geometry.offset_x, geometry.offset_y), (0, 0))


if __name__ == "__main__":
    unittest.main()
