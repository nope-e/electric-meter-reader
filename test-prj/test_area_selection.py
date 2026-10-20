"""Regression checks for choosing a readout among post-NMS candidates."""
import copy
import unittest

from detect_meter_area import MeterAreaPipeline, select_area


class AreaSelectionTests(unittest.TestCase):
    def test_readout_beats_larger_background(self):
        # Real failure: LCD photo 1792403667742 had a large wiring/background box.
        background = {'xyxy': [0, 2.67, 3072, 1318.57], 'score': .627388}
        readout = {'xyxy': [86.01, 2451.28, 2229.13, 3120.11], 'score': .883682}
        candidates = [background, readout]
        original = copy.deepcopy(candidates)
        self.assertIs(select_area(candidates), readout)
        self.assertIs(select_area(candidates, 'largest_area'), background)
        self.assertEqual(candidates, original)

    def test_tighter_readout_beats_nameplate(self):
        nameplate = {'xyxy': [1004.97, 928.42, 2153.35, 1699.29], 'score': .519661}
        readout = {'xyxy': [1047.38, 949.09, 2149.24, 1244.62], 'score': .634157}
        self.assertIs(select_area([nameplate, readout]), readout)

    def test_equal_confidence_uses_area(self):
        small = {'xyxy': [0, 0, 10, 10], 'score': .8}
        large = {'xyxy': [0, 0, 20, 10], 'score': .8}
        self.assertIs(select_area([small, large]), large)

    def test_empty_stays_empty(self):
        self.assertIsNone(select_area([]))
        self.assertIsNone(select_area([], 'largest_area'))

    def test_invalid_rule_fails_before_loading_models(self):
        with self.assertRaises(ValueError):
            select_area([], 'unknown')
        with self.assertRaises(ValueError):
            MeterAreaPipeline(selection='unknown')


if __name__ == '__main__':
    unittest.main()
