"""Regression cases for window geometry versus whole-meter type labels."""
from pathlib import Path
import tempfile
import unittest

from merge_window_data import ROOT, parse_labels, select_windows


def ann(name, box, line=1):
    return {'class':name, 'box':box, 'line':line, 'polygon':False}


class WindowMergeTests(unittest.TestCase):
    def test_type_inheritance_keeps_each_reading_geometry(self):
        lcd = ann('Reading', [.7,.4,.1,.05], 3)
        wheel = ann('Reading', [.2,.4,.1,.05], 4)
        labels = [ann('Digital', [.75,.5,.5,.9]),
                  ann('Analog', [.25,.5,.5,.9]), lcd, wheel,
                  ann('Serial No.', [.2,.7,.15,.05])]
        windows, reason = select_windows(labels, 'reading_parent', {})
        self.assertIsNone(reason)
        self.assertEqual([(a['class'], a['box']) for a in windows],
                         [('lcd',lcd['box']), ('wheel',wheel['box'])])

    def test_missing_or_conflicting_parent_cannot_silently_assign_type(self):
        reading = ann('Reading', [.5,.5,.2,.1])
        for parents in [[], [ann('Digital',[.5,.5,1.,1.]), ann('Analog',[.5,.5,1.,1.])]]:
            with self.subTest(parents=parents), self.assertRaises(ValueError):
                select_windows(parents+[reading], 'reading_parent', {})

    def test_reviewed_extra_whole_meter_reading_is_removed(self):
        strip = ann('Reading', [.5,.3,.3,.1])
        windows, _ = select_windows([strip,ann('Reading',[.5,.5,.9,.9])],
                                    'reading_parent', {'type':'lcd','drop_largest_reading':True})
        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0]['box'], strip['box'])

    def test_polygon_is_enclosed_by_box_and_missing_window_is_excluded(self):
        (ROOT/'.tmp').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT/'.tmp') as folder:
            label = Path(folder)/'polygon.txt'
            label.write_text('0 0.2 0.3 0.8 0.4 0.7 0.6 0.3 0.5\n')
            parsed = parse_labels(label, ['Reading'])
        for actual, expected in zip(parsed[0]['box'], [.5,.45,.6,.3]):
            self.assertAlmostEqual(actual, expected)
        self.assertTrue(parsed[0]['polygon'])
        selected, reason = select_windows([ann('frame',[.5,.5,.1,.1])], 'display', {})
        self.assertEqual(selected, [])
        self.assertIn('Missing positive', reason)


if __name__ == '__main__':
    unittest.main()
