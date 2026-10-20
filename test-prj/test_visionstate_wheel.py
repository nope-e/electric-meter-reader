"""Check carry handling and keep model preprocessing tied to the pinned upstream code."""
import ast
from pathlib import Path
import unittest

import numpy as np
from PIL import Image

from visionstate_wheel import decode_logits
from vendor.visionstate.common import cell_input, decode, decode_shifted


class VisionStateTests(unittest.TestCase):
    def test_red_and_black_digit_have_identical_darkest_channel_input(self):
        black = np.full((48,32,3),255,np.uint8)
        black[:,:16] = 0
        red = black.copy()
        red[:,:16] = [255,0,0]
        np.testing.assert_array_equal(cell_input(Image.fromarray(black)),cell_input(Image.fromarray(red)))

    def test_unfinished_last_wheel_is_floored_and_author_rounding_separate(self):
        logits = np.full((2,100),-100.)
        logits[0,30],logits[1,78] = 100,100
        result = decode_logits(logits)
        self.assertEqual(result['raw_digits'],'37')
        self.assertEqual(result['author_rounded_raw_digits'],'38')

    def test_joint_decoder_follows_carry(self):
        # Last wheel 9.7 advances the preceding wheel from 3 to 3.7.
        logp = np.full((2,100),-100.)
        logp[0,37],logp[1,97] = 0,0
        value,_ = decode(logp)
        self.assertAlmostEqual(value,39.7)

    def test_decode_rejects_nonfinite_logits(self):
        with self.assertRaises(ValueError):
            decode_logits(np.full((5,100),np.nan))

    def test_cached_runtime_decoder_matches_vendored_training_decoder(self):
        # The training common.py and deployed readers.py implement the same DP differently.
        # This checks the port against the actual pinned deployed function, when cached.
        path = Path(__file__).parent/'.tmp/visionstate/readers.py'
        if not path.exists():
            self.skipTest('Pinned runtime source not cached')
        tree = ast.parse(path.read_text(encoding='utf-8'))
        function = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='wheel_decode')
        namespace = {'np':np}
        exec(compile(ast.Module(body=[function],type_ignores=[]),str(path),'exec'),namespace)
        runtime = namespace['wheel_decode']
        rng = np.random.default_rng(20261020)
        for wheels in (1,2,5,6):
            for _ in range(20):
                logits = rng.normal(size=(wheels,100))
                value,score = decode(logits)
                other,other_score,_ = runtime(logits)
                self.assertAlmostEqual(value,other)
                self.assertAlmostEqual(score,other_score)


if __name__=='__main__':
    unittest.main()
