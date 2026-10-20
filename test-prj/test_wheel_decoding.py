"""Check carry-before-completion conventions and the 9 -> 0 wrap."""
import unittest
import numpy as np
from recognize_wheel import decode_class100, decode_cont


class WheelDecodingTests(unittest.TestCase):
    def test_class100_transition_is_preceding_digit(self):
        output = np.zeros(100,np.float32)
        output[37] = 1
        result = decode_class100(output)
        self.assertEqual(result['digit'],3)
        self.assertEqual(result['state'],3.7)

    def test_class100_wrap_keeps_nine_until_zero(self):
        for state,expected in ((99,9),(0,0)):
            output = np.eye(100,dtype=np.float32)[state]
            self.assertEqual(decode_class100(output)['digit'],expected)

    def test_cont_peak_at_next_digit_still_retains_previous(self):
        output = np.zeros(10,np.float32)
        output[3],output[4] = .2,.8
        result = decode_cont(output)
        self.assertAlmostEqual(result['state'],3.8)
        self.assertEqual(result['digit'],3)

    def test_cont_wrap_is_nine_then_zero(self):
        output = np.zeros(10,np.float32)
        output[9],output[0] = .2,.8
        result = decode_cont(output)
        self.assertAlmostEqual(result['state'],9.8)
        self.assertEqual(result['digit'],9)
        output[9],output[0] = 0,1
        self.assertEqual(decode_cont(output)['digit'],0)

    def test_cont_invalid_zero_fit_has_no_digit(self):
        result = decode_cont(np.zeros(10,np.float32))
        self.assertIsNone(result['state'])
        self.assertIsNone(result['digit'])


if __name__=='__main__':
    unittest.main()
