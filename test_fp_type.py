import unittest
import fp_type

class TestFixedPointType(unittest.TestCase):
    def test_init(self):
        fpt = fp_type.FixedPointType(8, 0)
        self.assertEqual(fpt.bit_width, 8)
        self.assertEqual(fpt.bin_point, 0)

if __name__ == '__main__':
    unittest.main()
