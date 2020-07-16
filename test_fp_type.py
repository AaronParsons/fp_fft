import unittest
import numpy as np
import fp_type

class TestFixedPointType(unittest.TestCase):
    def test_init(self):
        fpt = fp_type.FixedPointType(8, 0)
        self.assertEqual(fpt.bit_width, 8)
        self.assertEqual(fpt.bin_point, 0)
    def test_promote(self):
        fpt1 = fp_type.FixedPointType(8, 0)
        fpt2 = fp_type.FixedPointType(8, 2)
        fpt3 = fp_type.FixedPointType(3, 2)
        fpt4 = fp_type.FixedPointType(8, 7)
        fpt_11 = fpt1.promote(fpt1)
        fpt_12 = fpt1.promote(fpt2)
        fpt_13 = fpt1.promote(fpt3)
        fpt_23 = fpt2.promote(fpt3)
        fpt_14 = fpt1.promote(fpt4)
        fpt_24 = fpt2.promote(fpt4)
        fpt_34 = fpt3.promote(fpt4)
        self.assertEqual(fpt_11.bit_width, 8)
        self.assertEqual(fpt_11.bin_point, 0)
        self.assertEqual(fpt_12.bit_width, 10)
        self.assertEqual(fpt_12.bin_point, 2)
        self.assertEqual(fpt_13.bit_width, 10)
        self.assertEqual(fpt_13.bin_point, 2)
        self.assertEqual(fpt_23.bit_width, 8)
        self.assertEqual(fpt_23.bin_point, 2)
        self.assertEqual(fpt_14.bit_width, 15)
        self.assertEqual(fpt_14.bin_point, 7)
        self.assertEqual(fpt_24.bit_width, 13)
        self.assertEqual(fpt_24.bin_point, 7)
        self.assertEqual(fpt_34.bit_width, 8)
        self.assertEqual(fpt_34.bin_point, 7)
    def test_add(self):
        fpt1 = fp_type.FixedPointType(8, 0)
        fpt2 = fp_type.FixedPointType(8, 2)
        fpt3 = fp_type.FixedPointType(3, 2)
        fpt4 = fp_type.FixedPointType(8, 7)
        fpt_11 = fpt1 + fpt1
        fpt_12 = fpt1 + fpt2
        fpt_13 = fpt1 + fpt3
        fpt_23 = fpt2 + fpt3
        fpt_14 = fpt1 + fpt4
        fpt_24 = fpt2 + fpt4
        fpt_34 = fpt3 + fpt4
        self.assertEqual(fpt_11.bit_width, 9)
        self.assertEqual(fpt_11.bin_point, 0)
        self.assertEqual(fpt_12.bit_width, 11)
        self.assertEqual(fpt_12.bin_point, 2)
        self.assertEqual(fpt_13.bit_width, 11)
        self.assertEqual(fpt_13.bin_point, 2)
        self.assertEqual(fpt_23.bit_width, 9)
        self.assertEqual(fpt_23.bin_point, 2)
        self.assertEqual(fpt_14.bit_width, 16)
        self.assertEqual(fpt_14.bin_point, 7)
        self.assertEqual(fpt_24.bit_width, 14)
        self.assertEqual(fpt_24.bin_point, 7)
        self.assertEqual(fpt_34.bit_width, 9)
        self.assertEqual(fpt_34.bin_point, 7)
    def test_mult(self):
        fpt1 = fp_type.FixedPointType(8, 0)
        fpt2 = fp_type.FixedPointType(8, 2)
        fpt3 = fp_type.FixedPointType(3, 2)
        fpt4 = fp_type.FixedPointType(8, 7)
        fpt_11 = fpt1 * fpt1
        fpt_12 = fpt1 * fpt2
        fpt_13 = fpt1 * fpt3
        fpt_23 = fpt2 * fpt3
        fpt_14 = fpt1 * fpt4
        fpt_24 = fpt2 * fpt4
        fpt_34 = fpt3 * fpt4
        self.assertEqual(fpt_11.bit_width, 16)
        self.assertEqual(fpt_11.bin_point, 0)
        self.assertEqual(fpt_12.bit_width, 16)
        self.assertEqual(fpt_12.bin_point, 2)
        self.assertEqual(fpt_13.bit_width, 11)
        self.assertEqual(fpt_13.bin_point, 2)
        self.assertEqual(fpt_23.bit_width, 11)
        self.assertEqual(fpt_23.bin_point, 4)
        self.assertEqual(fpt_14.bit_width, 16)
        self.assertEqual(fpt_14.bin_point, 7)
        self.assertEqual(fpt_24.bit_width, 16)
        self.assertEqual(fpt_24.bin_point, 9)
        self.assertEqual(fpt_34.bit_width, 11)
        self.assertEqual(fpt_34.bin_point, 9)
    def test_to_float(self):
        fpt1 = fp_type.FixedPointType(8, 0)
        fpt2 = fp_type.FixedPointType(8, 2)
        fpt3 = fp_type.FixedPointType(3, 2)
        fpt4 = fp_type.FixedPointType(8, 7)
        data = np.arange(10)
        np.testing.assert_equal(data.astype(np.float), fpt1.to_float(data))
        np.testing.assert_equal(data.astype(np.float)/4, fpt2.to_float(data))
        np.testing.assert_equal((data % 8).astype(np.float)/4, fpt3.to_float(data))
        np.testing.assert_equal(data.astype(np.float)/128, fpt4.to_float(data))
    def test_cast(self):
        fpt1 = fp_type.FixedPointType(8, 0)
        fpt2 = fp_type.FixedPointType(8, 2)
        fpt3 = fp_type.FixedPointType(3, 2)
        data = np.arange(10)
        np.testing.assert_equal(data, fpt1.cast(data, fpt_in=fpt1))
        np.testing.assert_equal(data << 2, fpt2.cast(data, fpt_in=fpt1))
        np.testing.assert_equal(data >> 2, fpt1.cast(data, fpt_in=fpt2))
        np.testing.assert_equal(data % 8, fpt3.cast(data, fpt_in=fpt2))
        np.testing.assert_equal(data % 8, fpt3.cast(data))

if __name__ == '__main__':
    unittest.main()
