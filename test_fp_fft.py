import unittest
import numpy as np
import fp_fft

class TestFixedPointType(unittest.TestCase):
    def test_reshape_r2(self):
        data = np.arange(8)
        top, bot = fp_fft.reshape_r2(data, 1)
        np.testing.assert_equal(top, np.array([0,1,2,3]))
        np.testing.assert_equal(bot, np.array([4,5,6,7]))
        top, bot = fp_fft.reshape_r2(data, 2)
        np.testing.assert_equal(top, np.array([0,1,4,5]))
        np.testing.assert_equal(bot, np.array([2,3,6,7]))
        top, bot = fp_fft.reshape_r2(data, 3)
        np.testing.assert_equal(top, np.array([0,2,4,6]))
        np.testing.assert_equal(bot, np.array([1,3,5,7]))
        data = np.array([np.arange(4), np.arange(4)])
        top, bot = fp_fft.reshape_r2(data, 1)
        np.testing.assert_equal(top, np.array([[0,1],[0,1]]))
        np.testing.assert_equal(bot, np.array([[2,3],[2,3]]))
        

if __name__ == '__main__':
    unittest.main()
