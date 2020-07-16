import numpy as np

class FixedPointType:
    def __init__(self, bit_width, bin_point):
        self.bit_width = bit_width
        self.bin_point = bin_point
    def promote(self, fpt):
        whole_bits = max([fpt.bit_width - fpt.bin_point,
                          self.bit_width - self.bin_point])
        bin_point = max([self.bin_point, fpt.bin_point])
        bit_width = bin_point + whole_bits
        return FixedPointType(bit_width, bin_point)
    def __add__(self, fpt):
        fpt = self.promote(fpt)
        fpt.bit_width += 1
        return fpt
    def __mult__(self, fpt):
        bit_width = self.bit_width + fpt.bit_width
        bin_point = self.bin_point + fpt.bin_point
        return FixedPointType(bit_width, bin_point)
    def cast(self, data, fpt_in):
        if fpt_in.bin_point > self.bin_point:
            data = data >> (fpt_in.bin_point - self.bin_point)
        else:
            data = data << (self.bin_point - fpt_in.bin_point)
        data = data % self.bit_width
        return data

#def butterfly_r2(d_in, stage, dtype_in, dtype_out, dtype_tw):
#    d_in.shape = (2**stage, -1)
#    d1 = d_in[0::2].flatten()
#    d2 = d_in[1::2].flatten()
#    tw = twiddle(stage, stages, dtype_tw)
#    d2 *= tw
#    top = d1 + d2
#    bot = d1 - d2
#    d_out = np.concatenate([top, bot], axis=-1)
#    d_out = fp_cast(dout, dtype_out)
