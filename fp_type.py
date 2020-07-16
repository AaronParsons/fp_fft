import numpy as np

class FixedPointType:
    '''Data type for fixed-point binary arithmetic.'''
    def __init__(self, bit_width, bin_point):
        self.bit_width = bit_width
        self.bin_point = bin_point
    def promote(self, fpt):
        '''Return a new FixedPointType that can losslessly represent both
        the current and the provided FixedPointTypes.'''
        whole_bits = max([fpt.bit_width - fpt.bin_point,
                          self.bit_width - self.bin_point])
        bin_point = max([self.bin_point, fpt.bin_point])
        bit_width = bin_point + whole_bits
        return FixedPointType(bit_width, bin_point)
    def __add__(self, fpt):
        '''Return the data type resulting from summing this data type
        with the provided one, including the carry bit.'''
        fpt = self.promote(fpt)
        fpt.bit_width += 1
        return fpt
    def __mul__(self, fpt):
        '''Return the data type resulting from multiplying this data type
        with the provided one.'''
        bit_width = self.bit_width + fpt.bit_width
        bin_point = self.bin_point + fpt.bin_point
        return FixedPointType(bit_width, bin_point)
    def to_float(self, data):
        '''Convert the provided integer or numpy integer array into
        the its floating-point value according to the bit width and
        binary point of this data type.'''
        data = (data % 2**self.bit_width).astype(np.float)
        data /= 2**self.bin_point
        return data
    def cast(self, data, fpt_in=None):
        '''Cast the provided integer or numpy integer array into
        the fixed-point representation corresponding to this data type.
        Optionally provide the fixed-point data type of the orignal array
        to get the binary point right.'''
        if fpt_in is not None:
            if fpt_in.bin_point > self.bin_point:
                data = data >> (fpt_in.bin_point - self.bin_point)
            else:
                data = data << (self.bin_point - fpt_in.bin_point)
        data = data % 2**self.bit_width
        return data
