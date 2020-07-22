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
    def __repr__(self):
        return 'FixedPointType(%d,%d)' % (self.bit_width, self.bin_point)
    def _mask_bitwidth(self, data, bit_width=None):
        if bit_width is None:
            bit_width = self.bit_width
        if data.dtype == np.int64:
            return (data << (64 - bit_width)) >> (64 - bit_width)
        else:
            assert(data.dtype == np.int32)
            return (data << (32 - bit_width)) >> (32 - bit_width)
    def _downshift(self, data, bin_point=None):
        if bin_point is None:
            bin_point = self.bin_point
        # work-around bc downshifting negative numbers never goes to zero
        data = np.where(data > 0, data >> bin_point,
                                  -(-data >> bin_point))
        return data
    def to_float(self, data):
        '''Convert the provided integer or numpy integer array into
        the its floating-point value according to the bit width and
        binary point of this data type.'''
        data = self._mask_bitwidth(data).astype(np.float)
        data /= 2**self.bin_point
        return data
    def from_float(self, data):
        '''Convert the provided floating point number or numpy array into
        the its fixed-point representation according to the bit width and
        binary point of this data type.'''
        data = np.around(data * 2**self.bin_point).astype(np.int32)
        data = self._mask_bitwidth(data)
        return data
    def cast(self, data, fpt_in=None):
        '''Cast the provided integer or numpy integer array into
        the fixed-point representation corresponding to this data type.
        Optionally provide the fixed-point data type of the orignal array
        to get the binary point right.'''
        if self.bit_width > 32:
            data = data.astype(np.int64) # promote to enough digits
        if fpt_in is not None:
            if fpt_in.bin_point > self.bin_point:
                data = self._downshift(data, fpt_in.bin_point - self.bin_point)
            else:
                data = data << (self.bin_point - fpt_in.bin_point)
        data = self._mask_bitwidth(data)
        if self.bit_width <= 32:
            data = data.astype(np.int32) # drop extra digits if unneeded
        return data
    def round(self, data, fpt_in=None):
        '''Cast the provided integer or numpy integer array into
        the fixed-point representation corresponding to this data type.
        Optionally provide the fixed-point data type of the orignal array
        to get the binary point right.'''
        if fpt_in is not None:
            lost_bits = fpt_in.bin_point - self.bin_point
            rnd_off = data - (self._downshift(data,lost_bits) << lost_bits)
            rnd_off = self._downshift(rnd_off, lost_bits - 1)
            data = self.cast(data, fpt_in=fpt_in)
            data += rnd_off
        data = self._mask_bitwidth(data)
        return data
