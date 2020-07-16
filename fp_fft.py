import numpy as np
from fp_type import FixedPointType

def reshape_r2(data, stage):
    shape = data.shape
    data.shape = shape[:-1] + (2**stage, -1)
    top = data[...,0::2,:]
    bot = data[...,1::2,:]
    if len(shape) == 1:
        top, bot = top.flatten(), bot.flatten()
    else:
        top.shape = shape[:-1] + (-1,)
        bot.shape = shape[:-1] + (-1,)
    data.shape = shape
    return top, bot

#def butterfly_r2(data, stage, fptype_in, fptype_out, fptype_tw):
#    top, bot = reshape_r2(data, stage)
#    tw = twiddle(stage, stages, dtype_tw)
#    d2 *= tw
#    top = d1 + d2
#    bot = d1 - d2
#    d_out = np.concatenate([top, bot], axis=-1)
#    d_out = fp_cast(dout, dtype_out)
