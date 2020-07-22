import numpy as np
from fp_type import FixedPointType

def reshape_r2(data, stage):
    shape = data.shape
    data.shape = shape[:-1] + (2**stage, -1)
    top = data[...,0::2,:].copy() # inefficient, but avoids non-contiguous
    bot = data[...,1::2,:].copy() # inefficient, but avoids non-contiguous
    if len(shape) == 1:
        top, bot = top.flatten(), bot.flatten()
    else:
        top.shape = bot.shape = shape[:-1] + (-1,)
    data.shape = shape
    return top, bot

def unreshape_r2(top, bot, stage):
    shape = top.shape
    nfreq = shape[-1] * 2
    data = np.empty(shape[:-1] + (2**stage, nfreq // (2**stage)), 
                    dtype=top.dtype)
    top.shape = bot.shape = shape[:-1] + (-1, data.shape[-1])
    data[...,0::2,:] = top
    data[...,1::2,:] = bot
    data.shape = shape[:-1] + (nfreq,)
    top.shape = bot.shape = shape
    return data

def bit_reverse(data, nbits):
    if nbits <= 1:
        return data & 1
    else:
        return 2**(nbits-1) * (data & 1) + bit_reverse(data >> 1, nbits-1)

def twiddle_r2(stage, stages, fptype):
    ind = np.arange(2**(stages-1), dtype=np.int32)
    ind = bit_reverse(ind >> (stages - stage), stage-1)
    theta = -np.pi * FixedPointType(stage, stage-1).to_float(ind)
    tw_real = np.cos(theta)
    tw_imag = np.sin(theta)
    mx = (2**(fptype.bit_width-1) - 1) / 2**(fptype.bit_width-1)
    tw_real = fptype.from_float(tw_real.clip(-mx,mx))
    tw_imag = fptype.from_float(tw_imag.clip(-mx,mx))
    return tw_real, tw_imag

def butterfly_r2(d_real, d_imag, stage, stages,
                 fptype_in, fptype_out, fptype_tw,
                 shift):
    top_real, bot_real = reshape_r2(d_real, stage)
    top_imag, bot_imag = reshape_r2(d_imag, stage)
    tw_real, tw_imag = twiddle_r2(stage, stages, fptype_tw)
    fptype_btw = fptype_in * fptype_tw  # promote fptype for tw product
    fptype_btw = fptype_btw + fptype_btw # promote fptype for tw sum
    if fptype_btw.bit_width > 32:
        bot_real = bot_real.astype(np.int64)
        bot_imag = bot_imag.astype(np.int64)
    btw_real = bot_real * tw_real - bot_imag * tw_imag
    btw_imag = bot_imag * tw_real + bot_real * tw_imag
    tbtw_real = fptype_btw.cast(top_real, fptype_in) + btw_real
    tbtw_imag = fptype_btw.cast(top_imag, fptype_in) + btw_imag
    bbtw_real = fptype_btw.cast(top_real, fptype_in) - btw_real
    bbtw_imag = fptype_btw.cast(top_imag, fptype_in) - btw_imag
    fptype_tbtw = fptype_btw + fptype_btw # promote fptype for butterfly sum
    out_real = unreshape_r2(tbtw_real, bbtw_real, stage)
    out_imag = unreshape_r2(tbtw_imag, bbtw_imag, stage)
    if shift:
        fptype_tbtw.bin_point += 1
    out_real = fptype_out.round(out_real, fptype_tbtw)
    out_imag = fptype_out.round(out_imag, fptype_tbtw)
    return out_real, out_imag

def fft_r2(d_real, d_imag, stages, fptype_in, fptype_out, fptype_tw,
           shifts=None):
    if type(fptype_in) == FixedPointType:
        fptype_in = [fptype_in] * stages
    # fptype_in must be a FixedPointType or an iterable of length stages
    assert(len(fptype_in) == stages)

    if type(fptype_out) == FixedPointType:
        fptype_out = [fptype_out] * stages
    # fptype_out must be a FixedPointType or an iterable of length stages
    assert(len(fptype_out) == stages)

    if type(fptype_tw) == FixedPointType:
        fptype_tw = [fptype_tw] * stages
    # fptype_tw must be a FixedPointType or an iterable of length stages
    assert(len(fptype_tw) == stages)

    if shifts is None:
        shifts = [0] * stages
    # shift must either be none or iterable of length stages
    assert(len(shifts) == stages)

    # dimensions of data must agree with stages
    assert(d_real.shape[-1] == 2**stages)
    assert(d_imag.shape[-1] == 2**stages)

    # Do all the butterflies
    out_real, out_imag = d_real, d_imag
    for stage, ti, to, tt, shift in zip(range(1, stages+1),
                                        fptype_in, fptype_out, fptype_tw,
                                        shifts):
        out_real, out_imag = butterfly_r2(out_real, out_imag,
                                          stage, stages, ti, to, tt, shift)
    
    # Undo the bit-reversed ordering of output channels
    unscramble = bit_reverse(np.arange(2**stages), stages)
    return out_real[...,unscramble], out_imag[...,unscramble]
        
