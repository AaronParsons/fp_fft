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
    ind = np.arange(2**(stages-1))
    ind = bit_reverse(ind >> (stages - stage), stage-1)
    theta = -np.pi * FixedPointType(stage, stage-1).to_float(ind)
    tw_real = np.cos(theta)
    tw_imag = np.sin(theta)
    tw_real = fptype.from_float(tw_real)
    tw_imag = fptype.from_float(tw_imag)
    mx = 2**(fptype.bit_width-1) - 1
    return tw_real.clip(-mx,mx), tw_imag.clip(-mx,mx)

def butterfly_r2(d_real, d_imag, stage, stages, fptype_in, fptype_out, fptype_tw):
    top_real, bot_real = reshape_r2(d_real, stage)
    top_imag, bot_imag = reshape_r2(d_imag, stage)
    tw_real, tw_imag = twiddle_r2(stage, stages, fptype_tw)
    btw_real = bot_real * tw_real - bot_imag * tw_imag
    btw_imag = bot_imag * tw_real + bot_real * tw_imag
    fptype_btw = fptype_in * fptype_tw  # promote fptype for tw product
    fptype_btw = fptype_btw + fptype_btw # promote fptype for tw sum
    #print(top_real, fptype_in)
    #print(top_imag, fptype_in)
    #print()
    #print(bot_real, fptype_in)
    #print(bot_imag, fptype_in)
    #print()
    #print(fptype_btw.cast(top_real, fptype_in), fptype_btw)
    #print(fptype_btw.cast(top_imag, fptype_in), fptype_btw)
    #print()
    #print(btw_real, fptype_btw)
    #print(btw_imag, fptype_btw)
    #print()
    tbtw_real = fptype_btw.cast(top_real, fptype_in) + btw_real
    tbtw_imag = fptype_btw.cast(top_imag, fptype_in) + btw_imag
    bbtw_real = fptype_btw.cast(top_real, fptype_in) - btw_real
    bbtw_imag = fptype_btw.cast(top_imag, fptype_in) - btw_imag
    fptype_tbtw = fptype_btw + fptype_btw # promote fptype for butterfly sum
    #print(tbtw_real, fptype_tbtw)
    #print(tbtw_imag, fptype_tbtw)
    #print()
    #print(bbtw_real, fptype_tbtw)
    #print(bbtw_imag, fptype_tbtw)
    #print()
    out_real = unreshape_r2(tbtw_real, bbtw_real, stage)
    out_imag = unreshape_r2(tbtw_imag, bbtw_imag, stage)
    #print(out_real, fptype_tbtw)
    #print(out_imag, fptype_tbtw)
    #print()
    #out_real = np.concatenate([tbtw_real, bbtw_real], axis=-1)
    #out_imag = np.concatenate([tbtw_imag, bbtw_imag], axis=-1)
    out_real = fptype_out.round(out_real, fptype_tbtw)
    out_imag = fptype_out.round(out_imag, fptype_tbtw)
    #print(out_real, fptype_out)
    #print(out_imag, fptype_out)
    return out_real, out_imag
