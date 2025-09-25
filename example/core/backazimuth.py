import sys
sys.path.append(sys.path.append("nipfilter/"))
from core import stransform, istransform
import filter as filt_par
#sys.path.append(sys.path.append("s-transform/"))
import numpy as np

def calc_baz(trZ, trE, trN, Fs, azimuth, polarization):
    Sv, Tv, Fv = stransform(trZ, Fs, return_time_freq=True)
    Se, _, _ = stransform(trE, Fs, return_time_freq=True)
    Sn, _, _ = stransform(trN, Fs, return_time_freq=True)


    xpr = -int(np.sign(np.sin(np.radians(azimuth))))
    baz = filt_par.instantaneous_azimuth(Sv, Sn, Se, polarization, xpr)
    baz2 = filt_par.instantaneous_azimuth(Sv, Sn, Se, polarization, -xpr)

    Sv_shifted = filt_par.get_shift(polarization) * Sv
    Sr1 = filt_par.rotate_NE_RT(Sn, Se, baz)[0]
    nip1 = filt_par.NIP(Sr1, Sv_shifted)
    Sr2 = filt_par.rotate_NE_RT(Sn, Se, baz2)[0]
    nip2 = filt_par.NIP(Sr2, Sv_shifted)
    baz[nip1 > nip2] += 180.0

    nip_final = nip1
    dom_idx = np.where(nip1<nip2)
    nip_final[dom_idx] = nip2[dom_idx]
    weak_idx = np.where(nip_final<0.8)
    baz[weak_idx] = np.nan
    baz[baz>180.0] -= 360.0
    
    return baz, nip_final, Tv, Fv, Sv, Se, Sn