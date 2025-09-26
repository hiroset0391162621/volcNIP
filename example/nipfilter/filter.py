
from functools import lru_cache
import warnings
import numpy as np


from core import stransform, istransform

def get_shift(polarization):
    """
    Get the appropriate pi/2 phase advance or delay for the provided
    polarization.

    Parameters
    ----------
    polarization : str, {'retrograde', 'prograde', 'linear'}
        'retrograde' returns i, for a pi/2 phase advance.
        'prograde' or 'linear' returns -i, for a pi/2 phase delay

    Returns
    -------
    numpy.complex128
        Multiply this value (i or -i) into a complex vertical S-transform to
        shift its phase.

    """
    if polarization is 'retrograde':
        # phase advance
        shft = np.array(1j)
    elif polarization in ('prograde', 'linear'):
        # phase delay
        shft = -np.array(1j)
    else:
        raise ValueError("Polarization must be either 'prograde', 'retrograde', or 'linear'")

    return shft


def shift_phase(Sv, polarization):
    """
    Phase-shift an s-transform by the appropriate phase shift for
    prograde/retrograde motion.

    Shift is done on a complex MxN array by multiplication with either i or -i
    (imaginary unit).  This is mostly a reference for how to do/interpret phase
    shifts, as it's such a simple thing to do outside of a function.

    Parameters
    ----------
    Sv : numpy.ndarray (complex, rank 2)
    polarization : str, {'retrograde', 'prograde', 'linear'}
        'retrograde' will apply a pi/2 phase advance (normal Rayleigh waves)
        'prograde' or 'linear' will apply a pi/2 phase delay

    Returns
    -------
    numpy.ndarray (real, rank 2)

    """
    shft = get_shift(polarization)

    return Sv * shft




def rotate_NE_RT(Sn, Se, az):
    """
    Rotate North and East s-transforms to radial and transverse, through the propagation angle.

    Parameters
    ----------
    Sn, Se : numpy.ndarray (complex, rank 2)
        Complex, equal-sized s-transform arrays, for North and East components, respectively.
    az : float
        Rotation angle [degrees].

    Returns
    -------
    Sr, St : numpy.ndarray (rank 2)
        Complex s-transform arrays for radial and transverse components, respectively.
    """
    theta = np.radians(az)

    Sr = np.cos(theta)*Sn + np.sin(theta)*Se
    St = -np.sin(theta)*Sn + np.cos(theta)*Se

    return Sr, St


def NIP(Sr, Sv, polarization=None, eps=None):
    """
    Get the normalized inner product of two complex MxN stockwell transforms.

    Parameters
    ----------
    Sr, Sv: numpy.ndarray (complex, rank 2)
        The radial and vertical component s-transforms. If the polarization argument is omitted,
        Sv is assumed to be phase-shifted according to the desired polarization.
    polarization : str, optional
        If provided, the Sv will be phase-shifted according to this string before calculating the NIP.
        'retrograde' will apply a pi/2 phase advance (1j * Sv)
        'prograde' or 'linear' will apply a pi/2 phase delay (-1j * Sv)
    eps : float, optional
        Tolerance for small denominator values, for numerical stability.
        Useful for synthetic noise-free data.  Authors used 0.04.

    Returns
    -------
    nip : numpy.ndarray (rank 2)
        MxN array of floats between -1 and 1.
    """
    if polarization:
        Svhat = shift_phase(Sv, polarization)
    else:
        # Just a literal inner product, no shift.
        Svhat = Sv

    
    
    Avhat = np.abs(Svhat)
    if eps is not None:
        mask = (Avhat / Avhat.max()) < eps
        Avhat[mask] += eps*Avhat.max()

    ip = (Sr.real)*(Svhat.real) + (Sr.imag)*(Svhat.imag)
    n = np.abs(Sr) * Avhat
    nip = ip/n

    #nip = np.sin(np.angle(Sr) - np.angle(Sv))
    
    return nip
    

