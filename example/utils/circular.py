import cmath
import numpy as np
import copy

def mean(angles, deg=True):
    """Circular mean of angle data(default to degree)"""
    a = np.deg2rad(angles) if deg else np.array(angles)
    angles_complex = np.frompyfunc(cmath.exp, 1, 1)(a * 1j)
    mean = cmath.phase(angles_complex.sum()) % (2 * np.pi)
    return round(np.rad2deg(mean) if deg else mean, 7)


def var(angles, deg=True):
    """Circular variance of angle data(default to degree)
    0 <= var <= 1
    """
    a = np.deg2rad(angles) if deg else np.array(angles)
    angles_complex = np.frompyfunc(cmath.exp, 1, 1)(a * 1j)
    r = abs(angles_complex.sum()) / len(angles)
    return round(1 - r, 7)


def std(angles, deg=True):
    """Circular standard deviation of angle data(default to degree)
    0 <= std
    """
    a = np.deg2rad(angles) if deg else np.array(angles)
    
    angles_complex = np.frompyfunc(cmath.exp, 1, 1)(copy.deepcopy(a) * 1j)
    r = abs(angles_complex.sum()) / len(angles)
    std = np.sqrt(-2 * np.log(r))
    
    
    return round(np.rad2deg(std) if deg else std, 7)
    


