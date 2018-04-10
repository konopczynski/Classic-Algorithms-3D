import numpy as np


def Normalize(I,nMin,nMax):
    Imin = np.min(I)
    Imax = np.max(I)
    I = (I-Imin) * ((nMax-nMin)/(Imax-Imin)) + nMin
    return I



