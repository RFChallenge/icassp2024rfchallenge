import numpy as np

get_pow = lambda s: np.mean(np.abs(s)**2)

def get_sinr(s, i, units='dB'):
    sinr = get_pow(s)/get_pow(i)
    if units == 'dB':
        return 10*np.log10(sinr)
    return sinr
