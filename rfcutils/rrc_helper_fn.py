import sionna as sn
import numpy as np
import tensorflow as tf

def get_psf(samples_per_symbol, span_in_symbols, beta):
    # samples_per_symbol: Number of samples per symbol, i.e., the oversampling factor
    # beta: Roll-off factor
    # span_in_symbols: Filter span in symbold
    rrcf = sn.signal.RootRaisedCosineFilter(span_in_symbols, samples_per_symbol, beta)
    return rrcf

def matched_filter(sig, samples_per_symbol, span_in_symbols, beta):
    rrcf = get_psf(samples_per_symbol, span_in_symbols, beta)
    x_mf = rrcf(sig, padding="same")
    return x_mf