import sionna as sn
import numpy as np
import tensorflow as tf

NFFT = 64
CP_LEN = 16
OFDM_LEN = NFFT + CP_LEN
CODERATE = 1
n_streams_per_tx = 1

# Binary source to generate uniform i.i.d. bits
binary_source = sn.utils.BinarySource()

# 4-QAM constellation
NUM_BITS_PER_SYMBOL = 2
constellation = sn.mapping.Constellation("qam", NUM_BITS_PER_SYMBOL, trainable=False) # The constellation is set to be NOT trainable
stream_manager = sn.mimo.StreamManagement(np.array([[1]]), 1)

# Mapper and demapper
mapper = sn.mapping.Mapper(constellation=constellation)
demapper = sn.mapping.Demapper("app", constellation=constellation)

# AWGN channel
awgn_channel = sn.channel.AWGN()

# # The encoder maps information bits to coded bits
# encoder = sn.fec.ldpc.LDPC5GEncoder(k, n)

def get_resource_grid(num_ofdm_symbols):
    RESOURCE_GRID = sn.ofdm.ResourceGrid( num_ofdm_symbols=num_ofdm_symbols,
                                          fft_size=NFFT,
                                          subcarrier_spacing=20e6/NFFT,
                                          num_tx=1,
                                          num_streams_per_tx=n_streams_per_tx,
                                          num_guard_carriers=(4,3),
                                          dc_null=True,
                                          cyclic_prefix_length=CP_LEN,
                                          pilot_pattern=None,
                                          pilot_ofdm_symbol_indices=[])
    return RESOURCE_GRID

def generate_ofdm_signal(batch_size, num_ofdm_symbols, ebno_db=None):
    RESOURCE_GRID = get_resource_grid(num_ofdm_symbols)

    # Number of coded bits in a resource grid
    n = int(RESOURCE_GRID.num_data_symbols*NUM_BITS_PER_SYMBOL)
    # Number of information bits in a resource groud
    k = int(n*CODERATE)

    bits = binary_source([batch_size, 1, n_streams_per_tx, k])
    return modulate_ofdm_signal(bits, RESOURCE_GRID, ebno_db)


def ofdm_demod(sig, RESOURCE_GRID, no=1e-4):
    rg_demapper = sn.ofdm.ResourceGridDemapper(RESOURCE_GRID, stream_manager)
    ofdm_demod_block = sn.ofdm.OFDMDemodulator(NFFT, 0, CP_LEN)

    x_ofdm_demod = ofdm_demod_block(sig)
    x_demod = rg_demapper(tf.reshape(x_ofdm_demod, (sig.shape[0],1,1,-1,NFFT)))
    llr = demapper([x_demod,no])
    llr = tf.squeeze(llr, axis=[1,2])
    return tf.cast(llr > 0, tf.float32), x_ofdm_demod


def modulate_ofdm_signal(info_bits, RESOURCE_GRID, ebno_db=None):
    # codewords = encoder(info_bits) # using uncoded bits for now
    codewords = info_bits
    rg_mapper = sn.ofdm.ResourceGridMapper(RESOURCE_GRID)
    ofdm_mod = sn.ofdm.OFDMModulator(RESOURCE_GRID.cyclic_prefix_length)

    x = mapper(codewords)
    x_rg = rg_mapper(x)
    x_ofdm = ofdm_mod(x_rg)

    if ebno_db is None:
        y = x_ofdm
    else:
        no = sn.utils.ebnodb2no(ebno_db=10.0,
                                num_bits_per_symbol=NUM_BITS_PER_SYMBOL,
                                coderate=CODERATE,
                                resource_grid=RESOURCE_GRID)
        y = awgn_channel([x_ofdm, no])
    # squeeze axis corresponding to num_tx, num_streams_per_tx (assumed to be 1)
    y = tf.squeeze(y, axis=[1,2])
    x_rg = tf.squeeze(x_rg, axis=[1,2])
    info_bits = tf.squeeze(info_bits, axis=[1,2])
    return y, x_rg, info_bits, RESOURCE_GRID

