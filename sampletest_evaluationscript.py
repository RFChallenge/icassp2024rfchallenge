import os, sys
import numpy as np
from tqdm import tqdm
import pickle

get_db = lambda p: 10*np.log10(p)
get_pow = lambda s: np.mean(np.abs(s)**2)
get_sinr = lambda s, i: get_pow(s)/get_pow(i)
get_sinr_db = lambda s, i: get_db(get_sinr(s,i))

sig_len = 40960
n_per_batch = 100
all_sinr = np.arange(-30, 0.1, 3)

def run_demod_test(sig1_est, bit1_est, soi_type, interference_sig_type, testset_identifier):
    all_sig_mixture, all_sig1, all_bits1 = pickle.load(open(os.path.join('dataset', f'GroundTruth_{testset_identifier}_Dataset_{soi_type}_{interference_sig_type}.pkl'), 'rb'))

    # Evaluation pipeline
    def eval_mse(all_sig_est, all_sig_soi):
        assert all_sig_est.shape == all_sig_soi.shape, 'Invalid SOI estimate shape'
        return np.mean(np.abs(all_sig_est - all_sig_soi)**2, axis=1)

    def eval_ber(bit_est, bit_true):
        ber = np.sum((bit_est != bit_true).astype(np.float32), axis=1) / bit_true.shape[1]
        assert bit_est.shape == bit_true.shape, 'Invalid bit estimate shape'
        return ber

    all_mse, all_ber = [], []
    for idx, sinr in tqdm(enumerate(all_sinr)):
        batch_mse =  eval_mse(sig1_est[idx*n_per_batch:(idx+1)*n_per_batch], all_sig1[idx*n_per_batch:(idx+1)*n_per_batch])
        bit_true_batch = all_bits1[idx*n_per_batch:(idx+1)*n_per_batch]
        batch_ber = eval_ber(bit1_est[idx*n_per_batch:(idx+1)*n_per_batch], bit_true_batch)
        all_mse.append(batch_mse)
        all_ber.append(batch_ber)

    all_mse, all_ber = np.array(all_mse), np.array(all_ber)

    mse_mean = 10*np.log10(np.mean(all_mse, axis=-1))
    ber_mean = np.mean(all_ber, axis=-1)
    print(f'{"SINR [dB]":>12} {"MSE [dB]":>12} {"BER":>12}')
    print('==================================================')
    for sinr, mse, ber in zip(all_sinr, mse_mean, ber_mean):
        print(f'{sinr:>12} {mse:>12,.5f} {ber:>12,.5f}')
    return mse_mean, ber_mean

if __name__ == "__main__":
    soi_type, interference_sig_type = sys.argv[1], sys.argv[2]
    testset_identifier = sys.argv[3] # 'TestSet1Example'
    id_string = sys.argv[4] #'Default_TF_UNet'

    sig1_est = np.load(os.path.join('outputs', f'{id_string}_{testset_identifier}_estimated_soi_{soi_type}_{interference_sig_type}.npy'))
    bit1_est = np.load(os.path.join('outputs', f'{id_string}_{testset_identifier}_estimated_bits_{soi_type}_{interference_sig_type}.npy'))
    assert ~np.isnan(sig1_est).any(), 'NaN or Inf in Signal Estimate'
    assert ~np.isnan(bit1_est).any(), 'NaN or Inf in Bit Estimate'

    mse_mean, ber_mean = run_demod_test(sig1_est, bit1_est, soi_type, interference_sig_type, testset_identifier)
    pickle.dump((mse_mean, ber_mean),open(os.path.join('outputs', f'{id_string}_{testset_identifier}_exports_summary_{soi_type}_{interference_sig_type}.pkl'), 'wb'))
