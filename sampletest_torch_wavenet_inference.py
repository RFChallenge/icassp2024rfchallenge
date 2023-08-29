import os, sys
import numpy as np
import random
import h5py
from tqdm import tqdm
import pickle

import torch
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
import rfcutils
from src.torchwavenet import Wave
from omegaconf import OmegaConf
from src.config_torchwavenet import Config, parse_configs

get_db = lambda p: 10*np.log10(p)
get_pow = lambda s: np.mean(np.abs(s)**2)
get_sinr = lambda s, i: get_pow(s)/get_pow(i)
get_sinr_db = lambda s, i: get_db(get_sinr(s,i))

sig_len = 40960
n_per_batch = 100
all_sinr = np.arange(-30, 0.1, 3)

def get_soi_generation_fn(soi_sig_type):
    if soi_sig_type == 'QPSK':
        generate_soi = lambda n, s_len: rfcutils.generate_qpsk_signal(n, s_len//16)
        demod_soi = rfcutils.qpsk_matched_filter_demod
    elif soi_sig_type == 'QAM16':
        generate_soi = lambda n, s_len: rfcutils.generate_qam16_signal(n, s_len//16)
        demod_soi = rfcutils.qam16_matched_filter_demod
    elif soi_sig_type ==  'QPSK2':
        generate_soi = lambda n, s_len: rfcutils.generate_qpsk2_signal(n, s_len//4)
        demod_soi = rfcutils.qpsk2_matched_filter_demod
    elif soi_sig_type == 'OFDMQPSK':
        generate_soi = lambda n, s_len: rfcutils.generate_ofdm_signal(n, s_len//80)
        _,_,_,RES_GRID = rfcutils.generate_ofdm_signal(1, sig_len//80)
        demod_soi = lambda s: rfcutils.ofdm_demod(s, RES_GRID)
    else:
        raise Exception("SOI Type not recognized")
    return generate_soi, demod_soi


def run_inference(all_sig_mixture, soi_type, interference_sig_type):

    generate_soi, demod_soi = get_soi_generation_fn(soi_type)

    cfg = OmegaConf.load("src/configs/wavenet.yml")
    cli_cfg = None
    cfg: Config = Config(**parse_configs(cfg, cli_cfg))
    cfg.model_dir = f"torchmodels/dataset_{soi_type.lower()}_{interference_sig_type.lower()}_mixture_wavenet"
    nn_model = Wave(cfg.model).cuda()
    nn_model.load_state_dict(torch.load(cfg.model_dir+"/weights.pt")['model'])

    sig_mixture_comp = tf.stack((tf.math.real(all_sig_mixture), tf.math.imag(all_sig_mixture)), axis=-1)
    with torch.no_grad():
        nn_model.eval()
        all_sig1_out = []
        bsz = 100
        for i in tqdm(range(sig_mixture_comp.shape[0]//bsz)):
            sig_input = torch.Tensor(sig_mixture_comp[i*bsz:(i+1)*bsz].numpy()).transpose(1, 2).to('cuda')
            sig1_out = nn_model(sig_input)
            all_sig1_out.append(sig1_out.transpose(1,2).detach().cpu().numpy())
    sig1_out = tf.concat(all_sig1_out, axis=0)
    print(sig1_out.shape)
    sig1_est = tf.complex(sig1_out[:,:,0], sig1_out[:,:,1])

    bit_est = []
    for idx, sinr_db in tqdm(enumerate(all_sinr)):
        bit_est_batch, _ = demod_soi(sig1_est[idx*n_per_batch:(idx+1)*n_per_batch])
        bit_est.append(bit_est_batch)
    bit_est = tf.concat(bit_est, axis=0)
    sig1_est, bit_est = sig1_est.numpy(), bit_est.numpy()
    return sig1_est, bit_est

if __name__ == "__main__":
    soi_type, interference_sig_type = sys.argv[1], sys.argv[2]
    testset_identifier = sys.argv[3] # 'TestSet1Example'
    id_string = 'Default_Torch_WaveNet'
    all_sig_mixture = np.load(os.path.join('dataset', f'{testset_identifier}_testmixture_{soi_type}_{interference_sig_type}.npy'))
    sig1_est, bit1_est = run_inference(all_sig_mixture, soi_type, interference_sig_type)
    np.save(os.path.join('outputs', f'{id_string}_{testset_identifier}_estimated_soi_{soi_type}_{interference_sig_type}'), sig1_est)
    np.save(os.path.join('outputs', f'{id_string}_{testset_identifier}_estimated_bits_{soi_type}_{interference_sig_type}'), bit1_est)
