#!/bin/bash

# # Create examples in h5 files under the 'dataset' folder
# python dataset_utils/example_generate_rfc_mixtures.py --soi_sig_type QPSK --interference_sig_type EMISignal1
# python dataset_utils/example_generate_rfc_mixtures.py --soi_sig_type QPSK --interference_sig_type CommSignal2
# python dataset_utils/example_generate_rfc_mixtures.py --soi_sig_type QPSK --interference_sig_type CommSignal3
# python dataset_utils/example_generate_rfc_mixtures.py --soi_sig_type QPSK --interference_sig_type CommSignal5G1


# python dataset_utils/example_generate_rfc_mixtures.py --soi_sig_type OFDMQPSK --interference_sig_type EMISignal1
# python dataset_utils/example_generate_rfc_mixtures.py --soi_sig_type OFDMQPSK --interference_sig_type CommSignal2
# python dataset_utils/example_generate_rfc_mixtures.py --soi_sig_type OFDMQPSK --interference_sig_type CommSignal3
# python dataset_utils/example_generate_rfc_mixtures.py --soi_sig_type OFDMQPSK --interference_sig_type CommSignal5G1


# # Create TFDS Dataset from 'dataset' folder (for TF UNet training)
# tfds build dataset_utils/tfds_scripts/Dataset_QPSK_CommSignal2_Mixture.py --data_dir tfds/
# tfds build dataset_utils/tfds_scripts/Dataset_QPSK_CommSignal3_Mixture.py --data_dir tfds/
# tfds build dataset_utils/tfds_scripts/Dataset_QPSK_CommSignal5G1_Mixture.py --data_dir tfds/
# tfds build dataset_utils/tfds_scripts/Dataset_QPSK_EMISignal1_Mixture.py --data_dir tfds/

# tfds build dataset_utils/tfds_scripts/Dataset_OFDMQPSK_CommSignal2_Mixture.py --data_dir tfds/
# tfds build dataset_utils/tfds_scripts/Dataset_OFDMQPSK_CommSignal3_Mixture.py --data_dir tfds/
# tfds build dataset_utils/tfds_scripts/Dataset_OFDMQPSK_CommSignal5G1_Mixture.py --data_dir tfds/
# tfds build dataset_utils/tfds_scripts/Dataset_OFDMQPSK_EMISignal1_Mixture.py --data_dir tfds/


# # Create NPY Dataset from 'dataset' folder (for PyTorch Wavenet training)
# python dataset_utils/example_preprocess_npy_dataset.py QPSK_CommSignal2
# python dataset_utils/example_preprocess_npy_dataset.py QPSK_CommSignal3
# python dataset_utils/example_preprocess_npy_dataset.py QPSK_CommSignal5G1
# python dataset_utils/example_preprocess_npy_dataset.py QPSK_EMISignal1

# python dataset_utils/example_preprocess_npy_dataset.py OFDMQPSK_CommSignal2
# python dataset_utils/example_preprocess_npy_dataset.py OFDMQPSK_CommSignal3
# python dataset_utils/example_preprocess_npy_dataset.py OFDMQPSK_CommSignal5G1
# python dataset_utils/example_preprocess_npy_dataset.py OFDMQPSK_EMISignal1


# Create training set examples similar to TestSet from the Grand Challenge specifications
python dataset_utils/example_generate_competition_trainmixture.py --soi_sig_type QPSK --interference_sig_type EMISignal1 --n_per_batch 1000
python dataset_utils/example_generate_competition_trainmixture.py --soi_sig_type QPSK --interference_sig_type CommSignal2 --n_per_batch 1000
python dataset_utils/example_generate_competition_trainmixture.py --soi_sig_type QPSK --interference_sig_type CommSignal3 --n_per_batch 1000
python dataset_utils/example_generate_competition_trainmixture.py --soi_sig_type QPSK --interference_sig_type CommSignal5G1 --n_per_batch 1000

python dataset_utils/example_generate_competition_trainmixture.py --soi_sig_type OFDMQPSK --interference_sig_type EMISignal1 --n_per_batch 1000
python dataset_utils/example_generate_competition_trainmixture.py --soi_sig_type OFDMQPSK --interference_sig_type CommSignal2 --n_per_batch 1000
python dataset_utils/example_generate_competition_trainmixture.py --soi_sig_type OFDMQPSK --interference_sig_type CommSignal3 --n_per_batch 1000
python dataset_utils/example_generate_competition_trainmixture.py --soi_sig_type OFDMQPSK --interference_sig_type CommSignal5G1 --n_per_batch 1000
