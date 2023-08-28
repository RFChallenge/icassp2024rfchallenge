import os, sys

import numpy as np
import random
import h5py
import argparse

import rfcutils
import tensorflow_datasets as tfds
import tensorflow as tf

import glob, h5py


from src import unet_model as unet
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])

bsz = 64

all_datasets = ['QPSK_CommSignal2', 'QPSK2_CommSignal2', 'QAM16_CommSignal2', 'OFDMQPSK_CommSignal2',
                'QPSK_CommSignal3', 'QPSK2_CommSignal3', 'QAM16_CommSignal3', 'OFDMQPSK_CommSignal3', 'CommSignal2_CommSignal3',
                'QPSK_EMISignal1', 'QPSK2_EMISignal1', 'QAM16_EMISignal1', 'OFDMQPSK_EMISignal1', 'CommSignal2_EMISignal1',
                'QPSK_CommSignal5G1', 'QPSK2_CommSignal5G1', 'QAM16_CommSignal5G1', 'OFDMQPSK_CommSignal5G1', 'CommSignal2_CommSignal5G1']

def train_script(idx):
    dataset_type = all_datasets[idx]

    ds_train, _ = tfds.load(dataset_type, split="train[:90%]",
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
        data_dir='tfds'
    )
    ds_val, _ = tfds.load(dataset_type, split="train[90%:]",
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
        data_dir='tfds'
    )

    def extract_example(mixture, target):
        return mixture, target

    ds_train = ds_train.map(extract_example, num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.batch(bsz)
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

    ds_val = ds_val.map(extract_example, num_parallel_calls=tf.data.AUTOTUNE)
    ds_val = ds_val.batch(bsz)
    ds_val = ds_val.prefetch(tf.data.AUTOTUNE)


    window_len = 40960
    earlystopping = EarlyStopping(monitor='val_loss', patience=100)
    model_pathname = os.path.join('models', f'{dataset_type}_unet', 'checkpoint')
    checkpoint = ModelCheckpoint(filepath=model_pathname, monitor='val_loss', verbose=0, save_best_only=True, mode='min', save_weights_only=True)

    with mirrored_strategy.scope():
        nn_model = unet.get_unet_model((window_len, 2), k_sz=3, long_k_sz=101, k_neurons=32, lr=0.0003)
        nn_model.fit(ds_train, epochs=2000, batch_size=bsz, shuffle=True, verbose=1, validation_data=ds_val, callbacks=[checkpoint, earlystopping])

if __name__ == '__main__':
    train_script(int(sys.argv[1]))
