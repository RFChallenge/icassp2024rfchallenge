"""Dataset."""

import os
import tensorflow as tf
import tensorflow_datasets as tfds

import glob
import h5py
import numpy as np

_DESCRIPTION = """
RFChallenge at MIT v0.2.0
"""
_CITATION = """
MIT, “RF Challenge - AI Accelerator.” https://rfchallenge.mit.edu/
"""

soi_type = 'QPSK'
interference_sig_type = 'EMISignal1'

class DatasetQpskEmisignal1Mixture(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version('0.2.0')
    RELEASE_NOTES = {
      '0.2.0': 'RFChallenge 2023 release.',
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict({
                'mixture': tfds.features.Tensor(shape=(None, 2), dtype=tf.float32),
                'signal': tfds.features.Tensor(shape=(None, 2), dtype=tf.float32),
            }),
            supervised_keys=('mixture', 'signal'),
            homepage='https://rfchallenge.mit.edu/',
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        path = os.path.join('dataset', f'Dataset_{soi_type}_{interference_sig_type}_Mixture')

        return {
            'train': self._generate_examples(path),
        }

    def _generate_examples(self, path):
        """Yields examples."""
        for f in glob.glob(os.path.join(path, '*.h5')):
            with h5py.File(f,'r') as h5file:
                mixture = np.array(h5file.get('mixture'))
                target = np.array(h5file.get('target'))
                sig_type = h5file.get('sig_type')[()]
                if isinstance(sig_type, bytes):
                    sig_type = sig_type.decode("utf-8") 
            for i in range(mixture.shape[0]):
                yield f'data_{f}_{i}', {
                    'mixture': mixture[i],
                    'signal': target[i],
                }
