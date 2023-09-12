#!/bin/bash

# Create TestSet1Example in the 'dataset' folder
python sampletest_testmixture_generator.py QPSK EMISignal1
python sampletest_testmixture_generator.py QPSK CommSignal2
python sampletest_testmixture_generator.py QPSK CommSignal3
python sampletest_testmixture_generator.py QPSK CommSignal5G1

python sampletest_testmixture_generator.py OFDMQPSK EMISignal1
python sampletest_testmixture_generator.py OFDMQPSK CommSignal2
python sampletest_testmixture_generator.py OFDMQPSK CommSignal3
python sampletest_testmixture_generator.py OFDMQPSK CommSignal5G1
