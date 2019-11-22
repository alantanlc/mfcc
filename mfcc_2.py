#!/usr/bin/env python

import scipy.io.wavfile as w
from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank

# Load waveform
audio_name = '/home/alanwuha/Documents/Projects/datasets/iemocap/IEMOCAP_full_release/Session1/sentences/wav/Ses01F_impro01/Ses01F_impro01_F000.wav'
sample_rate, waveform = w.read(audio_name)

# Compute MFCC
mfcc_feat = mfcc(waveform, sample_rate, preemph=0)
d_mfcc_feat = delta(mfcc_feat, 2)
fbank_feat = logfbank(waveform, sample_rate)

print(fbank_feat[1:3, :])