import torchaudio
import torch.nn.functional as F
import math

# Steps at a Glance
# 1. Frame the signal into short frames.
# 2. For each frame, calculate the periodogram estimate of the power spectrum.
# 3. Apply the mel filterbank to the power spectra, sum the energy in each filter.
# 4. Take the logarithm of all filterbank energies.
# 5. Take the DCT of the log filterbank energies.
# 6. Keep DCT coefficients 2-13, discard the rest.

# Load waveform
audio_name = '/home/alanwuha/Documents/Projects/datasets/iemocap/IEMOCAP_full_release/Session1/sentences/wav/Ses01F_impro01/Ses01F_impro01_F000.wav'
waveform, sample_rate = torchaudio.load(audio_name)

# Step 1: Frame the signal into short frames.
# Frame the signal into 20-40ms frames.
# If the frame is much shorter, we don't have enough samples to get a reliable spectral estimate.
# If the frame is longer, the signal changes too much throughout the frame.
frame_length = 0.025 * sample_rate
step_length = 0.01 * sample_rate
n_frames = math.ceil((waveform.shape[0] - frame_length) / step_length) + 1
padding_length = int(frame_length + (n_frames - 1) * step_length - waveform.shape[0])
waveform = F.pad(waveform, pad=(0, padding_length))

# Step 2: Calculate the power spectrum of each frame.
# Periodogram estimate identifies the frequencies that are present in the frame.
# This is motivated by the human cochlea which vibrates at different locations/nerves that inform the brain on the presence of certain frequencies.

# Step 3: Apply the mel filterbank to the power spectra, sum the energy in each filter.
# Cochlea can not discern the difference between two closely spaced frequencies. This effect becomes more pronounced as the frequencies increase.
# For this reason, we take clumps of periodogram bins and sum them up to get an idea of how much energy exists in various frequency regions.
# This is performed by our Mel filterbank: the first filter is very narrow and gives an indication of how much energy exists near 0 Hertz. As the frequencies get higher, our filters get wider as we become less concerned about variations.
# The Mel scale tells us exactly how to space our filterbanks and how wide to make them.

# Step 4: Take the logarithm of all filterbank energies.
# Humans don't hear loudness on a linear scale. Generally, we need to put 8 times as much energy to double the perceived volume of a sound.
# This compression operation makes our features match more closely to what humans actually hear.
# Logarithm allows us to use cepstral mean subtraction, which is a channel normalisation technique.

# Step 5: Compute the DCT of the log filterbank energies.
# Because our filterbanks are all overlapping, the filterbank energies are quite correlated with each other.
# DCT decorrelates the energies which means diagonal covarience matrices can be used to model the features in e.g. a HMM classifier.
# Only 12 of the 26 DCT coefficients are kept because the higher DCT coefficients represent fast changes in the filterbank energies and it turns out that these fast changes actually degrade ASR performance, so we get a small improvement by dropping them.