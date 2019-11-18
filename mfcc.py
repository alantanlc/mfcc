import numpy as np
import scipy.io.wavfile as w

def periodogram_estimate(dft):
    return np.square(np.abs(dft)) / len(dft)

def discrete_fourier_transform(frame, n_fft=512):
    dft = np.zeros(n_fft).astype(complex)
    for k in range(n_fft):
        for n in range(len(frame)):
            dft[k] += frame[n] * np.exp(-2 * np.pi * 1j * k * n / len(frame))
    return dft

# Steps at a Glance
# 1. Frame the signal into short frames.
# 2. For each frame, calculate the periodogram estimate of the power spectrum.
# 3. Apply the mel filterbank to the power spectra, sum the energy in each filter.
# 4. Take the logarithm of all filterbank energies.
# 5. Take the DCT of the log filterbank energies.
# 6. Keep DCT coefficients 2-13, discard the rest.

# Load waveform
audio_name = '/home/alanwuha/Documents/Projects/datasets/iemocap/IEMOCAP_full_release/Session1/sentences/wav/Ses01F_impro01/Ses01F_impro01_F000.wav'
sample_rate, waveform = w.read(audio_name)

# Convert to floats by dividing 32768.0
waveform = waveform / 32768.0

# Step 1: Frame the signal into short frames.
# Frame the signal into 20-40ms frames.
# If the frame is much shorter, we don't have enough samples to get a reliable spectral estimate.
# If the frame is longer, the signal changes too much throughout the frame.
frame_length = np.int(0.025 * sample_rate)
step_length = np.int(0.01 * sample_rate)
n_frames = np.int(np.ceil((waveform.size - frame_length) / step_length) + 1)
padding_length = frame_length if waveform.size < frame_length else (frame_length + (n_frames - 1) * step_length - waveform.size)
waveform = np.pad(waveform, (0, padding_length))
frames = np.asarray([waveform[i*step_length : i*step_length+frame_length] for i in range(n_frames)])

# Step 2: Calculate the power spectrum of each frame.
# Periodogram estimate identifies the frequencies that are present in the frame.
# This is motivated by the human cochlea which vibrates at different locations/nerves that inform the brain on the presence of certain frequencies.
dfts = np.asarray([discrete_fourier_transform(frame, 512) for frame in frames])
dfts = np.asarray([dft[:257] for dft in dfts])
periodogram_estimates = np.asarray([periodogram_estimate(dft) for dft in dfts])

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