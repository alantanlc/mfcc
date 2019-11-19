import numpy as np
import scipy.io.wavfile as w
import cmath

def periodogram_estimate(dft):
    return np.square(np.abs(dft)) / len(dft)

def discrete_fourier_transform(frame, n_fft=512):
    dft = np.zeros(n_fft).astype(complex)
    for k in range(n_fft):
        for n in range(len(frame)):
            dft[k] += frame[n] * np.exp(-2 * np.pi * 1j * k * n / len(frame))
    return dft

def freq_to_mel_scale(freq):
    """
        The Mel scale relates perceived frequency, or pitch, of a pure tone to its actual measured frequency. Humans are much better at discerning small changes in pitch at low frequencies than they are at high frequencies. Incorporating this scale makes our features match more closely what humans hear.
    """
    return 1125 * np.log(1 + freq / 700)

def mel_scale_to_freq(mel):
    return 700 * (np.exp(mel / 1125) - 1)

def freq_to_nearest_fft_bin(freq, n_fft, sample_rate):
    return np.floor((n_fft + 1) * freq / sample_rate)

def compute_dft_complex(frame, n_fft=512):
    output = []
    for k in range(n_fft):  # For each output element
        s = complex(0)
        for t in range(len(frame)): # For each input element
            angle = 2j * cmath.pi * t * k / len(frame)
            s += frame[t] * cmath.exp(-angle)
        output.append(s)
    return output

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

# Step 1: Frame the signal into 20-40ms frames. 25ms is standard.
# This means the frame length for a 16kHz signal is 0.025 * 16000 = 400 samples.
# Frame step is usually something like 10ms (160 samples), which allows some overlap to the frames.
# The first 400 sample frame starts at sample 0, the next 400 sample frame starts at sample 160 etc until the end of the speech file is reached.
# If the speech file does not divide into an even number of frames, pad it with zeros so that it does.
frame_length = np.int(0.025 * sample_rate)
step_length = np.int(0.01 * sample_rate)
n_frames = np.int(np.ceil((waveform.size - frame_length) / step_length) + 1)
padding_length = frame_length if waveform.size < frame_length else (frame_length + (n_frames - 1) * step_length - waveform.size)
waveform = np.pad(waveform, (0, padding_length))
frames = np.asarray([waveform[i*step_length : i*step_length+frame_length] for i in range(n_frames)])

# Step 2: Calculate the power spectrum of each frame.
# Take the absolute value of the complex fourier transform, and square the result.
# We would generally perform a 512 point FFT and keep only the first 257 coefficients.
dfts = np.asarray([discrete_fourier_transform(frame, 512) for frame in frames])
dfts_257 = np.asarray([dft[:257] for dft in dfts])
periodogram_estimates = np.asarray([periodogram_estimate(dft) for dft in dfts_257])

# Step 3: Compute the Mel-spaced filterbank.
# This is a set of 20-40 (26 is standard) triangular filters that we apply to the periodogram power spectral estimate from step 2. Our filterbank comes in the form of 26 vectors of length 257.
# Each vector is mostly zeros, but is non-zero for a certain section of the spectrum.
# To calculate filterbank energies we multiply each filterbank with the power spectrum, then add up the coefficients.
# Once this is performed, we are left with 26 numbers that give us an indication of how much energy was in each filterbank.
lower_freq, upper_freq = 300, 8000
lower_mel_scale, upper_mel_scale = freq_to_mel_scale(lower_freq), freq_to_mel_scale(upper_freq)
mel_filterbanks = np.linspace(lower_mel_scale, upper_mel_scale, num=12)
freq_filterbanks = np.asarray([mel_scale_to_freq(mel) for mel in mel_filterbanks])
fft_bins = np.asarray([freq_to_nearest_fft_bin(freq, 512, sample_rate) for freq in freq_filterbanks])

x = 0

# Step 4: Take the logarithm of all filterbank energies.
# Humans don't hear loudness on a linear scale. Generally, we need to put 8 times as much energy to double the perceived volume of a sound.
# This compression operation makes our features match more closely to what humans actually hear.
# Logarithm allows us to use cepstral mean subtraction, which is a channel normalisation technique.

# Step 5: Compute the DCT of the log filterbank energies.
# Because our filterbanks are all overlapping, the filterbank energies are quite correlated with each other.
# DCT decorrelates the energies which means diagonal covarience matrices can be used to model the features in e.g. a HMM classifier.
# Only 12 of the 26 DCT coefficients are kept because the higher DCT coefficients represent fast changes in the filterbank energies and it turns out that these fast changes actually degrade ASR performance, so we get a small improvement by dropping them.