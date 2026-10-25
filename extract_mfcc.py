import os
import sys
import librosa
import numpy as np
import pandas as pd
import soundfile as sf

Fs = 16000
N = 16000
MIX_NOISE = True
SNR_DB = 15.0


def pad_or_trim(audio, length=N):
    if len(audio) >= length:
        return audio[:length]
    return np.concatenate([audio, np.zeros(length - len(audio), dtype=audio.dtype)])


def normalize(audio):
    return audio / (np.max(np.abs(audio)) + 1e-8)


def mix_noise(audio, noise, snr_db=SNR_DB):
    p_sig = np.mean(audio ** 2)
    p_noise = np.mean(noise ** 2) + 1e-12
    scale = np.sqrt(p_sig / (p_noise * (10 ** (snr_db / 10.0))))
    return audio + scale * noise


def load_noise_clips(data_dir="./data"):
    noise_dir = os.path.join(data_dir, "_background_noise_")
    noise_clips = []
    for fname in os.listdir(noise_dir):
        if fname.endswith(".wav"):
            noise_audio, _ = sf.read(os.path.join(noise_dir, fname))
            noise_clips.append(noise_audio)
    return noise_clips


def load_and_condition(row, data_dir="./data", noise_clips=None, rng=None):
    audio, _ = sf.read(os.path.join(data_dir, row["path"]))
    audio = pad_or_trim(audio, N)
    signals = [normalize(audio)]

    if MIX_NOISE and row["split_type"] == "training" and noise_clips:
        if rng is None:
            rng = np.random.default_rng()
        for noise_audio in noise_clips:
            if len(noise_audio) <= N:
                noise_slice = pad_or_trim(noise_audio, N)
            else:
                start = int(rng.integers(0, len(noise_audio) - N))
                noise_slice = noise_audio[start : start + N]
            mixed = mix_noise(audio, noise_slice, SNR_DB)
            signals.append(normalize(mixed))

    return signals


def compute_mfcc(audio, fs=Fs, num_coeffs=13):
    return librosa.feature.mfcc(
        y=audio,
        sr=fs,
        n_mfcc=num_coeffs,
        n_fft=512,
        win_length=round(0.03 * fs),
        hop_length=round(0.01 * fs),
    ).T


def extract_dataset_mfcc(
    list_csv="dataset_list.csv",
    data_dir="./data",
    num_coeffs=13,
    out_npz="mfcc_features.npz",
):
    df = pd.read_csv(list_csv)
    noise_clips = load_noise_clips(data_dir) if MIX_NOISE else None
    rng = np.random.default_rng()

    coeff_list, wake_word_list, split_list = [], [], []

    for i, row in df.iterrows():
        signals = load_and_condition(row, data_dir=data_dir, noise_clips=noise_clips, rng=rng)
        for sig in signals:
            coeffs = compute_mfcc(sig, Fs, num_coeffs=num_coeffs)
            coeff_list.append(coeffs.astype(np.float32))
            wake_word_list.append(int(row["is_wake_word"]))
            split_list.append(row["split_type"])

        if (i + 1) % 500 == 0 or (i + 1) == len(df):
            print(f"Processed {i + 1}/{len(df)} audio files...")

    mfcc_coeffs = np.stack(coeff_list, axis=0)
    is_wake_word = np.array(wake_word_list, dtype=np.int64)
    split_type = np.array(split_list)

    if out_npz:
        np.savez_compressed(
            out_npz,
            mfcc_coeffs=mfcc_coeffs,
            is_wake_word=is_wake_word,
            split_type=split_type,
        )
    return mfcc_coeffs, is_wake_word, split_type
