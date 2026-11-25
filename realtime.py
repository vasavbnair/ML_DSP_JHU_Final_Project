import json
import sys
import time
import librosa
import numpy as np
import sounddevice as sd
import torch

CNN_MODEL_PATH = "wake_word_data/models/base.pt"
STATS_PATH = "processed_features/stats_of_data.json"
Fs = 16000
N = 16000
WAKE_CLASS = 2 # 0: silence, 1: unknown, 2: marvin
THRESHOLD = 0.60
SILENCE_LEVEL = 0.01
MFCC_N_COEFFS = 20
NUM_FFT = 512


def get_mfcc(audio):
    audio = audio / (np.max(np.abs(audio)) + 1e-8)
    return librosa.feature.mfcc(
        y=audio, sr=Fs,
        n_mfcc=MFCC_N_COEFFS,
        n_fft=NUM_FFT,
        win_length=round(0.03 * Fs),
        hop_length=round(0.01 * Fs))


def get_log_mel(audio, mean, std):
    mel = librosa.feature.melspectrogram(
        y=audio, sr=Fs, n_fft=NUM_FFT, hop_length=round(0.01 * Fs), n_mels=MFCC_N_COEFFS, fmax=Fs // 2
    )
    log_mel = np.log(mel + 1e-6).astype(np.float32)
    return (log_mel - mean) / std


def main():
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else "cnn"
    if mode == "knn":
        print("knn mode TODO")
        return
    if mode != "cnn":
        print("Usage: python realtime.py [cnn|knn]")
        return

    with open(STATS_PATH) as f:
        stats = json.load(f)
    mean, std = stats["mean"], stats["std"]

    model = torch.jit.load(CNN_MODEL_PATH, map_location="cpu")
    model.eval()

    buffer = np.zeros(N, dtype=np.float32)
    step = int(Fs * 0.25)
    last_hit = 0.0

    print("Listening... press Ctrl+C to stop.")
    try:
        with sd.InputStream(samplerate=Fs, channels=1, dtype="float32") as stream:
            while True:
                chunk, _ = stream.read(step)
                buffer = np.roll(buffer, -step)
                buffer[-step:] = chunk[:, 0]

                if np.max(np.abs(buffer)) < SILENCE_LEVEL:
                    continue
                if time.time() - last_hit < 1.0:
                    continue

                feat = get_log_mel(buffer, mean, std)
                x = torch.tensor(feat[np.newaxis, np.newaxis, :, :], dtype=torch.float32)

                with torch.no_grad():
                    out = model(x)
                    probs = torch.softmax(out, dim=1)[0].numpy()

                if probs[WAKE_CLASS] >= THRESHOLD:
                    print(f"\nDetected 'marvin' ({probs[WAKE_CLASS] * 100:.1f}%)")
                    last_hit = time.time()
                else:
                    print(f"silence: {probs[0] * 100:4.1f}% | other: {probs[1] * 100:4.1f}% | marvin: {probs[2] * 100:4.1f}%", end="\r")
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
