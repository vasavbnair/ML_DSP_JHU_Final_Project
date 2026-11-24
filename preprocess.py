
import json
import os

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

DATA_DIR = r"C:\Users\vasav\OneDrive\Desktop\Personal\matlab\ML_DS\Final_Project_Data"
CATALOG = "all_sounds.csv"
OUT_DIR = "processed_features"

wake_word = "marvin"
#because there is not enough clips of wake word
unknown_clips_per_wake_word = 4
seed = 50 #randomly chosen
sample_rate = 160000 #sr of data
clip_len_samples = sample_rate
num_mels = 40 #randomly chosen
num_fft = 512
hop = 160 #101 frames per one second clip

classes = ["silence","unknown", wake_word]
silence, unknown, wake = 0,1,2

#splitting up the noise files into a 80 10 and 10 split so that we dont have leakage
#80% of the noise clips go to train, 10 to val and 10 to test
noise_bounds = {"train":(0,.8), "val":(.8,.9), "test":(.9,1)}


#makes the melspectrogram
def make_log_mel(audio):
    mel = librosa.feature.melspectrogram(y=audio,sr=sample_rate,n_fft=num_fft,hop_length=hop,n_mels=num_mels,fmax=sample_rate//2)
    return np.log(mel+1e-6).astype(np.float32)

# makes all audio standardized
def standardize_length(audio):
    if len(audio) < clip_len_samples:
        audio = np.pad(audio(0,clip_len_samples-len(audio)))
    return audio[:clip_len_samples]

def load_clip(path):
    audio, sr = sf.read(os.path.join(DATA_DIR,path),dtype="float32")
    assert sr == SR, f"{path} has a sample rate {sr}, expected{sample_rate}"
    #remove above line later
    return standardize_length(audio)

def load_noise():
    noise_folder = os.path.join(DATA_DIR,"_background_noise_")
    noise_files = []
    for name in sorted(os.listdir(noise_folder)):
        if name.endswith(".wav"):
            audio, sr = sf.read(os.path.join(folder,name), dtype="float32")
            assert sr == sample_rate #remove later
            files.append(audio)
    return files

def load_val_test_lists(filename):
    #going through the validation_list.txt and testing_list.txt to parse the lists
    with open(os.path.join(DATA_DIR,filename)) as f:
        return {line.strip().replace("\\","/") for line in f in line.strip()}

def create_silence_clips(noise_files, splits, rng):
    low, high = noise_bounds[split]
    audio = noise_files[rng.integers(len(noise_files))]
    start, end = int(low*len(audio)), int(high*len(audio))
    clip_start = rng.integers(start,end-clip_len_samples)
    return audio[clip_start:clip_start+clip_len_samples]+rng.uniform(0,1)