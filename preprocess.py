
import json
import os

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

DATA_DIR = r"C:\Users\vasav\OneDrive\Desktop\Personal\matlab\ML_DSP\Final_Project_Data"
all_sounds = "all_sounds.csv"
OUT_DIR = "processed_features"

wake_word = "marvin"
#because there is not enough clips of wake word
unknown_clips_per_wake_word = 4
seed = 50 #randomly chosen
sample_rate = 16000 #sr of data
clip_len_samples = sample_rate
num_mels = 40 #randomly chosen
num_fft = 512
hop = 160 #101 frames per one second clip

classes = ["silence","unknown", wake_word]
silence_val, unknown_val, wake_val = 0,1,2

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
        audio = np.pad(audio,(0,clip_len_samples-len(audio)))
    return audio[:clip_len_samples]

def load_clip(path):
    audio, sr = sf.read(os.path.join(DATA_DIR,path),dtype="float32")
    assert sr == sample_rate, f"{path} has a sample rate {sr}, expected{sample_rate}"
    #remove above line later
    return standardize_length(audio)

def load_noise():
    noise_folder = os.path.join(DATA_DIR,"_background_noise_")
    noise_files = []
    for name in sorted(os.listdir(noise_folder)):
        if name.endswith(".wav"):
            audio, sr = sf.read(os.path.join(noise_folder,name), dtype="float32")
            assert sr == sample_rate #remove later
            noise_files.append(audio)
    return noise_files

def load_val_test_lists(filename):
    #going through the vpealidation_list.txt and testing_list.txt to parse the lists
    with open(os.path.join(DATA_DIR,filename)) as f:
        return {line.strip().replace("\\","/") for line in f if line.strip()}

def create_silence_clips(noise_files, splits, rng):
    low, high = noise_bounds[splits]
    audio = noise_files[rng.integers(len(noise_files))]
    start, end = int(low*len(audio)), int(high*len(audio))
    clip_start = rng.integers(start,end-clip_len_samples)
    return audio[clip_start:clip_start+clip_len_samples]*rng.uniform(0,1)



def main():
    rng = np.random.default_rng(seed)
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(all_sounds)

    # the csv has path, label, speaker
    df["path"] = df["path"].str.replace("\\","/",regex=False) # replace \\ w/ /
    val_set = load_val_test_lists("validation_list.txt")
    test_set = load_val_test_lists("testing_list.txt")
    df["split"] = "train" #initializes everything as training
    df.loc[df["path"].isin(val_set),"split"] = "val" #if the path in split is in val_set rename train to val
    df.loc[df["path"].isin(test_set),"split"] = "test" #if the path in split is in val_set rename train to val

    noise_files = load_noise()
    my_splits = {}
    for split in ["train","val", "test"]:
        sub = df[df["split"] == split]
        wake = sub[sub["label"]==wake_word] #getting all wake words
        other = sub[sub["label"]!=wake_word]
        num_wake = len(wake)
        num_unknown = min(len(other), unknown_clips_per_wake_word *num_wake) # makes teh num of unknown comparable to wake
        unknown = other.sample(n=num_unknown,random_state=seed)
        print(f"[{split}] wake={num_wake}  unknown={num_unknown}  silence={num_wake}")
        X,Y, words = [],[],[]
        # x is logmels, y is 0,1,2 words are all the words we will use
        # tqdm is a progress bar
        for path, word in tqdm(zip(wake["path"],wake["label"]),total=num_wake,desc=f"{split} wake"):
            X.append(make_log_mel(load_clip(path)))
            Y.append(wake_val)
            words.append(word)
        for path, word in tqdm(zip(unknown["path"],unknown["label"]),total=num_unknown,desc=f"{split} unknown"):
            X.append(make_log_mel(load_clip(path)))
            Y.append(unknown_val)
            words.append(word)
        for i in range(num_wake): #for silence
            X.append(make_log_mel(create_silence_clips(noise_files,split,rng)))
            Y.append(silence_val)
            words.append("silence")
        X = np.stack(X)
        Y = np.array(Y,dtype=np.int64)
        words = np.array(words)

        order = rng.permutation(len(Y))
        my_splits[split] = (X[order],Y[order],words[order])

    # need to normalize using train
    mean = float(my_splits["train"][0].mean())
    std = float(my_splits["train"][0].std())
    with open(os.path.join(OUT_DIR,"stats_of_data.json"),"w") as f:
        json.dump({"mean":mean,"std":std,"classes": classes, "sr":sample_rate, "n_fft": num_fft, "hop":hop,"n_mels":num_mels},f,indent=2)
    print(f"Normalization: mean={mean:.4f}  std={std:.4f}")
    #need to save everything
    for split, (X, Y, words) in my_splits.items():
        X = ((X - mean) / std).astype(np.float32)
        out = os.path.join(OUT_DIR, f"{split}.npz")
        np.savez_compressed(out, X=X, y=Y, words=words)
        counts = {classes[i]: int((Y == i).sum()) for i in range(len(classes))}
        print(f"Saved {out}: X{X.shape}  {counts}  "
              f"({os.path.getsize(out) / 1e6:.1f} MB)")






if __name__ == "__main__":
    main()