import os
import pandas as pd
import soundfile as sf

data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

rows = [] # rows for the dataframe
for word in sorted(os.listdir(data_dir)):
    word_dir = os.path.join(data_dir,word) # gets the word dir
    if not os.path.isdir(word_dir) or word.startswith("_"): # dont want background noise or non words
        continue
    for fname in os.listdir(word_dir):
        if fname.endswith(".wav"):
            rows.append({"path": os.path.join(word, fname), "label":word, "speaker": fname.split("_nohash_")[0],})
df = pd.DataFrame(rows)
df.to_csv("all_sounds.csv", index=False)
print(df["label"].value_counts())
print("Speakers:", df["speaker"].nunique())
for p in df.sample(5, random_state=0)["path"]:
    audio,sr = sf.read(os.path.join(data_dir, p))
    print(p, "sr =", sr, "duration = ", round(len(audio)/sr ,3), "s")