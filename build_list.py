import os
import sys
import numpy as np
import pandas as pd


def scan_data_dir(data_dir="./data"):
    with open(os.path.join(data_dir, "validation_list.txt")) as f:
        val_set = {line.strip() for line in f if line.strip()}
    with open(os.path.join(data_dir, "testing_list.txt")) as f:
        test_set = {line.strip() for line in f if line.strip()}

    rows = []
    for word in os.listdir(data_dir):
        word_dir = os.path.join(data_dir, word)
        if not os.path.isdir(word_dir) or word.startswith("_"):
            continue
        for fname in os.listdir(word_dir):
            if not fname.endswith(".wav"):
                continue
            rel_path = f"{word}/{fname}"
            if rel_path in val_set:
                split_type = "validation"
            elif rel_path in test_set:
                split_type = "testing"
            else:
                split_type = "training"
            rows.append({"path": rel_path, "word": word, "split_type": split_type})

    return pd.DataFrame(rows)


def build_list(wake_word, data_dir="./data", out_csv="dataset_list.csv"):
    rng = np.random.default_rng()
    df = scan_data_dir(data_dir)

    if wake_word not in df["word"].unique():
        raise ValueError(f"'{wake_word}' not found in {data_dir}")

    df["is_wake_word"] = (df["word"] == wake_word).astype(int)

    selected_parts = []
    for current_split in ["training", "validation", "testing"]:
        split_rows = df[df["split_type"] == current_split]
        wake_word_rows = split_rows[split_rows["is_wake_word"] == 1]
        other_word_rows = split_rows[split_rows["is_wake_word"] == 0]

        num_wake_word = len(wake_word_rows)
        random_indices = rng.permutation(len(other_word_rows))[:num_wake_word]
        selected_other_rows = other_word_rows.iloc[random_indices]

        selected_parts += [wake_word_rows, selected_other_rows]

    dataset_df = pd.concat(selected_parts, ignore_index=True)
    if out_csv:
        dataset_df.to_csv(out_csv, index=False)
    return dataset_df

