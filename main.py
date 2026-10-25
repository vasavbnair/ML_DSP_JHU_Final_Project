from build_list import build_list
from extract_mfcc import extract_dataset_mfcc
from knn import run_knn

WAKE_WORD = "marvin"
DATA_DIR = "./data"
NUM_COEFFS = 13
K_VALUES = (1, 3, 5, 7, 9, 11)


def main():
    dataset_df = build_list(
        wake_word=WAKE_WORD,
        data_dir=DATA_DIR,
        out_csv="dataset_list.csv",
    )
    print(f"Wake word: {WAKE_WORD} | Base files in dataset_list.csv: {len(dataset_df)}")
    print(dataset_df.groupby(["split_type", "is_wake_word"]).size().unstack())

    mfcc_coeffs, is_wake_word, split_type = extract_dataset_mfcc(
        list_csv="dataset_list.csv",
        data_dir=DATA_DIR,
        num_coeffs=NUM_COEFFS,
        out_npz="mfcc_features.npz",
    )
    print(f"Saved mfcc_features.npz -> mfcc_coeffs shape: {mfcc_coeffs.shape}")

    run_knn(features_npz="mfcc_features.npz", k_values=K_VALUES)


if __name__ == "__main__":
    main()
