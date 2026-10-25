import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


def summarize_mfcc(mfcc_coeffs):
    mfcc_mean = np.mean(mfcc_coeffs, axis=1)
    mfcc_std = np.std(mfcc_coeffs, axis=1, ddof=1)
    return np.hstack([mfcc_mean, mfcc_std])


def run_knn(features_npz="mfcc_features.npz", k_values=(1, 3, 5, 7, 9, 11)):
    data = np.load(features_npz)
    mfcc_coeffs = data["mfcc_coeffs"]
    is_wake_word = data["is_wake_word"]
    split_type = data["split_type"]

    summary_features = summarize_mfcc(mfcc_coeffs)

    train_mask = split_type == "training"
    val_mask = split_type == "validation"
    test_mask = split_type == "testing"

    train_features, train_labels = summary_features[train_mask], is_wake_word[train_mask]
    val_features, val_labels = summary_features[val_mask], is_wake_word[val_mask]
    test_features, test_labels = summary_features[test_mask], is_wake_word[test_mask]

    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features)
    val_features_scaled = scaler.transform(val_features)
    test_features_scaled = scaler.transform(test_features)

    print("\n--- Find the best k to use ---")
    best_k = None
    best_val_acc = -1.0
    best_knn = None

    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(train_features_scaled, train_labels)
        val_preds = knn.predict(val_features_scaled)
        val_acc = accuracy_score(val_labels, val_preds) * 100.0
        print(f"  k = {k:2d}; Validation Accuracy: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_k = k
            best_knn = knn

    print(f"Selected best k = {best_k} (Validation Accuracy: {best_val_acc:.2f}%)")

    test_preds = best_knn.predict(test_features_scaled)
    test_acc = accuracy_score(test_labels, test_preds) * 100.0
    prec, rec, f1, _ = precision_recall_fscore_support(test_labels, test_preds, average="binary")
    cm = confusion_matrix(test_labels, test_preds)

    print(f"\n--- Final Testing Results (best k={best_k}) ---")
    print(f"Accuracy : {test_acc:.2f}%")
    print(f"Precision: {prec * 100.0:.2f}% | Recall: {rec * 100.0:.2f}% | F1: {f1 * 100.0:.2f}%")
    print("Confusion Matrix [[TN, FP], [FN, TP]]:")
    print(cm)

    return best_knn, scaler, best_k
