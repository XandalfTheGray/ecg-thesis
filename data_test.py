import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def load_data_numpy(data_dir):
    X = np.load(os.path.join(data_dir, 'X.npy'))
    Y = np.load(os.path.join(data_dir, 'Y.npy'))
    label_names = np.load(os.path.join(data_dir, 'label_names.npy'))
    return X, Y, label_names

def count_class_distribution(Y, label_names):
    """Count the number of samples in each class"""
    all_labels = [np.where(y == 1)[0] for y in Y]
    flattened_labels = [label for labels in all_labels for label in labels]
    counts = Counter(flattened_labels)
    print("\nClass Distribution:")
    for idx in sorted(counts.keys()):
        print(f"{label_names[idx]}: {counts[idx]} samples")
    return counts

def visualize_samples_by_class(X, Y, label_names, samples_per_class=10):
    """Visualize samples for each class in a grid layout"""
    num_classes = len(label_names)
    class_indices = {i: [] for i in range(num_classes)}
    for idx, y in enumerate(Y):
        active_labels = np.where(y == 1)[0]
        for label in active_labels:
            class_indices[label].append(idx)

    fig = plt.figure(figsize=(20, 4*num_classes))
    plt.suptitle('Examples for Each Class', fontsize=16)

    for class_idx in range(num_classes):
        indices = class_indices[class_idx]
        if not indices:
            continue
        sample_indices = np.random.choice(
            indices,
            min(samples_per_class, len(indices)),
            replace=False
        )
        for j, idx in enumerate(sample_indices):
            ax = plt.subplot(num_classes, samples_per_class, class_idx*samples_per_class + j + 1)
            segment = X[idx]

            for lead in range(segment.shape[1]):
                ax.plot(segment[:, lead], alpha=0.5, linewidth=0.5)

            if j == 0:
                ax.set_ylabel(f'{label_names[class_idx]}', rotation=0, labelpad=40)
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def main():
    data_dir = 'csnecg_preprocessed_data'
    X, Y, label_names = load_data_numpy(data_dir)

    # Count class distribution
    counts = count_class_distribution(Y, label_names)

    # Visualize samples
    visualize_samples_by_class(X, Y, label_names, samples_per_class=10)

if __name__ == '__main__':
    main() 