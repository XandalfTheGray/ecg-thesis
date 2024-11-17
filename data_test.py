import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from csnecg_data_preprocessing import load_data_numpy, ensure_data_available

def count_class_distribution(X, Y, label_names):
    """Count the number of samples in each class"""
    # Get indices where each class is positive (1)
    class_indices = np.where(Y == 1)
    
    # Count occurrences of each class
    counts = Counter(class_indices[1])  # [1] gives us the class indices
    
    print("\nClass Distribution:")
    for idx in sorted(counts.keys()):
        print(f"{label_names[idx]}: {counts[idx]} samples")
    return counts

def visualize_samples_by_class(X, Y, label_names, samples_per_class=10):
    """Visualize samples for each class in a grid layout"""
    num_classes = len(label_names)
    
    # Get indices for each class
    class_indices = {i: np.where(Y[:, i] == 1)[0] for i in range(num_classes)}
    
    # Create a large figure
    fig = plt.figure(figsize=(20, 4*num_classes))
    plt.suptitle('Examples for Each Class', fontsize=16)
    
    # Plot samples for each class
    for class_idx in range(num_classes):
        indices = class_indices[class_idx]
        if len(indices) == 0:
            continue
            
        # Take up to samples_per_class random samples
        sample_indices = np.random.choice(
            indices,
            min(samples_per_class, len(indices)),
            replace=False
        )
        
        for j, idx in enumerate(sample_indices):
            ax = plt.subplot(num_classes, samples_per_class, 
                           class_idx*samples_per_class + j + 1)
            segment = X[idx]
            
            # Plot all leads with different colors
            for lead in range(segment.shape[1]):
                ax.plot(segment[:, lead], alpha=0.5, linewidth=0.5)
            
            if j == 0:  # Only label the first plot in each row
                ax.set_ylabel(f'{label_names[class_idx]}', rotation=0, labelpad=40)
            
            ax.set_xticks([])
            ax.set_yticks([])
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def inspect_data(X, Y, label_names):
    """Detailed inspection of data contents"""
    print("\nInspecting data:")
    print(f"\nData shapes:")
    print(f" - X shape: {X.shape}")
    print(f" - Y shape: {Y.shape}")
    
    print(f"\nData types:")
    print(f" - X dtype: {X.dtype}")
    print(f" - Y dtype: {Y.dtype}")
    
    print(f"\nNumber of unique labels: {len(label_names)}")
    print(f"Label names: {label_names}")
    
    # Print some basic statistics
    print("\nData statistics:")
    print(f" - Number of samples: {len(X)}")
    print(f" - Time steps per sample: {X.shape[1]}")
    print(f" - Number of leads: {X.shape[2]}")
    
    # Count multilabel occurrences
    label_counts = Y.sum(axis=1)
    print(f"\nMultilabel statistics:")
    print(f" - Average labels per sample: {label_counts.mean():.2f}")
    print(f" - Max labels per sample: {label_counts.max()}")
    print(f" - Min labels per sample: {label_counts.min()}")

def main():
    # Set up paths
    base_path = '/content/drive/MyDrive/'  # Google Drive base path
    drive_data_dir = os.path.join(base_path, 'csnecg_preprocessed_data')
    local_data_dir = 'csnecg_preprocessed_data'
    peaks_per_signal = 10  # Adjust as needed
    
    # Ensure data is available locally
    ensure_data_available(local_data_dir, drive_data_dir, peaks_per_signal)
    
    # Load the data
    X, Y, label_names = load_data_numpy(local_data_dir, peaks_per_signal)
    
    # Inspect the data
    inspect_data(X, Y, label_names)
    
    # Count and display class distribution
    counts = count_class_distribution(X, Y, label_names)
    
    # Visualize samples
    visualize_samples_by_class(X, Y, label_names, samples_per_class=10)

if __name__ == '__main__':
    main() 