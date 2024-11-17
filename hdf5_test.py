import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from collections import Counter

# Import your modules
from models import build_cnn
from csnecg_data_preprocessing import prepare_csnecg_data, CHUNK_SIZE, BATCH_SIZE

def count_class_distribution(hdf5_file_path):
    """Count the number of samples in each class"""
    with h5py.File(hdf5_file_path, 'r') as f:
        labels = f['labels'][:]
        label_names = [name.decode('utf-8') for name in f['label_names'][()]]
        
        # Flatten and count all labels
        all_labels = [item for sublist in labels for item in sublist]
        counts = Counter(all_labels)
        
        print("\nClass Distribution:")
        for idx, count in sorted(counts.items()):
            print(f"{label_names[idx]}: {count} samples")
        return counts, label_names

def visualize_samples_by_class(hdf5_file_path, samples_per_class=10):
    """Visualize samples for each class in a grid layout"""
    with h5py.File(hdf5_file_path, 'r') as f:
        segments = f['segments']
        labels = f['labels']
        label_names = [name.decode('utf-8') for name in f['label_names'][()]]
        
        # Count classes and get indices for each class
        class_indices = {i: [] for i in range(len(label_names))}
        for idx, label_list in enumerate(labels):
            for label in label_list:
                class_indices[label].append(idx)
        
        # Create a large figure
        num_classes = len(label_names)
        fig = plt.figure(figsize=(20, 4*num_classes))
        plt.suptitle('10 Examples for Each Class', fontsize=16)
        
        # Plot samples for each class
        for class_idx in range(num_classes):
            available_samples = class_indices[class_idx]
            if not available_samples:
                continue
                
            # Take up to 10 random samples
            sample_indices = np.random.choice(
                available_samples, 
                min(samples_per_class, len(available_samples)), 
                replace=False
            )
            
            for j, sample_idx in enumerate(sample_indices):
                ax = plt.subplot(num_classes, samples_per_class, 
                               class_idx*samples_per_class + j + 1)
                segment = segments[sample_idx]
                
                # Plot all leads with different colors
                for lead in range(segment.shape[1]):
                    ax.plot(segment[:, lead], alpha=0.5, linewidth=0.5)
                
                if j == 0:  # Only label the first plot in each row
                    ax.set_ylabel(f'{label_names[class_idx]}', rotation=0, labelpad=40)
                
                ax.set_xticks([])
                ax.set_yticks([])
                
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout
        plt.show()

def inspect_hdf5_file(hdf5_file_path):
    """Detailed inspection of HDF5 file contents"""
    print(f"\nInspecting HDF5 file: {hdf5_file_path}")
    with h5py.File(hdf5_file_path, 'r') as f:
        print("\nDatasets in the HDF5 file:")
        for key in f.keys():
            print(f" - {key}: {f[key].shape}")
        
        segments = f['segments']
        print(f"\nSegments shape: {segments.shape}")
        print(f"Segments dtype: {segments.dtype}")
        print(f"Chunks: {segments.chunks}")
        
        labels = f['labels']
        print(f"\nLabels shape: {labels.shape}")
        print(f"Labels dtype: {labels.dtype}")
        print(f"Chunks: {labels.chunks}")
        
        label_names = [name.decode('utf-8') for name in f['label_names'][()]]
        print(f"\nNumber of unique labels: {len(label_names)}")
        print(f"Label names: {label_names}")

def test_data_loading(base_path, batch_size=BATCH_SIZE, max_samples=1000):
    print("\nTesting data loading using 'prepare_csnecg_data' function...")
    try:
        train_dataset, valid_dataset, test_dataset, num_classes, label_names, Num2Label = prepare_csnecg_data(
            base_path=base_path, 
            batch_size=batch_size,
            hdf5_file_path='csnecg_segments_1peaks.hdf5',
            max_samples=max_samples
        )
        print(f"Number of classes: {num_classes}")
        print(f"Label names: {label_names}")
        
        # Test batch shapes
        batch = next(iter(train_dataset))
        X_batch, y_batch = batch
        print(f"\nBatch shapes:")
        print(f" - X_batch shape: {X_batch.shape}")
        print(f" - y_batch shape: {y_batch.shape}")
        
    except Exception as e:
        print(f"Error during data loading: {e}")
        raise

def test_minimal_training(base_path, batch_size=BATCH_SIZE):
    """Run a minimal training test with just a few batches."""
    print("\nTesting minimal training capability...")
    try:
        # Load minimal data
        train_dataset, valid_dataset, _, num_classes, label_names, _ = prepare_csnecg_data(
            base_path=base_path, 
            batch_size=batch_size,
            hdf5_file_path='csnecg_segments_1peaks.hdf5'
        )
        
        # Take just 2 batches for training and 1 for validation
        train_subset = train_dataset.take(2)
        valid_subset = valid_dataset.take(1)
        
        # Build a simple model
        model = build_cnn(
            input_shape=(300, 12),
            num_classes=num_classes,
            activation='sigmoid'
        )
        
        model.compile(
            loss='binary_crossentropy',
            optimizer=tf.keras.optimizers.Adam(1e-4),
            metrics=['accuracy']
        )
        
        # Train for just 2 epochs
        print("\nRunning minimal training (2 epochs, 2 batches)...")
        history = model.fit(
            train_subset,
            epochs=2,
            validation_data=valid_subset,
            verbose=1
        )
        
        print("\nMinimal training completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error during minimal training test: {e}")
        return False

def main():
    hdf5_file_path = 'csnecg_segments_1peaks.hdf5'
    base_path = '.'
    
    # Inspect file structure
    inspect_hdf5_file(hdf5_file_path)
    
    # Count and display class distribution
    counts, label_names = count_class_distribution(hdf5_file_path)
    
    # Visualize samples
    visualize_samples_by_class(hdf5_file_path, samples_per_class=10)
    
    # Test data loading
    test_data_loading(base_path, batch_size=BATCH_SIZE)
    
    # Test minimal training
    test_minimal_training(base_path, batch_size=BATCH_SIZE)

if __name__ == '__main__':
    main()