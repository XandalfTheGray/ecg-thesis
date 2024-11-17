import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Import your modules
from models import build_cnn
from csnecg_data_preprocessing import prepare_csnecg_data

def inspect_hdf5_file(hdf5_file_path):
    """Detailed inspection of HDF5 file contents"""
    print(f"\nInspecting HDF5 file: {hdf5_file_path}")
    with h5py.File(hdf5_file_path, 'r') as f:
        # Print basic structure
        print("\nDatasets in the HDF5 file:")
        for key in f.keys():
            print(f" - {key}: {f[key].shape}")
        
        # Examine segments
        segments = f['segments']
        print(f"\nSegments shape: {segments.shape}")
        print(f"Segments dtype: {segments.dtype}")
        
        # Examine labels
        labels = f['labels']
        print(f"\nLabels shape: {labels.shape}")
        print(f"Labels dtype: {labels.dtype}")
        
        # Examine label names
        label_names = [name.decode('utf-8') for name in f['label_names'][()]]
        print(f"\nNumber of unique labels: {len(label_names)}")
        print(f"Label names: {label_names}")
        
        # Check first few samples
        print("\nFirst 5 samples:")
        for i in range(min(5, len(labels))):
            label_indices = labels[i]
            label_list = [label_names[idx] for idx in label_indices]
            print(f"Sample {i}: Labels = {label_list}")

def visualize_samples(hdf5_file_path, num_samples=5):
    print(f"\nVisualizing {num_samples} samples from HDF5 file: {hdf5_file_path}")
    with h5py.File(hdf5_file_path, 'r') as hdf5_file:
        segments = hdf5_file['segments']
        labels = hdf5_file['labels']
        label_names = [name.decode('utf-8') for name in hdf5_file['label_names'][()]]
        
        num_samples = min(num_samples, segments.shape[0])
        for i in range(num_samples):
            segment = segments[i]
            label_indices = labels[i]
            label_list = [label_names[idx] for idx in label_indices]
            
            plt.figure(figsize=(12, 4))
            for lead in range(segment.shape[1]):
                plt.plot(segment[:, lead], label=f'Lead {lead+1}')
            plt.title(f'Sample {i+1} - Labels: {", ".join(label_list)}')
            plt.xlabel('Time Steps')
            plt.ylabel('Amplitude')
            plt.legend(loc='upper right', ncol=4, fontsize='small')
            plt.tight_layout()
            plt.show()

def test_data_loading(base_path, batch_size=128, max_samples=1000):
    print("\nTesting data loading using 'prepare_csnecg_data' function...")
    try:
        train_dataset, valid_dataset, test_dataset, num_classes, label_names, Num2Label = prepare_csnecg_data(
            base_path=base_path, batch_size=batch_size, hdf5_file_path='csnecg_segments_10peaks.hdf5', max_samples=max_samples
        )
        print(f"Number of classes: {num_classes}")
        print(f"Label names: {label_names}")
    except Exception as e:
        print(f"Error during data loading: {e}")
        return

    # Iterate over just one batch
    print("\nTesting with a single batch...")
    try:
        batch = next(iter(train_dataset))
        X_batch, y_batch = batch
        print(f"Batch shapes:")
        print(f" - X_batch shape: {X_batch.shape}")
        print(f" - y_batch shape: {y_batch.shape}")
    except Exception as e:
        print(f"Error processing batch: {e}")
        return

def test_training_run(base_path, batch_size=128, max_samples=1000):
    print("\nTesting a short training run...")
    try:
        # Load data
        train_dataset, valid_dataset, test_dataset, num_classes, label_names, Num2Label = prepare_csnecg_data(
            base_path=base_path, 
            batch_size=batch_size,
            hdf5_file_path='csnecg_segments_10peaks.hdf5'  # Specify correct file
        )

        # Take only a very small subset for testing
        train_subset = train_dataset.take(1)  # Just take 1 batch
        valid_subset = valid_dataset.take(1)  # Just take 1 batch
        test_subset = test_dataset.take(1)    # Just take 1 batch

        # Build a simple CNN model
        model = build_cnn(
            input_shape=(300, 12),  # Fixed shape from HDF5
            num_classes=num_classes,
            activation='sigmoid'
        )

        model.compile(
            loss='binary_crossentropy',
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            metrics=['accuracy']
        )

        # Train for just one epoch on the small subset
        history = model.fit(
            train_subset,
            epochs=1,
            validation_data=valid_subset
        )

        print("Test training completed successfully!")
        
    except Exception as e:
        print(f"Error during training test: {e}")

def main():
    # Update paths to match your environment
    hdf5_file_path = 'csnecg_segments_10peaks.hdf5'
    base_path = '.'  # Current directory where the csnecg_preprocessed_data folder is located
    batch_size = 128
    max_samples = 1000  # Limit to 1000 samples for testing

    print("Testing HDF5 file...")
    inspect_hdf5_file(hdf5_file_path)
    
    print("\nTesting sample visualization...")
    visualize_samples(hdf5_file_path, num_samples=3)
    
    print("\nTesting data loading...")
    test_data_loading(base_path, batch_size=batch_size, max_samples=max_samples)
    
    print("\nTesting training pipeline...")
    test_training_run(base_path, batch_size=batch_size, max_samples=max_samples)

if __name__ == '__main__':
    main()