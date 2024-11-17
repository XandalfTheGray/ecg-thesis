# csnecg_train_cnn.py

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import tensorflow as tf
from tensorflow import keras
import argparse
import time
import h5py

# Import your modules
from models import build_cnn
from evaluation import (
    evaluate_multilabel_model,
    CustomProgressBar,
    TimingCallback,
    log_timing_info
)
from csnecg_data_preprocessing import prepare_csnecg_data

def main(time_steps, batch_size):
    # Enable mixed precision
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    
    # Set up base path for OUTPUTS on Google Drive
    base_path = '/content/drive/MyDrive/'
    base_output_dir = os.path.join(base_path, 'csnecg_output_plots')
    dataset_name = 'csnecg'
    model_type = 'cnn'
    output_dir = os.path.join(base_output_dir, f"{dataset_name}_{model_type}_{time_steps}steps_{batch_size}batch")
    os.makedirs(output_dir, exist_ok=True)
    
    learning_rate = 1e-3

    # Define model parameters
    model_params = {
        'l2_reg': 0.001,
        'filters': [32, 64, 128],
        'kernel_sizes': [5, 5, 5],
        'dropout_rates': [0.3, 0.3, 0.3, 0.3],
    }

    # Prepare data with optimized memory usage
    peaks_per_signal = 1
    (
        train_dataset,
        valid_dataset,
        test_dataset,
        num_classes,
        label_names,
        Num2Label,
        steps_per_epoch,
        validation_steps,
        test_steps
    ) = prepare_csnecg_data(
        base_path='.',
        batch_size=batch_size,
        hdf5_file_path=f'csnecg_segments_{peaks_per_signal}peaks.hdf5'
    )
    
    # Enable prefetching and parallel processing
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
    valid_dataset = valid_dataset.prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)
    
    # Build model with mixed precision
    model = build_cnn(
        input_shape=(time_steps, 12),
        num_classes=num_classes,
        activation='sigmoid',
        **model_params
    )
    
    # Use AMP-optimized optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
    
    # Compile the model
    model.compile(
        loss='binary_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )

    # Create timing callback
    timing_callback = TimingCallback()
    
    # Add timing callback to the callbacks list
    callbacks = [
        CustomProgressBar(),
        timing_callback,
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(output_dir, 'best_model.keras'),
            monitor='val_loss', save_best_only=True, verbose=1
        )
    ]

    # Add memory tracking callback
    class MemoryCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            memory = tf.config.experimental.get_memory_info('GPU:0')
            print(f"\nGPU Memory: {memory['peak'] / 1e9:.2f} GB")
    
    callbacks.append(MemoryCallback())
    
    # Train the model
    print("\nStarting model training...")
    training_start = time.time()
    
    history = model.fit(
        train_dataset,
        epochs=30,
        steps_per_epoch=steps_per_epoch,
        validation_data=valid_dataset,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1
    )
    
    training_end = time.time()
    total_training_time = training_end - training_start
    print(f"\nTotal training time: {total_training_time:.2f} seconds")
    print(f"Average time per epoch: {np.mean(timing_callback.times):.2f} seconds")

    # Generate predictions for test set
    print("\nGenerating predictions for test set...")
    test_timing = {}
    start_time = time.time()
    y_pred = model.predict(
        test_dataset,
        steps=test_steps,
        verbose=1
    )
    end_time = time.time()
    test_timing['Test'] = end_time - start_time
    print(f"Test set prediction time: {test_timing['Test']:.2f} seconds")

    y_pred_classes = (y_pred > 0.5).astype(int)
    y_true = np.concatenate([y for x, y in test_dataset], axis=0).astype(int)

    # Log timing information
    model_info = {
        'model_type': model_type,
        'dataset': dataset_name,
        'parameters': model_params
    }
    log_timing_info(timing_callback, model_info, output_dir)

    # Add test timing information to the log
    with open(os.path.join(output_dir, 'test_timing.txt'), 'w') as f:
        f.write("Prediction/Evaluation Timing:\n")
        for name, time_taken in test_timing.items():
            f.write(f"{name} Set Prediction Time: {time_taken:.2f} seconds\n")

    # Evaluate and visualize
    evaluate_multilabel_model(
        y_true=y_true,
        y_pred=y_pred_classes,
        y_scores=y_pred,
        label_names=label_names,
        output_dir=output_dir,
        history=history
    )

    print(f"\nTraining completed. Results saved in {output_dir}")

    # Save model parameters to a text file
    with open(os.path.join(output_dir, 'model_params.txt'), 'w') as f:
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Model Type: {model_type}\n")
        f.write(f"Time Steps: {time_steps}\n")
        f.write(f"Batch Size: {batch_size}\n")
        for key, value in model_params.items():
            f.write(f"  {key}: {value}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a CNN model on the preprocessed CSN ECG dataset.')
    parser.add_argument('--time_steps', type=int, default=300, 
                        help='Number of time steps in the preprocessed data (default: 300).')
    parser.add_argument('--batch_size', type=int, default=128, 
                        help='Batch size for training (default: 128)')
    args, unknown = parser.parse_known_args()
    main(args.time_steps, args.batch_size)