# csnecg_train_resnet.py

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

# Add the directory containing your modules to the Python path
sys.path.append('/content/ecg-thesis')

# Import your modules
from models import build_resnet18_1d, build_resnet34_1d, build_resnet50_1d
from evaluation import (
    evaluate_multilabel_model,
    CustomProgressBar,
    TimingCallback,
    log_timing_info
)
from csnecg_data_preprocessing import prepare_csnecg_data

def main(time_steps, batch_size, resnet_type):
    # Set up base path for OUTPUTS on Google Drive
    base_path = '/content/drive/MyDrive/'
    base_output_dir = os.path.join(base_path, 'csnecg_output_plots')
    dataset_name = 'csnecg'
    model_type = resnet_type
    output_dir = os.path.join(base_output_dir, f"{dataset_name}_{model_type}_{time_steps}steps_{batch_size}batch")
    os.makedirs(output_dir, exist_ok=True)

    learning_rate = 1e-3

    # Define model parameters
    model_params = {
        'l2_reg': 0.001,
    }

    # Prepare data - using current directory for HDF5 file
    peaks_per_signal = 1  # Match the value used in preprocessing
    (
        train_dataset,
        valid_dataset,
        test_dataset,
        num_classes,
        label_names,
        Num2Label,
    ) = prepare_csnecg_data(
        base_path='.',  # Current directory for HDF5 file
        batch_size=batch_size,
        hdf5_file_path=f'csnecg_segments_{peaks_per_signal}peaks.hdf5'
    )

    # Calculate dataset sizes from local HDF5
    with h5py.File(f'csnecg_segments_{peaks_per_signal}peaks.hdf5', 'r') as f:
        total_size = f['segments'].shape[0]
        train_size = int(0.7 * total_size)
        valid_size = int(0.15 * total_size)
        test_size = total_size - train_size - valid_size

    # Calculate steps correctly
    steps_per_epoch = train_size // batch_size
    validation_steps = valid_size // batch_size
    test_steps = test_size // batch_size

    # Build the ResNet model based on the specified type
    if resnet_type == 'resnet18':
        model = build_resnet18_1d(
            input_shape=(time_steps, 12),
            num_classes=num_classes,
            activation='sigmoid',
            **model_params
        )
    elif resnet_type == 'resnet34':
        model = build_resnet34_1d(
            input_shape=(time_steps, 12),
            num_classes=num_classes,
            activation='sigmoid',
            **model_params
        )
    elif resnet_type == 'resnet50':
        model = build_resnet50_1d(
            input_shape=(time_steps, 12),
            num_classes=num_classes,
            activation='sigmoid',
            **model_params
        )
    else:
        raise ValueError("Invalid ResNet type.")

    # Compile the model
    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate),
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

    # Train the model
    print("\nStarting model training...")
    training_start = time.time()
    
    history = model.fit(
        train_dataset,
        epochs=30,
        steps_per_epoch=steps_per_epoch,
        validation_data=valid_dataset,
        validation_steps=validation_steps,
        callbacks=callbacks
    )
    
    training_end = time.time()
    total_training_time = training_end - training_start
    print(f"\nTotal training time: {total_training_time:.2f} seconds")
    print(f"Average time per epoch: {np.mean(timing_callback.times):.2f} seconds")

    # Generate predictions for test set
    print("\nGenerating predictions for test set...")
    test_timing = {}
    start_time = time.time()
    y_pred = model.predict(test_dataset)
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
    parser = argparse.ArgumentParser(description='Train a ResNet model on the preprocessed CSN ECG dataset.')
    parser.add_argument('--time_steps', type=int, default=300, 
                        help='Number of time steps in the preprocessed data (default: 300).')
    parser.add_argument('--batch_size', type=int, default=128, 
                        help='Batch size for training (default: 128)')
    parser.add_argument('--resnet_type', type=str, choices=['resnet18', 'resnet34', 'resnet50'], 
                        default='resnet18', help='Type of ResNet model to train (default: resnet18)')
    args, unknown = parser.parse_known_args()
    main(args.time_steps, args.batch_size, args.resnet_type)