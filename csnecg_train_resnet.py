# csnecg_train_resnet.py

import os
import sys
import tensorflow as tf
import argparse
import time
import numpy as np

# Import your modules
from models import build_resnet18_1d, build_resnet34_1d, build_resnet50_1d
from evaluation import (
    evaluate_multilabel_model,
    CustomProgressBar,
    TimingCallback,
    log_timing_info
)
# Import functions from csnecg_data_preprocessing.py
from csnecg_data_preprocessing import load_data_numpy, prepare_data_for_training, ensure_data_available

def main(time_steps, batch_size, resnet_type, peaks_per_signal=1):
    # Set up base path for OUTPUTS on Google Drive
    base_path = '/content/drive/MyDrive/'
    base_output_dir = os.path.join(base_path, 'csnecg_output_plots')
    dataset_name = 'csnecg'
    model_type = resnet_type
    output_dir = os.path.join(base_output_dir, f"{dataset_name}_{model_type}_{time_steps}steps_{batch_size}batch")
    os.makedirs(output_dir, exist_ok=True)

    # Define Google Drive data directory
    drive_data_dir = os.path.join(base_path, 'csnecg_preprocessed_data')

    # Local data directory
    local_data_dir = 'csnecg_preprocessed_data'

    # Ensure data is available
    ensure_data_available(local_data_dir, drive_data_dir, peaks_per_signal)

    learning_rate = 1e-3

    model_params = {
        'l2_reg': 0.001,
    }

    # Load data with peaks_per_signal
    data_dir = local_data_dir
    X, Y, label_names = load_data_numpy(data_dir, peaks_per_signal)
    num_classes = Y.shape[1]

    # Prepare data
    train_dataset, valid_dataset, test_dataset, steps_per_epoch, validation_steps, test_steps = prepare_data_for_training(
        X, Y, batch_size=batch_size
    )

    # Build the ResNet model
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

    # Callbacks
    timing_callback = TimingCallback()
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
    parser = argparse.ArgumentParser(description='Train a ResNet model on the preprocessed CSN ECG dataset.')
    parser.add_argument('--time_steps', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--resnet_type', type=str, choices=['resnet18', 'resnet34', 'resnet50'], 
                      default='resnet18')
    parser.add_argument('--peaks_per_signal', type=int, default=1,  # Add new argument
                      help='Number of peaks per signal (default: 1)')
    args, unknown = parser.parse_known_args()
    main(args.time_steps, args.batch_size, args.resnet_type, args.peaks_per_signal)