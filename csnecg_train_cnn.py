# csnecg_train_cnn.py

import os
import sys
import argparse
import time
import numpy as np

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress INFO and WARNING messages

# Suppress absl logging
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
import absl.logging
absl.logging.set_verbosity('error')
import tensorflow as tf
sys.stderr = stderr

# Import your modules
from models import build_cnn
from evaluation import (
    evaluate_multilabel_model,
    CustomProgressBar,
    TimingCallback,
    log_timing_info
)
# Import functions from csnecg_data_preprocessing.py
from csnecg_data_preprocessing import load_data_numpy, prepare_data_for_training, ensure_data_available

def main(time_steps, batch_size, peaks_per_signal=1):
    # Set up base path for OUTPUTS on Google Drive
    base_path = '/content/drive/MyDrive/'
    base_output_dir = os.path.join(base_path, 'csnecg_output_plots')
    dataset_name = 'csnecg'
    model_type = 'cnn'
    output_dir = os.path.join(
        base_output_dir, 
        f"{dataset_name}_{model_type}_{time_steps}steps_{batch_size}batch"
    )
    os.makedirs(output_dir, exist_ok=True)
    
    learning_rate = 1e-3

    # Define Google Drive data directory
    drive_data_dir = os.path.join(base_path, 'csnecg_preprocessed_data')

    # Local data directory
    local_data_dir = 'csnecg_preprocessed_data'

    # Ensure data is available
    ensure_data_available(local_data_dir, drive_data_dir, peaks_per_signal)

    # Load data with peaks_per_signal
    data_dir = local_data_dir
    X, Y, label_names = load_data_numpy(data_dir, peaks_per_signal)
    num_classes = Y.shape[1]

    # Prepare data with class weights
    (train_dataset, valid_dataset, test_dataset, 
     steps_per_epoch, validation_steps, test_steps,
     y_test, class_weights) = prepare_data_for_training(
        X, Y, batch_size=batch_size
    )

    # Define model parameters
    model_params = {
        'l2_reg': 0.001,
        'filters': [32, 64, 128],
        'kernel_sizes': [5, 5, 5],
        'dropout_rates': [0.3, 0.3, 0.3, 0.3],
    }

    # Build model
    model = build_cnn(
        input_shape=(time_steps, 12),
        num_classes=num_classes,
        activation='sigmoid',
        **model_params
    )

    # Compile with class weights
    class_weight_dict = {i: w for i, w in enumerate(class_weights)}
    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )

    # Callbacks
    timing_callback = TimingCallback()
    callbacks = [
        CustomProgressBar(),
        timing_callback,
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_auc', mode='max',
            factor=0.5, patience=3, min_lr=1e-6, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_auc', mode='max',
            patience=10, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(output_dir, 'best_model.keras'),
            monitor='val_auc', mode='max',
            save_best_only=True, verbose=1
        )
    ]

    # Train with class weights
    print("\nStarting model training...")
    training_start = time.time()
    
    history = model.fit(
        train_dataset,
        epochs=30,
        validation_data=valid_dataset,
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=0
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
        verbose=1
    )
    end_time = time.time()
    test_timing['Test'] = end_time - start_time
    print(f"Test set prediction time: {test_timing['Test']:.2f} seconds")

    y_pred_classes = (y_pred > 0.5).astype(int)
    y_true = y_test

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
    parser.add_argument('--time_steps', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--peaks_per_signal', type=int, default=1,
                      help='Number of peaks per signal (default: 1)')
    args, unknown = parser.parse_known_args()
    main(args.time_steps, args.batch_size, args.peaks_per_signal)