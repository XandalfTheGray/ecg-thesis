# csnecg_train_transformer.py

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import seaborn as sns
import sys
import tensorflow as tf
from tensorflow import keras
from keras import mixed_precision
import argparse
import time

# Enable mixed precision for better performance on GPUs
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# Add the directory containing your modules to the Python path
sys.path.append('/content/ecg-thesis')

# Import your modules
from models import build_transformer
from evaluation import (
    evaluate_multilabel_model, 
    CustomProgressBar,
    TimingCallback,
    log_timing_info
)
from csnecg_data_preprocessing import prepare_csnecg_data

def main(time_steps, batch_size):
    # Set up output directories
    base_path = '/content/drive/MyDrive/'
    base_output_dir = os.path.join(base_path, 'csnecg_output_plots')
    dataset_name = 'csnecg'
    model_type = 'transformer'
    # Old way:
    # output_dir = os.path.join(base_output_dir, f"{dataset_name}_{model_type}_300steps_{batch_size}batch")
    # Using fixed 300-sample windows
    output_dir = os.path.join(base_output_dir, f"{dataset_name}_{model_type}_300steps_{batch_size}batch")
    os.makedirs(output_dir, exist_ok=True)

    model_params = {
        'head_size': 256,
        'num_heads': 8,
        'ff_dim': 16,
        'num_transformer_blocks': 4,
        'mlp_units': [128],
        'mlp_dropout': 0.3,
        'dropout': 0.25,
    }

    # Prepare data
    train_dataset, valid_dataset, test_dataset, num_classes, label_names, Num2Label = prepare_csnecg_data(
        base_path=base_path, batch_size=batch_size
    )

    # Build the Transformer model
    model = build_transformer(
        input_shape=(300, 12),  # Fixed window size
        num_classes=num_classes,
        activation='sigmoid',
        **model_params
    )

    # Compile the model with mixed precision optimizer
    optimizer = tf.keras.optimizers.Adam(1e-4)
    optimizer = mixed_precision.LossScaleOptimizer(optimizer)

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

    # Start prediction timing
    predict_start_time = time.time()

    # Train the model
    history = model.fit(
        train_dataset,
        epochs=30,
        validation_data=valid_dataset,
        callbacks=callbacks
    )

    # Time predictions
    test_timing = {}
    start_time = time.time()
    y_pred = model.predict(test_dataset)
    end_time = time.time()
    test_timing['Test'] = end_time - start_time

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
        f.write(f"Test Set Prediction Time: {test_timing['Test']:.2f} seconds\n")

    # Evaluate and visualize using the centralized evaluate_multilabel_model function
    evaluate_multilabel_model(
        y_true=y_true,
        y_pred=y_pred_classes,
        y_scores=y_pred,
        label_names=label_names,
        output_dir=output_dir,
        history=history  # Pass the history object
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
    parser = argparse.ArgumentParser(description='Train a Transformer model on the preprocessed CSN ECG dataset.')
    # parser.add_argument('--time_steps', type=int, choices=[500, 1000, 2000, 5000], required=True, 
    #                     help='Number of time steps in the preprocessed data.')
    parser.add_argument('--time_steps', type=int, default=300, 
                        help='Number of time steps in the preprocessed data (default: 300).')
    parser.add_argument('--batch_size', type=int, default=128, 
                        help='Batch size for training (default: 128)')
    args = parser.parse_args()
    main(args.time_steps, args.batch_size)