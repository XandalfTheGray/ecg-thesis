# File: csnecg_train_resnet.py
# This script trains a ResNet model on the preprocessed CSN ECG dataset

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import seaborn as sns
import sys
import tensorflow as tf
from tensorflow import keras
from keras.optimizers import SGD
import argparse

# Add the directory containing your modules to the Python path
sys.path.append('/content/ecg-thesis')

# Import your modules
from models import build_resnet18_1d, build_resnet34_1d, build_resnet50_1d
from evaluation import evaluate_multilabel_model, CustomProgressBar
from csnecg_data_preprocessing import prepare_csnecg_data

def main(time_steps, batch_size, resnet_type):
    # Set up output directories
    base_path = '/content/drive/MyDrive/'
    base_output_dir = os.path.join(base_path, 'csnecg_output_plots')
    dataset_name = 'csnecg'
    output_dir = os.path.join(base_output_dir, f"{dataset_name}_{resnet_type}_{time_steps}steps_{batch_size}batch")
    os.makedirs(output_dir, exist_ok=True)

    # Define model parameters
    model_params = {
        'l2_reg': 0.001,
    }

    # Prepare data
    train_dataset, valid_dataset, test_dataset, num_classes, label_names, Num2Label = prepare_csnecg_data(time_steps, base_path, batch_size)

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
        optimizer=tf.keras.optimizers.Adam(1e-3), # Alternatively, use SGD if desired
        metrics=['accuracy']
    )

    # Define callbacks for training
    callbacks = [
        CustomProgressBar(),
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
    history = model.fit(
        train_dataset,
        epochs=30,
        validation_data=valid_dataset,
        callbacks=callbacks
    )

    # Generate predictions
    y_pred = model.predict(test_dataset)
    y_pred_classes = (y_pred > 0.5).astype(int)
    y_true = np.concatenate([y for x, y in test_dataset], axis=0).astype(int)

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
        f.write(f"Model Type: {resnet_type}\n")
        f.write(f"Time Steps: {time_steps}\n")
        f.write(f"Batch Size: {batch_size}\n")
        for key, value in model_params.items():
            f.write(f"  {key}: {value}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a ResNet model on the preprocessed CSN ECG dataset.')
    parser.add_argument('--time_steps', type=int, choices=[500, 1000, 2000, 5000], required=True, 
                        help='Number of time steps in the preprocessed data.')
    parser.add_argument('--batch_size', type=int, default=128, 
                        help='Batch size for training (default: 128)')
    parser.add_argument('--resnet_type', type=str, choices=['resnet18', 'resnet34', 'resnet50'], 
                        default='resnet18', help='Type of ResNet model to train (default: resnet18)')
    args = parser.parse_args()
    main(args.time_steps, args.batch_size, args.resnet_type)