# mitbih_train_resnet.py

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import class_weight
from tensorflow import keras
from keras.optimizers import SGD
import time

# Update this import to include TimingCallback and log_timing_info
from evaluation import (
    print_stats, 
    showConfusionMatrix, 
    CustomProgressBar, 
    TimingCallback,
    log_timing_info
)

# Import models and evaluation
from models import build_resnet18_1d, build_resnet34_1d, build_resnet50_1d

# Import data preprocessing functions
from mitbih_data_preprocessing import prepare_mitbih_data

def main():
    # Setup
    base_output_dir = 'output_plots'
    dataset_name = 'mitbih'
    model_type = 'resnet50'  # 'resnet18', 'resnet34', or 'resnet50'

    # Create a unique directory name with dataset and model
    output_dir = os.path.join(base_output_dir, f"{dataset_name}_{model_type}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Model Parameters
    model_params = {
        'l2_reg': 0.001,
    }

    # Define data entries and labels
    data_entries = ['100', '101', '103', '105', '106', '107', '108', '109', '111', '112', '113',
                    '114', '115', '116', '117', '118', '119', '121', '122', '123', '124', '200', '201', '202',
                    '203', '205', '207', '208', '209', '210', '212', '213', '214', '215', '217', '219', '221',
                    '222', '223', '228', '230', '231', '232', '233', '234']
    valid_labels = ['N', 'V', 'A', 'R', 'L', '/']
    database_path = 'mit-bih-arrhythmia-database/mit-bih-arrhythmia-database-1.0.0/'

    # Prepare data
    X_train, X_valid, X_test, y_train, y_valid, y_test, num_classes, Num2Label = prepare_mitbih_data(
        data_entries, valid_labels, database_path
    )

    # One-Hot Encode
    y_nn_train = keras.utils.to_categorical(y_train, num_classes)
    y_nn_valid = keras.utils.to_categorical(y_valid, num_classes)
    y_nn_test = keras.utils.to_categorical(y_test, num_classes)

    # Class Weights
    class_weights = class_weight.compute_class_weight(
        'balanced', classes=np.unique(y_train), y=y_train
    )
    class_weight_dict = dict(enumerate(class_weights))

    # Build Model
    if model_type == 'resnet18':
        model = build_resnet18_1d(
            input_shape=(X_train.shape[1], X_train.shape[2]),
            num_classes=num_classes,
            l2_reg=model_params['l2_reg']
        )
    elif model_type == 'resnet34':
        model = build_resnet34_1d(
            input_shape=(X_train.shape[1], X_train.shape[2]),
            num_classes=num_classes,
            l2_reg=model_params['l2_reg']
        )
    elif model_type == 'resnet50':
        model = build_resnet50_1d(
            input_shape=(X_train.shape[1], X_train.shape[2]),
            num_classes=num_classes,
            l2_reg=model_params['l2_reg']
        )

    # Define optimizer with SGD and momentum
    optimizer = SGD(learning_rate=1e-3, momentum=0.9) # LR of 1e-4 for ResNet50 case

    # Compile model
    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )

    # Create timing callback
    timing_callback = TimingCallback()
    
    # Add timing callback to the callbacks list
    callbacks = [
        CustomProgressBar(),
        timing_callback,
        keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.1, patience=5)
    ]

    # Train model with timing
    print("\nStarting model training...")
    training_start = time.time()
    
    history = model.fit(
        X_train, y_nn_train,
        epochs=50, # 30 for ResNet50 case
        batch_size=256, # 128 for ResNet50 case
        validation_data=(X_valid, y_nn_valid),
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=0
    )
    
    training_end = time.time()
    total_training_time = training_end - training_start
    print(f"\nTotal training time: {total_training_time:.2f} seconds")
    print(f"Average time per epoch: {np.mean(timing_callback.times):.2f} seconds")

    # Time the prediction/evaluation phase
    test_timing = {}
    
    def evaluate_model(dataset, y_true, name):
        print(f"\nGenerating predictions for {name} set...")
        start_time = time.time()
        y_pred = np.argmax(model.predict(dataset), axis=1)
        end_time = time.time()
        test_timing[name] = end_time - start_time
        
        print(f"{name} Performance:")
        print(f"Prediction Time: {test_timing[name]:.2f} seconds")
        print_stats(y_pred, y_true)
        showConfusionMatrix(
            y_pred, y_true, f'confusion_matrix_{name.lower()}.png', output_dir, list(Num2Label.values())
        )

    evaluate_model(X_train, y_train, 'Training')
    evaluate_model(X_valid, y_valid, 'Validation')
    evaluate_model(X_test, y_test, 'Test')

    # Print summary of timing information
    print("\nTiming Summary:")
    print(f"Total Training Time: {total_training_time:.2f} seconds")
    print(f"Average Time per Epoch: {np.mean(timing_callback.times):.2f} seconds")
    print("\nPrediction Times:")
    for name, time_taken in test_timing.items():
        print(f"{name} Set: {time_taken:.2f} seconds")

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

    # Plot
    for metric in ['loss', 'accuracy']:
        plt.figure()
        plt.plot(history.history[metric], label=f'Train {metric.capitalize()}')
        plt.plot(history.history[f'val_{metric}'], label=f'Val {metric.capitalize()}')
        plt.title(f'Model {metric.capitalize()}')
        plt.xlabel('Epoch')
        plt.ylabel(metric.capitalize())
        plt.legend()
        plt.savefig(os.path.join(output_dir, f'{metric}.png'))
        plt.close()

    # Save model parameters to a text file
    with open(os.path.join(output_dir, 'model_params.txt'), 'w') as f:
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Model Type: {model_type}\n")
        f.write("Model Parameters:\n")
        for key, value in model_params.items():
            f.write(f"  {key}: {value}\n")

if __name__ == '__main__':
    main()
