# mitbih_train_transformer.py

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import class_weight
from tensorflow import keras
import tensorflow as tf

# Import models and evaluation
from models import build_transformer
from evaluation import print_stats, showConfusionMatrix

# Import data preprocessing functions
from mitbih_data_preprocessing import prepare_mitbih_data

def main():
    # Setup
    base_output_dir = 'output_plots'
    dataset_name = 'mitbih'
    model_type = 'transformer'

    # Create a unique directory name with dataset and model
    output_dir = os.path.join(base_output_dir, f"{dataset_name}_{model_type}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Model Parameters
    model_params = {
        'head_size': 256,
        'num_heads': 4,
        'ff_dim': 4,
        'num_transformer_blocks': 4,
        'mlp_units': [128],
        'mlp_dropout': 0.4,
        'dropout': 0.25,
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

    # Ensure all input data is float32
    X_train = tf.cast(X_train, dtype=tf.float32)
    X_valid = tf.cast(X_valid, dtype=tf.float32)
    X_test = tf.cast(X_test, dtype=tf.float32)

    # Build the model
    model = build_transformer(
        input_shape=(X_train.shape[1], X_train.shape[2]),
        num_classes=num_classes,
        activation='softmax',
        **model_params
    )

    # Use a standard optimizer without mixed precision
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

    # Compile the model
    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )

    # Train with adjusted parameters for transformer
    transformer_batch_size = 64  # Adjust this value based on your GPU memory
    transformer_epochs = 50  # Increase epochs to compensate for smaller batch size

    history = model.fit(
        X_train, y_nn_train, 
        epochs=transformer_epochs, 
        validation_data=(X_valid, y_nn_valid),
        batch_size=transformer_batch_size, 
        shuffle=True, 
        class_weight=class_weight_dict,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.1, patience=5)
        ]
    )

    # Evaluate
    def evaluate_model(dataset, y_true, name):
        y_pred = np.argmax(model.predict(dataset), axis=1)
        print(f"\n{name} Performance")
        print_stats(y_pred, y_true)
        showConfusionMatrix(
            y_pred, y_true, f'confusion_matrix_{name.lower()}.png', output_dir, list(Num2Label.values())
        )

    evaluate_model(X_train, y_train, 'Training')
    evaluate_model(X_valid, y_valid, 'Validation')
    evaluate_model(X_test, y_test, 'Test')

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