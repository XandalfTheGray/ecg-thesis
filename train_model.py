# train_model.py

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from tensorflow import keras

# Import custom modules
from data_preprocessing import processRecord, segmentSignal, create_nn_labels
from models import build_cnn, build_resnet18_1d, build_resnet34_1d, build_resnet50_1d
from evaluation import print_stats, showConfusionMatrix

def main():
    """
    Main function to train and evaluate the ECG classification model.
    """
    # Setup directories and paths
    output_dir = 'output_plots'
    os.makedirs(output_dir, exist_ok=True)
    database_path = os.path.join('mit-bih-arrhythmia-database', 'mit-bih-arrhythmia-database-1.0.0')

    # Define data entries and labels
    dataEntries = [100, 101, 103, 105, 106, 107, 108, 109, 111, 112, 113,
                  114, 115, 116, 117, 118, 119, 121, 122, 123, 124, 200, 201, 202,
                  203, 205, 207, 208, 209, 210, 212, 213, 214, 215, 217, 219, 221,
                  222, 223, 228, 230, 231, 232, 233, 234]
    valid_labels = ['N', 'V', 'A', 'R', 'L', '/']
    label2Num = {label: idx for idx, label in enumerate(valid_labels)}
    Num2Label = {idx: label for idx, label in enumerate(valid_labels)}

    # Load and preprocess data
    X, Y_cl = [], []
    for record in dataEntries:
        rec = processRecord(record, database_path)
        tX, tY, _ = segmentSignal(rec, valid_labels, label2Num)
        X.extend(tX)
        Y_cl.extend(tY)
    X, Y_cl = np.array(X), np.array(Y_cl)

    # Display class distribution
    recLabels, labelCounts = np.unique(Y_cl, return_counts=True)
    label_dict = dict(zip(recLabels, labelCounts))
    print("Class distribution in the dataset:")
    for label_num, count in label_dict.items():
        print(f"Class {Num2Label[label_num]}: {count} samples")
    print("Total samples:", Y_cl.shape[0])

    # Split data into training, validation, and test sets with stratification
    X_train, X_test, y_cl_train, y_cl_test = train_test_split(
        X, Y_cl, test_size=0.10, random_state=12, stratify=Y_cl
    )
    X_train, X_valid, y_cl_train, y_cl_valid = train_test_split(
        X_train, y_cl_train, test_size=0.10, random_state=87, stratify=y_cl_train
    )

    # One-Hot Encode Labels
    num_classes = len(valid_labels)
    y_nn_train = create_nn_labels(y_cl_train, num_classes)
    y_nn_valid = create_nn_labels(y_cl_valid, num_classes)
    y_nn_test = create_nn_labels(y_cl_test, num_classes)

    # Scale Data
    scaler = StandardScaler()
    n_timesteps, n_features = X_train.shape[1], 1
    X_train_cnn = scaler.fit_transform(X_train.reshape(-1, n_timesteps)).reshape(-1, n_timesteps, n_features)
    X_valid_cnn = scaler.transform(X_valid.reshape(-1, n_timesteps)).reshape(-1, n_timesteps, n_features)
    X_test_cnn = scaler.transform(X_test.reshape(-1, n_timesteps)).reshape(-1, n_timesteps, n_features)

    # Handle Class Imbalance with Class Weights
    class_weights_vals = class_weight.compute_class_weight('balanced',
                                                            classes=np.unique(y_cl_train),
                                                            y=y_cl_train)
    class_weight_dict = dict(enumerate(class_weights_vals))
    print("Class weights:", class_weight_dict)

    # Model Parameters
    model_type = 'resnet18'  # Options: 'cnn', 'resnet18', 'resnet34', 'resnet50'
    model_params = {
        'l2_reg': 0.001,                        # Increased L2 regularization to penalize large weights
        'filters': (64, 128, 256, 512),          # Standard ResNet18 filter sizes
        'kernel_sizes': (3, 3, 3, 3),            # Consistent kernel sizes for simplicity
        'dropout_rates': (0.3, 0.3, 0.3, 0.6)    # Increased dropout rates to prevent overfitting
    }

    # Build Model
    if model_type == 'cnn':
        model = build_cnn(input_shape=(n_timesteps, n_features),
                         num_classes=num_classes,
                         **model_params)
    elif model_type == 'resnet18':
        model = build_resnet18_1d(input_shape=(n_timesteps, n_features),
                                 num_classes=num_classes,
                                 l2_reg=model_params['l2_reg'])
    elif model_type == 'resnet34':
        model = build_resnet34_1d(input_shape=(n_timesteps, n_features),
                                 num_classes=num_classes,
                                 l2_reg=model_params['l2_reg'])
    elif model_type == 'resnet50':
        model = build_resnet50_1d(input_shape=(n_timesteps, n_features),
                                 num_classes=num_classes,
                                 l2_reg=model_params['l2_reg'])
    else:
        raise ValueError("Invalid model type. Choose 'cnn', 'resnet18', 'resnet34', or 'resnet50'.")

    # Compile Model
    model.compile(
        loss='categorical_crossentropy',
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        metrics=['accuracy']
    )
    print(model.summary())

    # Define Callbacks
    callbacks = [
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1),
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
        keras.callbacks.ModelCheckpoint(filepath=os.path.join(output_dir, 'best_model.keras'),
                                        monitor='val_loss', save_best_only=True, verbose=1)
    ]


    # Train Model
    history = model.fit(
        X_train_cnn, y_nn_train,
        epochs=30,  # Reduced from 50 for quicker experimentation
        validation_data=(X_valid_cnn, y_nn_valid),
        batch_size=256,  # Reduced from 512
        shuffle=True,
        verbose=1,
        callbacks=callbacks,
        class_weight=class_weight_dict
    )

    # Evaluation Function
    def evaluate_model(dataset, y_true, dataset_name):
        y_preds = model.predict(dataset)
        y_pred = np.argmax(y_preds, axis=1)
        print(f"\n{dataset_name} Data Performance")
        print_stats(y_pred, y_true)
        showConfusionMatrix(y_pred, y_true, f'confusion_matrix_{dataset_name.lower()}.png', output_dir, valid_labels)

    # Evaluate on Training, Validation, and Test Sets
    evaluate_model(X_train_cnn, np.argmax(y_nn_train, axis=1), 'Training')
    evaluate_model(X_valid_cnn, np.argmax(y_nn_valid, axis=1), 'Validation')
    evaluate_model(X_test_cnn, np.argmax(y_nn_test, axis=1), 'Test')

    # Plot Training History
    for metric in ['loss', 'accuracy']:
        plt.figure()
        plt.plot(history.history[metric], label=f'Training {metric.capitalize()}')
        plt.plot(history.history[f'val_{metric}'], label=f'Validation {metric.capitalize()}')
        plt.title(f'Model {metric.capitalize()}')
        plt.xlabel('Epoch')
        plt.ylabel(metric.capitalize())
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{metric}_plot.png'))
        plt.close()

if __name__ == '__main__':
    main()
