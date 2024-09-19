# train_model.py

import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
import matplotlib
import matplotlib.pyplot as plt

# Import custom modules
from data_preprocessing import processRecord, segmentSignal, create_nn_labels
from models import build_cnn, build_resnet1d
from evaluation import print_stats, showConfusionMatrix

def main():
    """
    Main function to train and evaluate the ECG classification model.

    This function performs the following steps:
    - Loads and preprocesses the ECG data.
    - Splits the data into training, validation, and test sets.
    - Builds and compiles the selected model.
    - Trains the model on the training data.
    - Evaluates the model on the training, validation, and test sets.
    - Saves performance metrics and confusion matrices.
    """
    # Set up directories and paths
    output_dir = 'output_plots'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Define database path
    database_path = os.path.join('mit-bih-arrhythmia-database', 'mit-bih-arrhythmia-database-1.0.0')

    # Sampling rate and data entries
    samplingRate = 360
    dataEntries = [100, 101, 103, 105, 106, 107, 108, 109, 111, 112, 113,
                   114, 115, 116, 117, 118, 119, 121, 122, 123, 124, 200, 201, 202,
                   203, 205, 207, 208, 209, 210, 212, 213, 214, 215, 217, 219, 221,
                   222, 223, 228, 230, 231, 232, 233, 234]
    invalidDataEntries = [102, 104]

    # Define valid labels and create label mappings
    valid_labels = ['N', 'V', 'A', 'R', 'L', '/']
    label2Num = dict(zip(valid_labels, np.arange(len(valid_labels))))
    Num2Label = dict(zip(np.arange(len(valid_labels)), valid_labels))

    # Data Wrangling: Load and preprocess data
    X = []  # Input signals
    Y_cl = []  # Class labels
    Z = []  # Letter labels

    for record in dataEntries:
        rec = processRecord(record, database_path)
        tX, tY, tZ = segmentSignal(rec, valid_labels, label2Num)
        X.extend(tX)
        Y_cl.extend(tY)
        Z.extend(tZ)

    # Convert to numpy arrays
    X = np.array(X)
    Y_cl = np.array(Y_cl)
    Z = np.array(Z)

    # Display class distribution
    recLabels, labelCounts = np.unique(Y_cl, return_counts=True)
    label_dict = dict(zip(recLabels, labelCounts))
    print("Class distribution in the dataset:")
    for label_num, count in label_dict.items():
        print(f"Class {Num2Label[label_num]}: {count} samples")
    print("Total samples:", Y_cl.shape[0])

    # Split data into training, validation, and test sets
    X_train, X_test, y_cl_train, y_cl_test = train_test_split(X, Y_cl, test_size=0.10, random_state=12)
    X_train, X_valid, y_cl_train, y_cl_valid = train_test_split(X_train, y_cl_train, test_size=0.10, random_state=87)

    # Create one-hot encoded labels
    num_classes = len(valid_labels)
    y_nn_train = create_nn_labels(y_cl_train, num_classes)
    y_nn_valid = create_nn_labels(y_cl_valid, num_classes)
    y_nn_test = create_nn_labels(y_cl_test, num_classes)

    # Clear previous Keras sessions
    keras.backend.clear_session()

    # Define input shape
    n_timesteps = X_train.shape[1]
    n_features = 1  # Since the data is 1D
    input_shape = (n_timesteps, n_features)

    # Reshape data for model input
    X_train_cnn = X_train.reshape((X_train.shape[0], n_timesteps, n_features))
    X_valid_cnn = X_valid.reshape((X_valid.shape[0], n_timesteps, n_features))
    X_test_cnn = X_test.reshape((X_test.shape[0], n_timesteps, n_features))

    # Build and compile the model
    # Choose between 'cnn' and 'resnet1d'
    model_type = 'resnet1d'  # Change to 'cnn' to use the CNN model

    if model_type == 'cnn':
        model = build_cnn(input_shape, num_classes)
    elif model_type == 'resnet1d':
        model = build_resnet1d(input_shape, num_classes)
    else:
        raise ValueError("Invalid model type. Choose 'cnn' or 'resnet1d'.")

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())

    # Train the model
    history = model.fit(X_train_cnn, y_nn_train,
                        epochs=20,
                        validation_data=(X_valid_cnn, y_nn_valid),
                        batch_size=512,
                        shuffle=True,
                        verbose=1)

    # Evaluate the model and save results
    class_labels = valid_labels  # For confusion matrix labels

    # Training data evaluation
    print("\nTraining Data Performance")
    y_preds_train = model.predict(X_train_cnn)
    y_pred_train = np.argmax(y_preds_train, axis=1)
    y_true_train = np.argmax(y_nn_train, axis=1)
    print_stats(y_pred_train, y_true_train)
    showConfusionMatrix(y_pred_train, y_true_train, 'confusion_matrix_training.png', output_dir, class_labels)

    # Validation data evaluation
    print("\nValidation Data Performance")
    y_preds_valid = model.predict(X_valid_cnn)
    y_pred_valid = np.argmax(y_preds_valid, axis=1)
    y_true_valid = np.argmax(y_nn_valid, axis=1)
    print_stats(y_pred_valid, y_true_valid)
    showConfusionMatrix(y_pred_valid, y_true_valid, 'confusion_matrix_validation.png', output_dir, class_labels)

    # Test data evaluation
    print("\nTest Data Performance")
    y_preds_test = model.predict(X_test_cnn)
    y_pred_test = np.argmax(y_preds_test, axis=1)
    y_true_test = np.argmax(y_nn_test, axis=1)
    print_stats(y_pred_test, y_true_test)
    showConfusionMatrix(y_pred_test, y_true_test, 'confusion_matrix_test.png', output_dir, class_labels)

    # Plot training & validation loss values
    plt.figure()
    plt.plot(history.history['loss'], marker='.', label='Training Loss')
    plt.plot(history.history['val_loss'], marker='.', label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Categorical Crossentropy)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loss_plot.png'))
    plt.close()

    # Plot training & validation accuracy values
    plt.figure()
    plt.plot(history.history['accuracy'], marker='.', label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], marker='.', label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_plot.png'))
    plt.close()

if __name__ == '__main__':
    main()
