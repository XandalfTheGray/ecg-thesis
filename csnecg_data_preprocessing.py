# csnecg_data_preprocessing.py
# This script preprocesses the CSN ECG dataset, extracting SNOMED-CT codes and ECG data

import numpy as np
import os
import random
from scipy.io import loadmat
import wfdb
import logging
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from collections import Counter
import tensorflow as tf

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_snomed_ct_mapping(csv_path, class_mapping):
    """
    Load SNOMED-CT codes and their corresponding class names from a CSV file.
    """
    df = pd.read_csv(csv_path)
    
    # Create a reverse mapping from condition names to class names
    condition_to_class = {}
    for class_name, conditions in class_mapping.items():
        for condition in conditions:
            condition_to_class[condition.lower()] = class_name
    
    # Create the final mapping from SNOMED-CT codes to class names
    mapping = {}
    for _, row in df.iterrows():
        snomed_ct_code = str(row['Snomed_CT'])
        condition_name = row['Full Name'].lower()
        if condition_name in condition_to_class:
            mapping[snomed_ct_code] = condition_to_class[condition_name]
    
    logging.info(f"Loaded {len(mapping)} SNOMED-CT codes from CSV")
    return mapping

def extract_snomed_ct_codes(header):
    """
    Extract all SNOMED-CT codes from the header of an ECG record.
    """
    codes = []
    for comment in header.comments:
        if comment.startswith('Dx:'):
            codes_part = comment.split(':')[1].strip()
            codes = [code.strip() for code in codes_part.split(',')]
            break
    if not codes:
        logging.warning(f"No Dx field found in header comments: {header.comments}")
    return codes

def pad_ecg_data(ecg_data, desired_length):
    """
    Pad or truncate ECG data to a fixed number of time steps.
    """
    current_length = ecg_data.shape[0]
    if current_length < desired_length:
        padding = np.zeros((desired_length - current_length, ecg_data.shape[1]))
        ecg_padded = np.vstack((ecg_data, padding))
    else:
        ecg_padded = ecg_data[:desired_length, :]
    return ecg_padded

def plot_ecg_signal(ecg_signal, record_name, plot_dir, class_names):
    """
    Plot and save an ECG signal.
    """
    plt.figure(figsize=(15, 5))
    for lead in range(ecg_signal.shape[1]):
        plt.plot(ecg_signal[:, lead], label=f'Lead {lead+1}')
    class_label = ', '.join(class_names)
    plt.title(f'ECG Signal for Record: {record_name} | Classes: {class_label}')
    plt.xlabel('Time Steps')
    plt.ylabel('Amplitude')
    plt.legend(loc='upper right', ncol=4, fontsize='small')
    plt.tight_layout()
    
    # Create the plot directory if it doesn't exist
    os.makedirs(plot_dir, exist_ok=True)
    
    # Save the plot with a simplified filename
    simplified_name = record_name.replace('\\', '_').replace('/', '_')
    plt.savefig(os.path.join(plot_dir, f'{simplified_name}.png'))
    plt.close()

def load_data(database_path, data_entries, snomed_ct_mapping, max_records=None, desired_lengths=[500, 1000, 2000], num_plots_per_class=1, plot_dir='output_plots/class_ecg_plots/'):
    """
    Load and preprocess ECG data from the CSN dataset and save to disk.
    """
    for desired_length in desired_lengths:
        X, Y_cl = [], []
        processed_records = 0
        skipped_records = 0
        diagnosis_counts = {}
        no_code_count = 0
        plotted_classes = set()
        
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        
        logging.info(f"Starting to process data for {desired_length} time steps. Max records: {max_records}")
        logging.info(f"Number of data entries: {len(data_entries)}")
        
        # Randomly sample records if max_records is specified
        if max_records is not None:
            random.seed(42)
            data_entries = random.sample(data_entries, min(max_records, len(data_entries)))
        
        for i, record in enumerate(data_entries):
            mat_file = os.path.join(database_path, record + '.mat')
            hea_file = os.path.join(database_path, record + '.hea')
        
            if not os.path.exists(mat_file) or not os.path.exists(hea_file):
                logging.warning(f"Files not found for record {record}")
                skipped_records += 1
                continue
        
            try:
                # Load ECG data from .mat file
                mat_data = loadmat(mat_file)
                if 'val' not in mat_data:
                    logging.warning(f"'val' key not found in mat file for record {record}")
                    skipped_records += 1
                    continue
                ecg_data = mat_data['val']
        
                if ecg_data.ndim != 2:
                    logging.warning(f"Unexpected ECG data dimensions for record {record}: {ecg_data.shape}")
                    skipped_records += 1
                    continue
        
                # Read the header file
                try:
                    record_header = wfdb.rdheader(os.path.join(database_path, record))
                except ValueError as ve:
                    logging.error(f"Error processing record {record}: {str(ve)}")
                    skipped_records += 1
                    continue
                except Exception as e:
                    logging.error(f"Unexpected error processing record {record}: {str(e)}")
                    skipped_records += 1
                    continue
        
                # Extract SNOMED-CT codes from the header
                snomed_ct_codes = extract_snomed_ct_codes(record_header)
        
                if not snomed_ct_codes:
                    no_code_count += 1
                    logging.warning(f"No SNOMED-CT codes found for record {record}")
                    continue
        
                # Map SNOMED-CT codes to class names
                valid_classes = []
                for code in snomed_ct_codes:
                    if code in snomed_ct_mapping:
                        class_name = snomed_ct_mapping[code]
                        valid_classes.append(class_name)
                        diagnosis_counts[class_name] = diagnosis_counts.get(class_name, 0) + 1
                    else:
                        valid_classes.append('Other')
                        diagnosis_counts['Other'] = diagnosis_counts.get('Other', 0) + 1
        
                # Remove duplicates and 'Other' if there are other valid classes
                valid_classes = list(set(valid_classes))
                if len(valid_classes) > 1 and 'Other' in valid_classes:
                    valid_classes.remove('Other')
        
                # Pad or truncate ECG data to desired length
                ecg_padded = pad_ecg_data(ecg_data.T, desired_length)
                X.append(ecg_padded)
                Y_cl.append(valid_classes)
        
                # Plot ECG signals for each class (up to num_plots_per_class)
                for cls in valid_classes:
                    if cls not in plotted_classes and list(Y_cl).count(cls) <= num_plots_per_class:
                        plot_ecg_signal(ecg_padded, record, plot_dir, valid_classes)
                        logging.info(f"Plotted ECG signal for class {cls} in record {record}")
                        plotted_classes.add(cls)
        
                processed_records += 1
        
                if processed_records % 100 == 0:
                    logging.info(f"Processed {processed_records} records")
        
                if max_records and processed_records >= max_records:
                    break
        
            except Exception as e:
                logging.error(f"Error processing record {record}: {str(e)}")
                skipped_records += 1
        
        logging.info(f"Processed {processed_records} records")
        logging.info(f"Skipped {skipped_records} records")
        logging.info(f"Records with no SNOMED-CT codes: {no_code_count}")
        logging.info(f"Diagnosis counts: {diagnosis_counts}")
        
        if len(X) == 0:
            logging.error("No data was processed successfully")
            continue
        
        # Convert lists to numpy arrays
        X = np.array(X)
        Y_cl = np.array(Y_cl, dtype=object)
        
        # Save preprocessed data to disk
        preprocessed_data_dir = os.path.join('csnecg_preprocessed_data', f'{desired_length}_signal_time_steps')
        os.makedirs(preprocessed_data_dir, exist_ok=True)
        np.save(os.path.join(preprocessed_data_dir, 'X.npy'), X)
        np.save(os.path.join(preprocessed_data_dir, 'Y.npy'), Y_cl)
        logging.info(f"Saved preprocessed data to '{preprocessed_data_dir}' directory")

def load_preprocessed_data(time_steps, base_path):
    """
    Load preprocessed data from disk.
    """
    data_dir = os.path.join(base_path, 'csnecg_preprocessed_data', f'{time_steps}_signal_time_steps')
    X = np.load(os.path.join(data_dir, 'X.npy'))
    Y_cl = np.load(os.path.join(data_dir, 'Y.npy'), allow_pickle=True)
    return X, Y_cl

def prepare_labels(Y_cl):
    """
    Prepare labels for multi-label classification.
    """
    unique_classes = set(class_name for sublist in Y_cl for class_name in sublist)
    mlb = MultiLabelBinarizer(classes=sorted(unique_classes))
    Y_binarized = mlb.fit_transform(Y_cl)
    label_names = mlb.classes_
    Num2Label = {idx: label for idx, label in enumerate(label_names)}
    label2Num = {label: idx for idx, label in enumerate(label_names)}
    num_classes = len(label_names)
    return Y_binarized, label_names, Num2Label, label2Num, num_classes

def filter_data(X, Y_binarized):
    """
    Remove samples with no positive labels.
    """
    valid_samples = Y_binarized.sum(axis=1) > 0
    X = X[valid_samples]
    Y_binarized = Y_binarized[valid_samples]
    return X, Y_binarized

def remove_rare_combinations(X, Y_binarized, min_combination_count=2):
    """
    Remove samples with rare label combinations.
    """
    label_combinations = [tuple(row) for row in Y_binarized]
    combination_counts = Counter(label_combinations)
    valid_combinations = [comb for comb, count in combination_counts.items() if count >= min_combination_count]
    valid_indices = [i for i, comb in enumerate(label_combinations) if comb in valid_combinations]
    X = X[valid_indices]
    Y_binarized = Y_binarized[valid_indices]
    return X, Y_binarized

def split_and_scale_data(X, Y_binarized, test_size=0.2, val_size=0.25):
    """
    Split the data into train, validation, and test sets, and scale the features.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y_binarized, test_size=test_size, random_state=42
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train, y_train, test_size=val_size, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scale_dataset(X_train, scaler)
    X_valid_scaled = scale_dataset(X_valid, scaler)
    X_test_scaled = scale_dataset(X_test, scaler)

    return X_train_scaled, X_valid_scaled, X_test_scaled, y_train, y_valid, y_test

def scale_dataset(X, scaler):
    """
    Scale the dataset using StandardScaler.
    """
    num_samples, num_timesteps, num_channels = X.shape
    X_reshaped = X.reshape(-1, num_channels)
    X_scaled = scaler.fit_transform(X_reshaped)
    return X_scaled.reshape(num_samples, num_timesteps, num_channels)

def create_tf_dataset(X, y, batch_size=32, shuffle=True, prefetch=True):
    """
    Create a TensorFlow dataset from numpy arrays.
    """
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(X))
    dataset = dataset.batch(batch_size)
    if prefetch:
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

def prepare_csnecg_data(time_steps, base_path, batch_size):
    """
    Prepare CSN ECG data for model training, excluding the 'Other' class.
    """
    X, Y_cl = load_preprocessed_data(time_steps, base_path)
    Y_binarized, label_names, Num2Label, label2Num, num_classes = prepare_labels(Y_cl)
    X, Y_binarized = filter_data(X, Y_binarized)
    X, Y_binarized = remove_rare_combinations(X, Y_binarized)

    # Identify the index of the "Other" class
    other_class_index = label_names.index('Other')  # Ensure 'label_names' includes 'Other'

    # Create a mask to filter out "Other" class samples
    non_other_mask = Y_binarized[:, other_class_index] == 0  # Assuming one-hot encoding

    # Apply the mask to data and labels
    X = X[non_other_mask]
    Y_binarized = Y_binarized[non_other_mask]

    # Remove the 'Other' column from Y_binarized
    Y_binarized = np.delete(Y_binarized, other_class_index, axis=1)

    # Update label names by removing "Other"
    label_names = [label for label in label_names if label != 'Other']

    # Update the number of classes
    num_classes = len(label_names)

    # Split and scale the data
    X_train, X_valid, X_test, y_train, y_valid, y_test = split_and_scale_data(X, Y_binarized)

    # Ensure that the Other class was removed
    print(f"Number of classes after removal of 'Other': {num_classes}")
    print(f"Shape of Y_binarized: {Y_binarized.shape}")

    # Create TensorFlow datasets
    train_dataset = create_tf_dataset(X_train, y_train, batch_size=batch_size)
    valid_dataset = create_tf_dataset(X_valid, y_valid, batch_size=batch_size, shuffle=False)
    test_dataset = create_tf_dataset(X_test, y_test, batch_size=batch_size, shuffle=False)

    return train_dataset, valid_dataset, test_dataset, num_classes, label_names, Num2Label

def main():
    """
    Main function to preprocess the CSN ECG dataset and save processed data to disk.
    """
    # Define paths
    database_path = os.path.join('a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0', 
                                 'a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0',
                                 'WFDBRecords')
    csv_path = os.path.join('a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0', 
                            'a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0',
                            'ConditionNames_SNOMED-CT.csv')

    # Define the class mapping
    class_mapping = {
        'AFIB': ['Atrial fibrillation', 'Atrial flutter'],
        'GSVT': ['Supraventricular tachycardia', 'Atrial tachycardia', 'Sinus node dysfunction', 
                 'Sinus tachycardia', 'Atrioventricular nodal reentry tachycardia', 'Atrioventricular reentrant tachycardia'],
        'SB': ['Sinus bradycardia'],
        'SR': ['Sinus rhythm', 'Sinus irregularity']
    }

    # Load SNOMED-CT mapping with class mapping
    snomed_ct_mapping = load_snomed_ct_mapping(csv_path, class_mapping)

    # Get all record names
    data_entries = []
    for subdir, dirs, files in os.walk(database_path):
        for file in files:
            if file.endswith('.mat'):
                record_path = os.path.join(subdir, file)
                record_name = os.path.relpath(record_path, database_path)
                record_name = os.path.splitext(record_name)[0]
                data_entries.append(record_name)

    # Load data with plotting
    load_data(
        database_path, 
        data_entries, 
        snomed_ct_mapping, 
        max_records=45152,  # Set to desired number of records
        desired_lengths=[500, 1000, 2000, 5000],  # Process data for these lengths
        num_plots_per_class=1,
        plot_dir='output_plots/class_ecg_plots/'
    )

if __name__ == '__main__':
    main()