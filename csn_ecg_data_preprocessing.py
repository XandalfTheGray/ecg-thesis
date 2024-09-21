# csn_ecg_data_preprocessing.py

import numpy as np
import os
import random
from scipy.io import loadmat

def load_data(database_path, data_entries, valid_labels, label2Num, max_records=None):
    X, Y_cl = [], []
    processed_records = 0
    
    print(f"Starting to process CSN ECG data. Max records: {max_records}")
    print(f"Database path: {database_path}")
    print(f"Number of data entries: {len(data_entries)}")
    print(f"Valid labels: {valid_labels}")
    
    # Optionally limit the number of records
    if max_records is not None:
        data_entries = random.sample(data_entries, min(max_records, len(data_entries)))
    
    for record in data_entries:
        print(f"\nProcessing record: {record}")
        mat_file = os.path.join(database_path, record + '.mat')
        print(f"Full mat file path: {mat_file}")
        
        if not os.path.exists(mat_file):
            print(f"Error: Mat file not found for record {record}")
            continue
        
        try:
            # Load the mat file
            mat_data = loadmat(mat_file)
            
            # Print the keys in the mat file
            print(f"Keys in the mat file: {mat_data.keys()}")
            
            # Assuming the ECG data is stored directly in the mat file
            # You might need to adjust this based on the actual structure
            if 'val' in mat_data:
                ecg_signal = mat_data['val']
                print(f"Successfully read signal. Shape: {ecg_signal.shape}")
            else:
                print("Error: 'val' key not found in the mat file")
                continue
            
            # Check if 'rhythm' exists in the mat file
            if 'rhythm' in mat_data:
                rhythm_annotations = mat_data['rhythm']
                print(f"Rhythm annotations: {rhythm_annotations}")
            else:
                print("Warning: 'rhythm' key not found in the mat file")
                rhythm_annotations = ['Unknown']  # Default label if not found
            
            # Process the rhythm annotations
            valid_beats = 0
            for label in rhythm_annotations:
                if label in valid_labels:
                    X.append(ecg_signal)
                    Y_cl.append(label2Num[label])
                    valid_beats += 1
            
            processed_records += 1
            print(f"Processed record {record}. Valid beats extracted: {valid_beats}")
            print(f"Current X length: {len(X)}, Y_cl length: {len(Y_cl)}")
        
        except Exception as e:
            print(f"Error processing record {record}: {str(e)}")
            print(f"Error type: {type(e).__name__}")
            continue  # Skip problematic records
    
    print(f"\nProcessed {processed_records} records")
    print(f"Final X length: {len(X)}, Y_cl length: {len(Y_cl)}")
    
    if len(X) > 0:
        print(f"Shape of first item in X: {X[0].shape}")
        print(f"Unique labels in Y_cl: {set(Y_cl)}")
    
    return np.array(X), np.array(Y_cl)
