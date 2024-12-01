import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import wfdb  # Add this import to read ECG files

def load_npz_data(data_dir, peaks_per_signal):
    """Load data directly from npz files"""
    peaks_dir = os.path.join(data_dir, f'peaks_{peaks_per_signal}')
    
    # Load the npz files with allow_pickle=True
    X = np.load(os.path.join(peaks_dir, 'X.npz'), allow_pickle=True)['arr_0']
    Y = np.load(os.path.join(peaks_dir, 'Y.npz'), allow_pickle=True)['arr_0']
    label_names = np.load(os.path.join(peaks_dir, 'label_names.npz'), allow_pickle=True)['arr_0']
    
    return X, Y, label_names

def count_class_distribution(X, Y, label_names):
    """Count the number of samples in each class"""
    # Get indices where each class is positive (1)
    class_indices = np.where(Y == 1)
    
    # Count occurrences of each class
    counts = Counter(class_indices[1])  # [1] gives us the class indices
    
    print("\nClass Distribution:")
    for idx in sorted(counts.keys()):
        print(f"{label_names[idx]}: {counts[idx]} samples")
    return counts

def visualize_samples_by_class(X, Y, label_names, samples_per_class=10):
    """Visualize samples for each class in a grid layout"""
    num_classes = len(label_names)
    
    # Get indices for each class
    class_indices = {i: np.where(Y[:, i] == 1)[0] for i in range(num_classes)}
    
    # Create a large figure
    fig = plt.figure(figsize=(20, 4*num_classes))
    plt.suptitle('Examples for Each Class', fontsize=16)
    
    # Plot samples for each class
    for class_idx in range(num_classes):
        indices = class_indices[class_idx]
        if len(indices) == 0:
            continue
            
        # Take up to samples_per_class random samples
        sample_indices = np.random.choice(
            indices,
            min(samples_per_class, len(indices)),
            replace=False
        )
        
        for j, idx in enumerate(sample_indices):
            ax = plt.subplot(num_classes, samples_per_class, 
                           class_idx*samples_per_class + j + 1)
            segment = X[idx]
            
            # Plot all leads with different colors
            for lead in range(segment.shape[1]):
                ax.plot(segment[:, lead], alpha=0.5, linewidth=0.5)
            
            if j == 0:  # Only label the first plot in each row
                ax.set_ylabel(f'{label_names[class_idx]}', rotation=0, labelpad=40)
            
            ax.set_xticks([])
            ax.set_yticks([])
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def inspect_data(X, Y, label_names):
    """Detailed inspection of data contents"""
    print("\nInspecting data:")
    print(f"\nData shapes:")
    print(f" - X shape: {X.shape}")
    print(f" - Y shape: {Y.shape}")
    
    print(f"\nData types:")
    print(f" - X dtype: {X.dtype}")
    print(f" - Y dtype: {Y.dtype}")
    
    print(f"\nNumber of unique labels: {len(label_names)}")
    print(f"Label names: {label_names}")
    
    # Print some basic statistics
    print("\nData statistics:")
    print(f" - Number of samples: {len(X)}")
    print(f" - Time steps per sample: {X.shape[1]}")
    print(f" - Number of leads: {X.shape[2]}")
    
    # Count multilabel occurrences
    label_counts = Y.sum(axis=1)
    print(f"\nMultilabel statistics:")
    print(f" - Average labels per sample: {label_counts.mean():.2f}")
    print(f" - Max labels per sample: {label_counts.max()}")
    print(f" - Min labels per sample: {label_counts.min()}")

def analyze_lead_magnitudes(X):
    """Analyze the magnitudes of each lead across all samples."""
    num_leads = X.shape[2]
    lead_stats = []

    for lead_idx in range(num_leads):
        lead_data = X[:, :, lead_idx].flatten()
        mean = np.mean(lead_data)
        std = np.std(lead_data)
        min_val = np.min(lead_data)
        max_val = np.max(lead_data)
        lead_stats.append({
            'Lead': lead_idx + 1,
            'Mean': mean,
            'Std Dev': std,
            'Min': min_val,
            'Max': max_val
        })

    print("\nLead Magnitude Statistics:")
    for stats in lead_stats:
        print(f"Lead {stats['Lead']}: Mean = {stats['Mean']:.2f}, "
              f"Std Dev = {stats['Std Dev']:.2f}, Min = {stats['Min']:.2f}, Max = {stats['Max']:.2f}")

# Add a function to load raw ECG data
def load_raw_ecg_data(database_path, record_names, batch_size=100, max_records=None):
    """Load raw ECG signals from the database in batches with detailed diagnostics."""
    if max_records:
        record_names = record_names[:max_records]
    
    total_records = len(record_names)
    lead_stats = {i: {
        'sum': 0, 
        'sum_sq': 0, 
        'min': float('inf'), 
        'max': float('-inf'), 
        'count': 0,
        'null_count': 0,  # Track null/invalid values
        'valid_records': 0  # Track number of valid records per lead
    } for i in range(12)}
    
    problematic_records = []
    
    # Process records in batches
    for batch_start in range(0, total_records, batch_size):
        batch_end = min(batch_start + batch_size, total_records)
        print(f"Processing records {batch_start}-{batch_end} of {total_records}")
        
        # Process each record in the batch
        for record in record_names[batch_start:batch_end]:
            try:
                record_path = os.path.join(database_path, record)
                record_data = wfdb.rdrecord(record_path)
                signal = record_data.p_signal
                
                # Diagnostic print for first record
                if batch_start == 0 and record == record_names[0]:
                    print(f"\nFirst record shape: {signal.shape}")
                    print(f"First record data type: {signal.dtype}")
                    print("First record first few values per lead:")
                    for lead_idx in range(signal.shape[1]):
                        print(f"Lead {lead_idx + 1}: {signal[:5, lead_idx]}")
                
                # Update statistics for each lead
                for lead_idx in range(signal.shape[1]):
                    lead_data = signal[:, lead_idx]
                    
                    # Check for invalid values
                    valid_mask = ~np.isnan(lead_data) & ~np.isinf(lead_data)
                    valid_data = lead_data[valid_mask]
                    
                    if len(valid_data) > 0:
                        lead_stats[lead_idx]['sum'] += np.sum(valid_data)
                        lead_stats[lead_idx]['sum_sq'] += np.sum(np.square(valid_data))
                        lead_stats[lead_idx]['min'] = min(lead_stats[lead_idx]['min'], np.min(valid_data))
                        lead_stats[lead_idx]['max'] = max(lead_stats[lead_idx]['max'], np.max(valid_data))
                        lead_stats[lead_idx]['count'] += len(valid_data)
                        lead_stats[lead_idx]['valid_records'] += 1
                    
                    # Count invalid values
                    lead_stats[lead_idx]['null_count'] += np.sum(~valid_mask)
                    
                    # Check for problematic records
                    if np.sum(~valid_mask) > 0:
                        problematic_records.append((record, lead_idx + 1))
                
            except Exception as e:
                print(f"Error reading record {record}: {e}")
                continue
                
        # Force garbage collection after each batch
        import gc
        gc.collect()
    
    # Print diagnostic information
    print("\nDiagnostic Information:")
    print(f"Total records processed: {total_records}")
    print("\nPer-Lead Statistics:")
    for lead_idx in range(12):
        stats = lead_stats[lead_idx]
        print(f"\nLead {lead_idx + 1}:")
        print(f"  Valid records: {stats['valid_records']}/{total_records}")
        print(f"  Total valid points: {stats['count']}")
        print(f"  Invalid points: {stats['null_count']}")
        
        if stats['count'] > 0:
            mean = stats['sum'] / stats['count']
            variance = (stats['sum_sq'] / stats['count']) - (mean ** 2)
            std = np.sqrt(variance) if variance > 0 else 0
            print(f"  Mean = {mean:.2f}")
            print(f"  Std Dev = {std:.2f}")
            print(f"  Min = {stats['min']:.2f}")
            print(f"  Max = {stats['max']:.2f}")
        else:
            print("  No valid data points for statistics calculation")
    
    if problematic_records:
        print("\nProblematic Records (first 10):")
        for record, lead in problematic_records[:10]:
            print(f"Record: {record}, Lead: {lead}")
    
    # Calculate final statistics
    print("\nRaw Lead Magnitude Statistics:")
    for lead_idx in range(12):
        stats = lead_stats[lead_idx]
        if stats['count'] > 0:
            mean = stats['sum'] / stats['count']
            variance = (stats['sum_sq'] / stats['count']) - (mean ** 2)
            std = np.sqrt(variance) if variance > 0 else 0
            print(f"Lead {lead_idx + 1}: "
                  f"Mean = {mean:.2f}, "
                  f"Std Dev = {std:.2f}, "
                  f"Min = {stats['min']:.2f}, "
                  f"Max = {stats['max']:.2f}")
        else:
            print(f"Lead {lead_idx + 1}: No valid data points")

def main():
    # Set up paths for the raw ECG data
    database_path = os.path.join('a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0',
                                'a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0',
                                'WFDBRecords')
    
    print(f"Using database path: {database_path}")
    if not os.path.exists(database_path):
        print(f"Database path does not exist: {database_path}")
        return
    
    # Collect all record names
    record_names = []
    for root, dirs, files in os.walk(database_path):
        for file in files:
            if file.endswith('.hea'):
                record_path = os.path.join(root, file)
                record_name = os.path.splitext(os.path.relpath(record_path, database_path))[0]
                record_names.append(record_name)
    
    if not record_names:
        print("No ECG records found. Please check the database path.")
        return
    
    # Process data in batches
    batch_size = 1000  # Adjust this value based on your system's memory
    max_records = None  # Set to a number if you want to limit processing
    
    # Load and analyze the raw ECG data
    load_raw_ecg_data(database_path, record_names, batch_size, max_records)

if __name__ == '__main__':
    main() 