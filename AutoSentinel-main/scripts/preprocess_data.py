"""
Data preprocessing pipeline for AutoSentinel
"""
import csv
import json
import random
from pathlib import Path

def load_config():
    """Load configuration"""
    with open("configs/config.json", "r") as f:
        return json.load(f)

def preprocess_cicids_data(input_path, output_dir):
    """Preprocess CICIDS2017 dataset for ML training"""
    print(f"Loading data from {input_path}")
    
    # Read CSV data
    data = []
    with open(input_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            data.append(row)
    
    print(f"Loaded {len(data)} samples")
    
    # Separate features and labels
    features = []
    labels = []
    
    for row in data:
        # Convert numeric features
        feature_row = []
        for i, value in enumerate(row[:-1]):  # All except last column (label)
            try:
                feature_row.append(float(value))
            except ValueError:
                feature_row.append(0.0)  # Handle non-numeric values
        
        features.append(feature_row)
        labels.append(row[-1])  # Last column is label
    
    # Create label mapping
    unique_labels = list(set(labels))
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    encoded_labels = [label_to_idx[label] for label in labels]
    
    # Split data (simple random split)
    n_samples = len(features)
    indices = list(range(n_samples))
    random.shuffle(indices)
    
    train_size = int(0.7 * n_samples)
    val_size = int(0.2 * n_samples)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save splits
    def save_split(indices, prefix):
        split_features = [features[i] for i in indices]
        split_labels = [encoded_labels[i] for i in indices]
        
        with open(output_path / f"{prefix}_features.csv", 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(split_features)
        
        with open(output_path / f"{prefix}_labels.csv", 'w', newline='') as f:
            writer = csv.writer(f)
            for label in split_labels:
                writer.writerow([label])
    
    save_split(train_indices, "train")
    save_split(val_indices, "val")
    save_split(test_indices, "test")
    
    # Save label mapping
    with open(output_path / "label_mapping.json", 'w') as f:
        json.dump(label_to_idx, f, indent=2)
    
    print(f"Preprocessed data saved to {output_path}")
    print(f"Training samples: {len(train_indices)}")
    print(f"Validation samples: {len(val_indices)}")
    print(f"Test samples: {len(test_indices)}")
    print(f"Label mapping: {label_to_idx}")

if __name__ == "__main__":
    preprocess_cicids_data("data/raw/sample_cicids2017.csv", "data/processed")
