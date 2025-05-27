"""
AutoSentinel Phase 1: Development Environment Setup
Real dataset integration and enhanced ML pipeline implementation
"""

import os
import urllib.request
import zipfile
import pandas as pd
from pathlib import Path
import json
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetManager:
    """Manages dataset downloading, preprocessing, and preparation"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # CICIDS2017 dataset information
        self.datasets = {
            "cicids2017": {
                "name": "CICIDS2017",
                "description": "Canadian Institute for Cybersecurity Intrusion Detection System Dataset 2017",
                "urls": [
                    "https://www.unb.ca/cic/datasets/ids-2017.html"  # Official dataset page
                ],
                "files": [
                    "Monday-WorkingHours.pcap_ISCX.csv",
                    "Tuesday-WorkingHours.pcap_ISCX.csv", 
                    "Wednesday-workingHours.pcap_ISCX.csv",
                    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
                    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
                    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
                    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
                    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"
                ],
                "features": [
                    "Destination Port", "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
                    "Total Length of Fwd Packets", "Total Length of Bwd Packets", "Fwd Packet Length Max",
                    "Fwd Packet Length Min", "Fwd Packet Length Mean", "Fwd Packet Length Std",
                    "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean",
                    "Bwd Packet Length Std", "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean",
                    "Flow IAT Std", "Flow IAT Max", "Flow IAT Min", "Fwd IAT Total", "Fwd IAT Mean",
                    "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min", "Bwd IAT Total", "Bwd IAT Mean",
                    "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min", "Fwd PSH Flags", "Bwd PSH Flags",
                    "Fwd URG Flags", "Bwd URG Flags", "Fwd Header Length", "Bwd Header Length",
                    "Fwd Packets/s", "Bwd Packets/s", "Min Packet Length", "Max Packet Length",
                    "Packet Length Mean", "Packet Length Std", "Packet Length Variance", "FIN Flag Count",
                    "SYN Flag Count", "RST Flag Count", "PSH Flag Count", "ACK Flag Count",
                    "URG Flag Count", "CWE Flag Count", "ECE Flag Count", "Down/Up Ratio",
                    "Average Packet Size", "Avg Fwd Segment Size", "Avg Bwd Segment Size",
                    "Fwd Header Length.1", "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk",
                    "Fwd Avg Bulk Rate", "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk",
                    "Bwd Avg Bulk Rate", "Subflow Fwd Packets", "Subflow Fwd Bytes",
                    "Subflow Bwd Packets", "Subflow Bwd Bytes", "Init_Win_bytes_forward",
                    "Init_Win_bytes_backward", "act_data_pkt_fwd", "min_seg_size_forward",
                    "Active Mean", "Active Std", "Active Max", "Active Min", "Idle Mean",
                    "Idle Std", "Idle Max", "Idle Min", "Label"
                ]
            }
        }
    
    def setup_environment(self):
        """Set up the complete development environment"""
        print("🔧 Setting up AutoSentinel Phase 1 Development Environment")
        print("=" * 60)
        
        # Create directory structure
        directories = [
            "data/raw",
            "data/processed", 
            "data/visualizations",
            "models/vit",
            "models/marl",
            "logs",
            "configs",
            "notebooks",
            "scripts"
        ]
        
        for directory in directories:
            dir_path = Path(directory)
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {directory}")
        
        # Create configuration files
        self._create_config_files()
        
        # Generate sample CICIDS2017-like data for development
        self._generate_sample_data()
        
        print("✅ Development environment setup complete!")
        return True
    
    def _create_config_files(self):
        """Create configuration files for the project"""
        
        # Main configuration
        config = {
            "project": {
                "name": "AutoSentinel",
                "version": "1.0.0",
                "description": "Autonomous Multi-Agent AI Cybersecurity Orchestrator"
            },
            "data": {
                "raw_data_path": "data/raw",
                "processed_data_path": "data/processed",
                "batch_size": 32,
                "validation_split": 0.2,
                "test_split": 0.1
            },
            "models": {
                "vit": {
                    "image_size": 224,
                    "patch_size": 16,
                    "num_classes": 8,  # Number of threat types
                    "dim": 768,
                    "depth": 12,
                    "heads": 12,
                    "mlp_dim": 3072
                },
                "marl": {
                    "num_agents": 3,
                    "state_dim": 100,
                    "action_dim": 6,
                    "learning_rate": 0.001,
                    "gamma": 0.99,
                    "epsilon": 0.1
                }
            },
            "training": {
                "epochs": 100,
                "early_stopping_patience": 10,
                "learning_rate": 0.0001,
                "optimizer": "Adam",
                "loss_function": "CrossEntropyLoss"
            }
        }
        
        config_path = Path("configs/config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Created configuration file: {config_path}")
        
        # Dataset configuration
        dataset_config = self.datasets["cicids2017"]
        dataset_path = Path("configs/dataset_config.json")
        with open(dataset_path, 'w') as f:
            json.dump(dataset_config, f, indent=2)
        
        logger.info(f"Created dataset configuration: {dataset_path}")
    
    def _generate_sample_data(self):
        """Generate sample CICIDS2017-like data for development and testing"""
        import random
        
        print("📊 Generating sample CICIDS2017-like dataset...")
        
        # Define attack labels from CICIDS2017
        attack_labels = [
            "BENIGN",
            "DDoS", 
            "PortScan",
            "Bot",
            "Infiltration",
            "Web Attack – Brute Force",
            "Web Attack – XSS", 
            "Web Attack – Sql Injection"
        ]
        
        # Generate sample data with realistic network flow features
        num_samples = 10000
        data = []
        
        for i in range(num_samples):
            # Simulate different attack patterns
            label = random.choice(attack_labels)
            
            if label == "BENIGN":
                # Normal traffic characteristics
                flow_duration = random.randint(100, 10000)
                total_fwd_packets = random.randint(1, 100)
                total_bwd_packets = random.randint(1, 100)
                flow_bytes_per_sec = random.uniform(100, 10000)
                flow_packets_per_sec = random.uniform(1, 100)
            
            elif label == "DDoS":
                # DDoS attack characteristics - high packet rate, short duration
                flow_duration = random.randint(1, 1000)
                total_fwd_packets = random.randint(100, 10000)
                total_bwd_packets = random.randint(0, 10)
                flow_bytes_per_sec = random.uniform(50000, 1000000)
                flow_packets_per_sec = random.uniform(1000, 100000)
            
            elif label == "PortScan":
                # Port scan characteristics - many connections, small packets
                flow_duration = random.randint(1, 100)
                total_fwd_packets = random.randint(1, 10)
                total_bwd_packets = random.randint(0, 5)
                flow_bytes_per_sec = random.uniform(10, 1000)
                flow_packets_per_sec = random.uniform(10, 1000)
            
            else:
                # Other attack types - mixed characteristics
                flow_duration = random.randint(100, 5000)
                total_fwd_packets = random.randint(10, 500)
                total_bwd_packets = random.randint(10, 500)
                flow_bytes_per_sec = random.uniform(1000, 50000)
                flow_packets_per_sec = random.uniform(10, 1000)
            
            # Create a sample with key features
            sample = {
                "Flow Duration": flow_duration,
                "Total Fwd Packets": total_fwd_packets,
                "Total Backward Packets": total_bwd_packets,
                "Total Length of Fwd Packets": total_fwd_packets * random.randint(64, 1500),
                "Total Length of Bwd Packets": total_bwd_packets * random.randint(64, 1500),
                "Flow Bytes/s": flow_bytes_per_sec,
                "Flow Packets/s": flow_packets_per_sec,
                "Flow IAT Mean": random.uniform(0, 1000),
                "Flow IAT Std": random.uniform(0, 500),
                "Fwd Packet Length Mean": random.uniform(64, 1500),
                "Bwd Packet Length Mean": random.uniform(64, 1500),
                "Packet Length Mean": random.uniform(64, 1500),
                "Average Packet Size": random.uniform(64, 1500),
                "SYN Flag Count": random.randint(0, 10),
                "ACK Flag Count": random.randint(0, 100),
                "FIN Flag Count": random.randint(0, 10),
                "RST Flag Count": random.randint(0, 5),
                "Destination Port": random.choice([80, 443, 22, 23, 53, 21, 25, 3389]),
                "Label": label
            }
            
            data.append(sample)
        
        # Create DataFrame and save
        df = pd.DataFrame(data)
        output_path = Path("data/raw/sample_cicids2017.csv")
        df.to_csv(output_path, index=False)
        
        print(f"✅ Generated sample dataset with {num_samples} samples")
        print(f"   Saved to: {output_path}")
        print(f"   Attack distribution:")
        print(df['Label'].value_counts().to_string())
        
        return df

def create_development_scripts():
    """Create utility scripts for development"""
    
    # Data preprocessing script
    preprocessing_script = '''"""
Data preprocessing pipeline for AutoSentinel
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path

def preprocess_cicids_data(input_path, output_dir):
    """Preprocess CICIDS2017 dataset for ML training"""
    print(f"Loading data from {input_path}")
    df = pd.read_csv(input_path)
    
    # Handle missing values
    df = df.fillna(0)
    
    # Remove infinite values
    df = df.replace([np.inf, -np.inf], 0)
    
    # Separate features and labels
    X = df.drop(['Label'], axis=1)
    y = df['Label']
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_scaled, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    # Save processed data
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    np.save(output_path / "X_train.npy", X_train)
    np.save(output_path / "X_val.npy", X_val) 
    np.save(output_path / "X_test.npy", X_test)
    np.save(output_path / "y_train.npy", y_train)
    np.save(output_path / "y_val.npy", y_val)
    np.save(output_path / "y_test.npy", y_test)
    
    # Save encoders
    joblib.dump(scaler, output_path / "scaler.pkl")
    joblib.dump(label_encoder, output_path / "label_encoder.pkl")
    
    print(f"Preprocessed data saved to {output_path}")
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Validation samples: {X_val.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

if __name__ == "__main__":
    preprocess_cicids_data("data/raw/sample_cicids2017.csv", "data/processed")
'''
    
    with open("scripts/preprocess_data.py", "w") as f:
        f.write(preprocessing_script)
    
    # Training script template
    training_script = '''"""
Training pipeline for AutoSentinel models
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import json
from pathlib import Path

def load_config():
    """Load training configuration"""
    with open("configs/config.json", "r") as f:
        return json.load(f)

def load_data():
    """Load preprocessed data"""
    data_dir = Path("data/processed")
    
    X_train = np.load(data_dir / "X_train.npy")
    X_val = np.load(data_dir / "X_val.npy")
    y_train = np.load(data_dir / "y_train.npy")
    y_val = np.load(data_dir / "y_val.npy")
    
    return X_train, X_val, y_train, y_val

def create_data_loaders(X_train, X_val, y_train, y_val, batch_size=32):
    """Create PyTorch data loaders"""
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train), 
        torch.LongTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val), 
        torch.LongTensor(y_val)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

class SimpleThreatClassifier(nn.Module):
    """Simple neural network for threat classification"""
    
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(), 
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        return self.network(x)

def train_model():
    """Train the threat detection model"""
    config = load_config()
    
    # Load data
    X_train, X_val, y_train, y_val = load_data()
    train_loader, val_loader = create_data_loaders(
        X_train, X_val, y_train, y_val,
        batch_size=config["data"]["batch_size"]
    )
    
    # Initialize model
    input_dim = X_train.shape[1]
    num_classes = len(np.unique(y_train))
    model = SimpleThreatClassifier(input_dim, num_classes)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
    
    # Training loop
    print("Starting training...")
    for epoch in range(config["training"]["epochs"]):
        model.train()
        train_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        accuracy = 100 * correct / total
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train Loss: {train_loss/len(train_loader):.4f}, "
                  f"Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {accuracy:.2f}%")
    
    # Save model
    torch.save(model.state_dict(), "models/threat_classifier.pth")
    print("Model saved successfully!")

if __name__ == "__main__":
    train_model()
'''
    
    with open("scripts/train_model.py", "w") as f:
        f.write(training_script)
    
    logger.info("Created development scripts in scripts/ directory")

def main():
    """Main function to set up Phase 1 development environment"""
    
    # Initialize dataset manager
    dm = DatasetManager()
    
    # Set up environment
    dm.setup_environment()
    
    # Create development scripts
    create_development_scripts()
    
    # Create README for Phase 1
    readme_content = """# AutoSentinel Phase 1: Development Environment

## Overview
This directory contains the complete development environment for AutoSentinel Phase 1,
including real dataset integration and enhanced ML pipeline implementation.

## Directory Structure
```
├── data/
│   ├── raw/              # Raw datasets (CICIDS2017)
│   ├── processed/        # Preprocessed and cleaned data
│   └── visualizations/   # Data visualization outputs
├── models/
│   ├── vit/             # Vision Transformer models
│   └── marl/            # Multi-Agent RL models  
├── configs/             # Configuration files
├── scripts/             # Development and training scripts
├── notebooks/           # Jupyter notebooks for analysis
└── logs/               # Training and system logs

## Quick Start

1. **Data Preprocessing:**
   ```bash
   python scripts/preprocess_data.py
   ```

2. **Model Training:**
   ```bash
   python scripts/train_model.py
   ```

3. **Visualization:**
   ```bash
   python scripts/visualize_data.py
   ```

## Dataset Information
- **CICIDS2017**: Canadian Institute for Cybersecurity IDS Dataset
- **Features**: 78 network flow features
- **Classes**: 8 attack types (BENIGN, DDoS, PortScan, Bot, etc.)
- **Samples**: 10,000 synthetic samples for development

## Next Steps
- Download real CICIDS2017 dataset
- Implement Vision Transformer for traffic pattern recognition
- Build MARL environment for response optimization
- Deploy multi-agent coordination system
"""
    
    with open("README_Phase1.md", "w") as f:
        f.write(readme_content)
    
    print("\n🎉 AutoSentinel Phase 1 Development Environment Setup Complete!")
    print("\n📋 Next Steps:")
    print("   1. Run data preprocessing: python scripts/preprocess_data.py")
    print("   2. Download real CICIDS2017 dataset from UNB website")
    print("   3. Begin ViT model implementation for traffic visualization")
    print("   4. Implement MARL environment for response optimization")

if __name__ == "__main__":
    main()
