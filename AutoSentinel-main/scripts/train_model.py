"""
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
