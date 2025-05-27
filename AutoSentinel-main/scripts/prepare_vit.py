"""
Vision Transformer preparation for network traffic visualization
"""
import json
import csv
import random
import math
from pathlib import Path

def create_traffic_visualizations():
    """Convert network flow data to 2D visualizations for ViT training"""
    print("Creating traffic visualizations for ViT training...")
    
    # Load processed data
    features_path = Path("data/processed/train_features.csv")
    labels_path = Path("data/processed/train_labels.csv")
    
    if not features_path.exists():
        print("[ERROR] No processed data found. Run preprocess_data.py first.")
        return
    
    # Read features
    features = []
    with open(features_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            features.append([float(x) for x in row])
    
    # Read labels
    labels = []
    with open(labels_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            labels.append(int(row[0]))
    
    print(f"Loaded {len(features)} feature vectors")
    
    # Create visualization directory
    vis_dir = Path("data/visualizations")
    vis_dir.mkdir(exist_ok=True)
    
    # Create simple 2D grid representations
    grid_size = 16  # 16x16 grid for visualization
    
    visualization_data = []
    
    for i, (feature_vector, label) in enumerate(zip(features[:1000], labels[:1000])):  # Limit for demo
        # Normalize features to 0-1 range
        min_val = min(feature_vector)
        max_val = max(feature_vector)
        
        if max_val > min_val:
            normalized = [(x - min_val) / (max_val - min_val) for x in feature_vector]
        else:
            normalized = [0.5] * len(feature_vector)
        
        # Pad or truncate to grid_size^2
        target_size = grid_size * grid_size
        if len(normalized) < target_size:
            # Pad with zeros
            normalized.extend([0.0] * (target_size - len(normalized)))
        else:
            # Truncate
            normalized = normalized[:target_size]
        
        # Reshape to grid
        grid = []
        for row in range(grid_size):
            grid_row = normalized[row * grid_size:(row + 1) * grid_size]
            grid.append(grid_row)
        
        # Save visualization metadata
        vis_data = {
            "id": i,
            "label": label,
            "grid": grid,
            "grid_size": grid_size,
            "original_features": len(feature_vector)
        }
        
        visualization_data.append(vis_data)
        
        # Save individual grid as CSV for inspection
        if i < 10:  # Save first 10 for inspection
            grid_path = vis_dir / f"grid_{i}_label_{label}.csv"
            with open(grid_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(grid)
    
    # Save all visualization data
    with open(vis_dir / "visualization_data.json", 'w') as f:
        json.dump(visualization_data, f, indent=2)
    
    print(f"Created {len(visualization_data)} traffic visualizations")
    print(f"   Grid size: {grid_size}x{grid_size}")
    print(f"   Saved to: {vis_dir}")

def create_vit_config():
    """Create ViT-specific configuration"""
    config = {
        "model": {
            "name": "TrafficViT",
            "type": "vision_transformer",
            "input_size": [16, 16],  # Grid size
            "patch_size": [4, 4],
            "num_classes": 8,
            "embed_dim": 768,
            "depth": 12,
            "num_heads": 12,
            "mlp_ratio": 4.0,
            "dropout": 0.1
        },
        "training": {
            "batch_size": 32,
            "learning_rate": 0.0001,
            "epochs": 100,
            "optimizer": "AdamW",
            "scheduler": "cosine",
            "warmup_epochs": 10
        },
        "data": {
            "augmentation": {
                "rotation": 0.1,
                "scaling": 0.1,
                "noise": 0.05
            }
        }
    }
    
    with open("configs/vit_config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    print("Created ViT configuration")

if __name__ == "__main__":
    create_traffic_visualizations()
    create_vit_config()
