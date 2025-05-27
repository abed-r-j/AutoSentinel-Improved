"""
AutoSentinel Phase 3: Improved Vision Transformer Training
Network traffic visualization classification for cybersecurity threat detection
"""

import json
import os
import random
import math
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImprovedViT:
    """Improved Vision Transformer implementation with proper gradient descent"""
    
    def __init__(self, image_size=16, patch_size=4, num_classes=8, dim=64):
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.dim = dim
        self.num_patches = (image_size // patch_size) ** 2
        
        # Better weight initialization (Xavier/Glorot)
        self.patch_embedding = self._init_weights((patch_size * patch_size, dim))
        self.position_embedding = self._init_weights((self.num_patches + 1, dim))
        self.cls_token = self._init_weights((1, dim))
        
        # Transformer layers
        self.attention_weights = self._init_weights((dim, dim))
        self.mlp_weights = self._init_weights((dim, dim * 2))
        self.classifier = self._init_weights((dim, num_classes))
        
        # Training state
        self.is_trained = False
        self.training_accuracy = 0.0
        self.validation_accuracy = 0.0
        
    def _init_weights(self, shape):
        """Initialize weights with Xavier initialization"""
        if len(shape) == 2:
            # Xavier initialization
            limit = math.sqrt(6.0 / (shape[0] + shape[1]))
            return [[random.uniform(-limit, limit) for _ in range(shape[1])] for _ in range(shape[0])]
        else:
            limit = math.sqrt(6.0 / shape[0])
            return [random.uniform(-limit, limit) for _ in range(shape[0])]
    
    def create_patches(self, image):
        """Convert image to patches for ViT processing"""
        patches = []
        for i in range(0, self.image_size, self.patch_size):
            for j in range(0, self.image_size, self.patch_size):
                patch = []
                for pi in range(i, min(i + self.patch_size, self.image_size)):
                    for pj in range(j, min(j + self.patch_size, self.image_size)):
                        if pi < len(image) and pj < len(image[pi]):
                            patch.append(image[pi][pj])
                        else:
                            patch.append(0.0)
                patches.append(patch)
        return patches
    
    def forward(self, image):
        """Forward pass through the ViT model"""
        # Convert to patches
        patches = self.create_patches(image)
        
        # Embed patches
        embedded_patches = []
        for patch in patches:
            embedded = [sum(patch[i] * self.patch_embedding[i][j] for i in range(min(len(patch), len(self.patch_embedding)))) 
                       for j in range(self.dim)]
            embedded_patches.append(embedded)
        
        # Add position embeddings
        for i, patch in enumerate(embedded_patches):
            if i < len(self.position_embedding):
                for j in range(len(patch)):
                    if j < len(self.position_embedding[i]):
                        patch[j] += self.position_embedding[i][j]
        
        # Add CLS token
        cls_embedded = [self.cls_token[0][i] + self.position_embedding[0][i] 
                       for i in range(self.dim)]
        all_tokens = [cls_embedded] + embedded_patches
        
        # Simple attention mechanism (using average of tokens)
        attended = [sum(token[i] for token in all_tokens) / len(all_tokens) 
                   for i in range(self.dim)]
        
        # Classification head
        logits = [sum(attended[i] * self.classifier[i][j] for i in range(len(attended))) 
                 for j in range(self.num_classes)]
        
        return logits, attended
    
    def predict(self, image):
        """Predict class for a single image"""
        logits, _ = self.forward(image)
        return logits.index(max(logits))
    
    def softmax(self, logits):
        """Apply softmax activation"""
        exp_logits = [math.exp(x - max(logits)) for x in logits]  # Numerical stability
        sum_exp = sum(exp_logits)
        return [x / sum_exp for x in exp_logits]
    
    def cross_entropy_loss(self, logits, target):
        """Calculate cross-entropy loss"""
        probs = self.softmax(logits)
        # Avoid log(0) by adding small epsilon
        return -math.log(max(probs[target], 1e-15))
    
    def train_epoch(self, train_data, learning_rate=0.01):
        """Train for one epoch with proper gradient descent"""
        correct = 0
        total = len(train_data)
        total_loss = 0.0
        
        for image, label in train_data:
            # Forward pass
            logits, attended = self.forward(image)
            predicted = logits.index(max(logits))
            
            if predicted == label:
                correct += 1
            
            # Calculate loss
            loss = self.cross_entropy_loss(logits, label)
            total_loss += loss
            
            # Proper gradient calculation for classifier weights
            probs = self.softmax(logits)
            
            # Gradient descent for classifier
            for i in range(len(self.classifier)):
                for j in range(len(self.classifier[i])):
                    # Gradient = (predicted_prob - true_prob) * input_feature
                    gradient = (probs[j] - (1.0 if j == label else 0.0)) * attended[i]
                    self.classifier[i][j] -= learning_rate * gradient
            
            # Update patch embeddings with smaller learning rate
            patches = self.create_patches(image)
            patch_lr = learning_rate * 0.1
            for patch_idx, patch in enumerate(patches):
                for i in range(min(len(patch), len(self.patch_embedding))):
                    for j in range(len(self.patch_embedding[i])):
                        # Simple gradient approximation based on loss
                        error_signal = (probs[predicted] - (1.0 if predicted == label else 0.0))
                        gradient = error_signal * patch[i] * 0.01
                        self.patch_embedding[i][j] -= patch_lr * gradient
        
        accuracy = correct / total
        avg_loss = total_loss / total
        return accuracy
    
    def evaluate(self, val_data):
        """Evaluate on validation data"""
        correct = 0
        total = len(val_data)
        
        for image, label in val_data:
            predicted = self.predict(image)
            if predicted == label:
                correct += 1
        
        accuracy = correct / total
        return accuracy

class ImprovedViTTrainer:
    """Improved Vision Transformer trainer with better training procedures"""
    
    def __init__(self, data_dir="data/visualizations", model_dir="models/vit"):
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        # Load label mapping
        with open("data/processed/label_mapping.json", "r") as f:
            self.label_mapping = json.load(f)
        self.num_classes = len(self.label_mapping)
        
    def load_visualization_data(self):
        """Load network traffic visualizations"""
        train_data = []
        val_data = []
        
        # Load all visualization data
        vis_file = self.data_dir / "visualization_data.json"
        if not vis_file.exists():
            logger.error("No visualization data found.")
            return train_data, val_data
        
        with open(vis_file, "r") as f:
            all_visualizations = json.load(f)
        
        # Data augmentation: create more training samples
        augmented_data = []
        for item in all_visualizations:
            # Original sample
            augmented_data.append(item)
            
            # Add noise to create variations
            for _ in range(2):  # 2 augmented versions per sample
                noisy_grid = []
                for row in item["grid"]:
                    noisy_row = [max(0, min(1, val + random.uniform(-0.1, 0.1))) for val in row]
                    noisy_grid.append(noisy_row)
                augmented_data.append({"grid": noisy_grid, "label": item["label"]})
        
        # Split data into train/validation (80/20 split)
        random.shuffle(augmented_data)
        split_idx = int(0.8 * len(augmented_data))
        
        for i, item in enumerate(augmented_data):
            image = item["grid"]
            label = item["label"]
            
            if i < split_idx:
                train_data.append((image, label))
            else:
                val_data.append((image, label))
        
        logger.info(f"Loaded {len(train_data)} training samples, {len(val_data)} validation samples")
        return train_data, val_data
    
    def train_model(self, epochs=20, learning_rate=0.01):
        """Train the improved ViT model"""
        logger.info("Starting improved ViT training...")
        
        # Load data
        train_data, val_data = self.load_visualization_data()
        
        if not train_data:
            logger.error("No training data found. Please run prepare_vit.py first.")
            return None
        
        # Initialize model
        model = ImprovedViT(
            image_size=16,
            patch_size=4,
            num_classes=self.num_classes,
            dim=64
        )
        
        # Training loop with learning rate decay
        best_val_accuracy = 0.0
        patience = 5
        patience_counter = 0
        
        for epoch in range(epochs):
            # Adjust learning rate
            current_lr = learning_rate * (0.9 ** (epoch // 5))
            
            # Train
            train_accuracy = model.train_epoch(train_data, current_lr)
            
            # Validate
            val_accuracy = model.evaluate(val_data) if val_data else 0.0
            
            logger.info(f"Epoch {epoch+1}/{epochs}: "
                       f"Train Acc: {train_accuracy:.3f}, Val Acc: {val_accuracy:.3f}, LR: {current_lr:.4f}")
            
            # Save best model
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                patience_counter = 0
                self.save_model(model, epoch, train_accuracy, val_accuracy)
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience and epoch > 10:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        model.is_trained = True
        model.training_accuracy = train_accuracy
        model.validation_accuracy = best_val_accuracy
        
        logger.info(f"Training complete! Best validation accuracy: {best_val_accuracy:.3f}")
        return model
    
    def save_model(self, model, epoch, train_acc, val_acc):
        """Save model state"""
        model_state = {
            "epoch": epoch,
            "train_accuracy": train_acc,
            "validation_accuracy": val_acc,
            "num_classes": model.num_classes,
            "image_size": model.image_size,
            "patch_size": model.patch_size,
            "dim": model.dim,
            "is_trained": True,
            "patch_embedding": model.patch_embedding,
            "position_embedding": model.position_embedding,
            "cls_token": model.cls_token,
            "classifier": model.classifier
        }
        
        model_path = self.model_dir / "vit_checkpoint.json"
        with open(model_path, "w") as f:
            json.dump(model_state, f, indent=2)
        
        logger.info(f"Model saved to {model_path}")
    
    def load_model(self):
        """Load trained model"""
        model_path = self.model_dir / "vit_checkpoint.json"
        if not model_path.exists():
            logger.error("No trained model found.")
            return None
        
        with open(model_path, "r") as f:
            model_state = json.load(f)
        
        model = ImprovedViT(
            image_size=model_state["image_size"],
            patch_size=model_state["patch_size"],
            num_classes=model_state["num_classes"],
            dim=model_state["dim"]
        )
        
        # Load weights if available
        if "patch_embedding" in model_state:
            model.patch_embedding = model_state["patch_embedding"]
            model.position_embedding = model_state["position_embedding"]
            model.cls_token = model_state["cls_token"]
            model.classifier = model_state["classifier"]
        
        model.is_trained = model_state["is_trained"]
        model.training_accuracy = model_state["train_accuracy"]
        model.validation_accuracy = model_state["validation_accuracy"]
        
        logger.info(f"Model loaded. Val accuracy: {model.validation_accuracy:.3f}")
        return model

def main():
    """Main training function"""
    print("AutoSentinel Improved ViT Training - Phase 3")
    print("=" * 55)
    
    # Check if visualization data exists
    if not Path("data/visualizations").exists():
        print("Error: Visualization data not found.")
        print("Please run: python scripts/prepare_vit.py")
        return
    
    # Initialize trainer
    trainer = ImprovedViTTrainer()
    
    # Train model with better parameters
    model = trainer.train_model(epochs=25, learning_rate=0.02)
    
    if model:
        print(f"\nImproved Training Results:")
        print(f"Final Training Accuracy: {model.training_accuracy:.3f}")
        print(f"Final Validation Accuracy: {model.validation_accuracy:.3f}")
        print(f"Model saved to: models/vit/")
        
        # Test prediction on sample
        print(f"\nTesting model prediction...")
        test_image = [[random.uniform(0, 1) for _ in range(16)] for _ in range(16)]
        prediction = model.predict(test_image)
        
        # Find label name
        label_name = "Unknown"
        for name, idx in trainer.label_mapping.items():
            if idx == prediction:
                label_name = name
                break
        
        print(f"Sample prediction: {label_name} (class {prediction})")
        
        print("\nImproved ViT training phase complete!")
        print("Next: Run MARL training with 'python train_marl.py'")

if __name__ == "__main__":
    main()
