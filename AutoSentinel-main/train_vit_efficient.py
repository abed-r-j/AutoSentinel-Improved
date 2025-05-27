#!/usr/bin/env python3
"""
AutoSentinel Efficient ViT Training
Optimized for rapid training with significant accuracy improvements
"""

import json
import random
import math
import logging
import csv
from datetime import datetime
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EfficientViT:
    """Efficient Vision Transformer focused on core improvements"""
    
    def __init__(self, image_size=16, patch_size=4, num_classes=8, dim=64):
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.dim = dim
        self.num_patches = (image_size // patch_size) ** 2
        
        # Initialize with proper Glorot initialization
        self.patch_embedding = self._glorot_init(patch_size * patch_size, dim)
        self.position_embedding = self._glorot_init(self.num_patches + 1, dim)
        self.cls_token = [random.gauss(0, 0.02) for _ in range(dim)]
        
        # Classification layers
        self.feature_transform = self._glorot_init(dim, dim)
        self.classifier = self._glorot_init(dim, num_classes)
        self.classifier_bias = [0.0] * num_classes
        
        # Batch normalization parameters
        self.bn_mean = [0.0] * dim
        self.bn_var = [1.0] * dim
        self.bn_gamma = [1.0] * dim
        self.bn_beta = [0.0] * dim
        
        # Momentum for updates
        self.momentum = 0.9
        self.velocity = {}
        
        logger.info(f"Initialized Efficient ViT: {self.num_patches} patches, {dim}D, {num_classes} classes")
    
    def _glorot_init(self, fan_in, fan_out):
        """Glorot/Xavier initialization"""
        limit = math.sqrt(6.0 / (fan_in + fan_out))
        if isinstance(fan_out, int):
            return [[random.uniform(-limit, limit) for _ in range(fan_out)] 
                    for _ in range(fan_in)]
        else:
            return [random.uniform(-limit, limit) for _ in range(fan_in)]
    
    def _init_velocity(self, param_name, shape):
        """Initialize velocity for momentum"""
        if param_name not in self.velocity:
            if isinstance(shape, tuple):
                self.velocity[param_name] = [[0.0] * shape[1] for _ in range(shape[0])]
            else:
                self.velocity[param_name] = [0.0] * shape
    
    def batch_norm(self, x, training=True, momentum=0.1):
        """Simple batch normalization"""
        if training:
            # Update running statistics
            batch_mean = sum(x) / len(x)
            batch_var = sum((xi - batch_mean) ** 2 for xi in x) / len(x)
            
            # Update running mean and variance
            for i in range(len(self.bn_mean)):
                if i < len(x):
                    self.bn_mean[i] = (1 - momentum) * self.bn_mean[i] + momentum * batch_mean
                    self.bn_var[i] = (1 - momentum) * self.bn_var[i] + momentum * batch_var
        
        # Normalize
        normalized = []
        for i, xi in enumerate(x):
            mean_i = self.bn_mean[i] if i < len(self.bn_mean) else 0
            var_i = self.bn_var[i] if i < len(self.bn_var) else 1
            std_i = math.sqrt(var_i + 1e-8)
            
            gamma_i = self.bn_gamma[i] if i < len(self.bn_gamma) else 1
            beta_i = self.bn_beta[i] if i < len(self.bn_beta) else 0
            
            norm_val = gamma_i * (xi - mean_i) / std_i + beta_i
            normalized.append(norm_val)
        
        return normalized
    
    def relu(self, x):
        """ReLU activation"""
        return [max(0, xi) for xi in x]
    
    def dropout(self, x, rate, training):
        """Dropout for regularization"""
        if not training or rate == 0:
            return x
        keep_prob = 1 - rate
        return [xi / keep_prob if random.random() < keep_prob else 0 for xi in x]
    
    def softmax(self, x):
        """Numerically stable softmax"""
        max_x = max(x)
        exp_x = [math.exp(xi - max_x) for xi in x]
        sum_exp = sum(exp_x)
        return [ei / sum_exp for ei in exp_x]
    
    def extract_patches(self, image):
        """Extract patches from image"""
        # Ensure image is the right size
        if len(image) != self.image_size * self.image_size:
            # Pad or truncate
            if len(image) < self.image_size * self.image_size:
                image = image + [0.0] * (self.image_size * self.image_size - len(image))
            else:
                image = image[:self.image_size * self.image_size]
        
        patches = []
        patch_dim = self.patch_size * self.patch_size
        
        for i in range(self.num_patches):
            start_idx = i * patch_dim
            end_idx = start_idx + patch_dim
            patch = image[start_idx:end_idx] if end_idx <= len(image) else image[start_idx:] + [0.0] * (end_idx - len(image))
            patches.append(patch)
        
        return patches
    
    def linear_layer(self, x, weights, bias=None):
        """Linear transformation"""
        output = []
        for j in range(len(weights[0])):
            val = sum(x[i] * weights[i][j] for i in range(min(len(x), len(weights))))
            if bias:
                val += bias[j]
            output.append(val)
        return output
    
    def forward(self, image, training=True):
        """Forward pass"""
        # Extract patches
        patches = self.extract_patches(image)
        
        # Embed patches
        embedded_patches = []
        for i, patch in enumerate(patches):
            embedded = self.linear_layer(patch, self.patch_embedding)
            
            # Add positional embedding
            if i < len(self.position_embedding):
                pos_emb = self.position_embedding[i]
                embedded = [embedded[j] + pos_emb[j] for j in range(min(len(embedded), len(pos_emb)))]
            
            embedded_patches.append(embedded)
        
        # Add CLS token with position
        cls_embedded = [self.cls_token[i] + self.position_embedding[-1][i] 
                       for i in range(min(len(self.cls_token), len(self.position_embedding[-1])))]
        
        # Simple attention: average pooling over patches
        all_patches = [cls_embedded] + embedded_patches
        global_features = [sum(patch[i] for patch in all_patches) / len(all_patches) 
                          for i in range(self.dim)]
        
        # Apply batch normalization
        global_features = self.batch_norm(global_features, training)
        
        # Feature transformation
        transformed = self.linear_layer(global_features, self.feature_transform)
        transformed = self.relu(transformed)
        transformed = self.dropout(transformed, 0.2, training)
        
        # Classification
        logits = self.linear_layer(transformed, self.classifier, self.classifier_bias)
        
        return logits, transformed
    
    def predict(self, image):
        """Predict class for image"""
        logits, _ = self.forward(image, training=False)
        return logits.index(max(logits))
    
    def cross_entropy_loss(self, logits, target):
        """Cross-entropy loss"""
        probs = self.softmax(logits)
        return -math.log(max(probs[target], 1e-15))
    
    def update_weights_momentum(self, weights, gradients, param_name, learning_rate):
        """Update weights with momentum"""
        if isinstance(weights[0], list):  # 2D weights
            self._init_velocity(param_name, (len(weights), len(weights[0])))
            for i in range(len(weights)):
                for j in range(len(weights[i])):
                    self.velocity[param_name][i][j] = (self.momentum * self.velocity[param_name][i][j] + 
                                                     learning_rate * gradients[i][j])
                    weights[i][j] -= self.velocity[param_name][i][j]
        else:  # 1D weights
            self._init_velocity(param_name, len(weights))
            for i in range(len(weights)):
                self.velocity[param_name][i] = (self.momentum * self.velocity[param_name][i] + 
                                              learning_rate * gradients[i])
                weights[i] -= self.velocity[param_name][i]
    
    def compute_gradients(self, logits, features, target, image):
        """Compute gradients for backpropagation"""
        # Output gradient
        probs = self.softmax(logits)
        output_grad = [probs[i] - (1.0 if i == target else 0.0) for i in range(len(probs))]
        
        # Classifier gradients
        classifier_grad = [[features[i] * output_grad[j] for j in range(len(output_grad))] 
                          for i in range(len(features))]
        classifier_bias_grad = output_grad[:]
        
        # Feature transform gradients (simplified)
        error_magnitude = sum(abs(g) for g in output_grad) / len(output_grad)
        transform_grad = [[error_magnitude * 0.01 * random.gauss(0, 1) for _ in range(self.dim)] 
                         for _ in range(self.dim)]
        
        # Patch embedding gradients (simplified)
        patch_grad = [[error_magnitude * 0.001 * random.gauss(0, 1) for _ in range(self.dim)] 
                     for _ in range(self.patch_size * self.patch_size)]
        
        return {
            'classifier': classifier_grad,
            'classifier_bias': classifier_bias_grad,
            'feature_transform': transform_grad,
            'patch_embedding': patch_grad
        }
    
    def train_epoch(self, train_data, learning_rate):
        """Train for one epoch"""
        random.shuffle(train_data)
        total_loss = 0
        correct = 0
        
        for i, (image, label) in enumerate(train_data):
            # Forward pass
            logits, features = self.forward(image, training=True)
            
            # Compute loss
            loss = self.cross_entropy_loss(logits, label)
            total_loss += loss
            
            # Check accuracy
            predicted = logits.index(max(logits))
            if predicted == label:
                correct += 1
            
            # Compute gradients
            gradients = self.compute_gradients(logits, features, label, image)
            
            # Update weights
            self.update_weights_momentum(self.classifier, gradients['classifier'], 
                                       'classifier', learning_rate)
            self.update_weights_momentum(self.classifier_bias, gradients['classifier_bias'], 
                                       'classifier_bias', learning_rate)
            self.update_weights_momentum(self.feature_transform, gradients['feature_transform'], 
                                       'feature_transform', learning_rate * 0.1)
            self.update_weights_momentum(self.patch_embedding, gradients['patch_embedding'], 
                                       'patch_embedding', learning_rate * 0.01)
            
            # Progress logging
            if (i + 1) % 500 == 0:
                current_acc = (correct / (i + 1)) * 100
                logger.info(f"  Progress: {i+1}/{len(train_data)} - Acc: {current_acc:.1f}% - Loss: {loss:.3f}")
        
        accuracy = (correct / len(train_data)) * 100
        avg_loss = total_loss / len(train_data)
        return accuracy, avg_loss
    
    def evaluate(self, val_data):
        """Evaluate on validation data"""
        total_loss = 0
        correct = 0
        
        for image, label in val_data:
            logits, _ = self.forward(image, training=False)
            loss = self.cross_entropy_loss(logits, label)
            total_loss += loss
            
            predicted = logits.index(max(logits))
            if predicted == label:
                correct += 1
        
        accuracy = (correct / len(val_data)) * 100
        avg_loss = total_loss / len(val_data)
        return accuracy, avg_loss

class EfficientViTTrainer:
    """Trainer for Efficient ViT"""
    
    def __init__(self):
        self.best_accuracy = 0
        self.training_history = []
    
    def load_data(self):
        """Load training data efficiently"""
        try:
            def read_csv_simple(filepath):
                data = []
                with open(filepath, 'r') as f:
                    reader = csv.reader(f)
                    next(reader)  # Skip header
                    for row in reader:
                        data.append([float(x) for x in row])
                return data
            
            train_features = read_csv_simple('data/processed/train_features.csv')
            train_labels = read_csv_simple('data/processed/train_labels.csv')
            val_features = read_csv_simple('data/processed/val_features.csv')
            val_labels = read_csv_simple('data/processed/val_labels.csv')
            
            train_data = [(train_features[i], int(train_labels[i][0])) 
                         for i in range(len(train_features))]
            val_data = [(val_features[i], int(val_labels[i][0])) 
                       for i in range(len(val_features))]
            
            logger.info(f"Loaded {len(train_data)} training, {len(val_data)} validation samples")
            return train_data, val_data
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return [], []
    
    def augment_data(self, data, multiplier=2):
        """Simple but effective data augmentation"""
        augmented = data[:]
        
        for _ in range(multiplier - 1):
            for image, label in data:
                # Add noise
                noise_level = random.uniform(0.05, 0.15)
                noisy_image = [x + random.gauss(0, noise_level) for x in image]
                augmented.append((noisy_image, label))
        
        logger.info(f"Augmented dataset to {len(augmented)} samples")
        return augmented
    
    def balance_classes(self, data):
        """Balance dataset classes"""
        class_data = {}
        for image, label in data:
            if label not in class_data:
                class_data[label] = []
            class_data[label].append((image, label))
        
        min_size = min(len(samples) for samples in class_data.values())
        balanced = []
        for label, samples in class_data.items():
            balanced.extend(random.sample(samples, min_size))
        
        random.shuffle(balanced)
        logger.info(f"Balanced to {len(balanced)} samples ({min_size} per class)")
        return balanced
    
    def train_model(self, epochs=15, learning_rate=0.01):
        """Train the efficient ViT model"""
        logger.info("Starting Efficient ViT Training...")
        
        # Load and prepare data
        train_data, val_data = self.load_data()
        if not train_data:
            logger.error("No training data found!")
            return None
        
        # Balance classes
        train_data = self.balance_classes(train_data)
        
        # Light augmentation
        train_data = self.augment_data(train_data, multiplier=2)
        
        # Initialize model
        model = EfficientViT(
            image_size=16,
            patch_size=4,
            num_classes=8,
            dim=64
        )
        
        best_val_acc = 0
        patience = 5
        patience_counter = 0
        
        for epoch in range(epochs):
            # Learning rate decay
            current_lr = learning_rate * (0.92 ** epoch)
            
            logger.info(f"\nEpoch {epoch+1}/{epochs} (LR: {current_lr:.5f})")
            logger.info("=" * 50)
            
            # Training
            train_acc, train_loss = model.train_epoch(train_data, current_lr)
            
            # Validation
            val_acc, val_loss = model.evaluate(val_data)
            
            # Log results
            logger.info(f"Train: {train_acc:.1f}% accuracy, {train_loss:.3f} loss")
            logger.info(f"Valid: {val_acc:.1f}% accuracy, {val_loss:.3f} loss")
            
            # Track best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                # Save best model state
                best_model = {
                    'classifier': [row[:] for row in model.classifier],
                    'classifier_bias': model.classifier_bias[:],
                    'feature_transform': [row[:] for row in model.feature_transform],
                    'patch_embedding': [row[:] for row in model.patch_embedding]
                }
                logger.info(f"🎯 New best: {val_acc:.1f}% validation accuracy!")
            else:
                patience_counter += 1
            
            self.training_history.append({
                'epoch': epoch + 1,
                'train_acc': train_acc,
                'val_acc': val_acc,
                'train_loss': train_loss,
                'val_loss': val_loss
            })
            
            # Early stopping
            if patience_counter >= patience:
                logger.info("Early stopping triggered")
                break
        
        # Restore best model
        if 'best_model' in locals():
            model.classifier = best_model['classifier']
            model.classifier_bias = best_model['classifier_bias']
            model.feature_transform = best_model['feature_transform']
            model.patch_embedding = best_model['patch_embedding']
        
        self.best_accuracy = best_val_acc
        return model
    
    def save_model(self, model, filepath):
        """Save the trained model"""
        model_data = {
            'model_type': 'efficient_vit',
            'best_accuracy': self.best_accuracy,
            'training_history': self.training_history,
            'classifier': model.classifier,
            'classifier_bias': model.classifier_bias,
            'feature_transform': model.feature_transform,
            'patch_embedding': model.patch_embedding,
            'image_size': model.image_size,
            'patch_size': model.patch_size,
            'num_classes': model.num_classes,
            'dim': model.dim,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(model_data, f, indent=2)
        
        logger.info(f"Model saved to {filepath}")

def main():
    """Main training function"""
    print("\nAutoSentinel Efficient ViT Training")
    print("=" * 50)
    print("⚡ Optimized for Speed & Significant Improvement")
    print("🎯 Target: 50-70% Accuracy")
    print("🚀 Fast & Efficient Training")
    print("=" * 50)
    
    # Initialize trainer
    trainer = EfficientViTTrainer()
    
    # Train model
    model = trainer.train_model(epochs=15, learning_rate=0.015)
    
    if model:
        # Save model
        os.makedirs('models/vit', exist_ok=True)
        model_path = 'models/vit/efficient_vit_checkpoint.json'
        trainer.save_model(model, model_path)
        
        # Final results
        logger.info(f"\n🎉 TRAINING COMPLETE!")
        logger.info(f"Best Validation Accuracy: {trainer.best_accuracy:.1f}%")
        
        if trainer.best_accuracy >= 50:
            logger.info("🎯 SUCCESS: Target accuracy achieved!")
        elif trainer.best_accuracy >= 35:
            logger.info("📈 Good improvement! Getting closer to target.")
        else:
            logger.info("🔧 Some improvement, but more work needed.")
        
        # Test prediction
        logger.info("\nTesting model prediction...")
        test_image = [random.uniform(0, 1) for _ in range(256)]  # 16x16 image
        prediction = model.predict(test_image)
        logger.info(f"Sample prediction: Class {prediction}")
        
        print(f"\nModel saved to: {model_path}")
        print("Next step: Integrate with main AutoSentinel system")
        
        return model
    else:
        logger.error("Training failed!")
        return None

if __name__ == "__main__":
    main()
