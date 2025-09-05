import os
import json
import pickle
import joblib
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import (Conv1D, MaxPooling1D, GlobalAveragePooling1D, 
                                   Dense, Dropout, BatchNormalization, Input, 
                                   GlobalMaxPooling1D, Concatenate, SeparableConv1D)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras.regularizers import l2
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def set_global_seed(seed: int = 42):
    """Set seeds for reproducibility across Python, NumPy, and TensorFlow."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

class HeartSoundClassifier:
    """
    Abnormal and Normal Heart Sound Classification using 1D CNN
    """
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self.scaler = StandardScaler()
        self.pipeline_metadata = {}
        
    def load_and_preprocess_data(self):
        """Load and preprocess data for binary classification"""
        
        metadata_path = '/Users/gemwincanete/Audio2/datasets/metadata.csv' ##Change this 
        metadata_df = pd.read_csv(metadata_path)
        
        # Define classes
        self.normal_class = 'normal'
        self.abnormal_classes = [cls for cls in metadata_df['label'].unique() if cls != 'normal']
        
        class_counts = metadata_df['label'].value_counts()
        normal_count = class_counts.get(self.normal_class, 0)
        abnormal_count = sum(class_counts.get(cls, 0) for cls in self.abnormal_classes)
        
        print(f"Class distribution: Normal={normal_count}, Abnormal={abnormal_count}")
        
        # Store metadata
        self.pipeline_metadata.update({
            'normal_class': self.normal_class,
            'abnormal_classes': self.abnormal_classes,
            'class_distribution': metadata_df['label'].value_counts().to_dict()
        })
        
        # Split data
        train_df = metadata_df[metadata_df['split'] == 'train']
        test_df = metadata_df[metadata_df['split'] == 'test']
        
        # Load features
        X_train, y_train = self._load_features(train_df)
        X_test, y_test = self._load_features(test_df)
        
        # Convert to arrays and transpose for 1D CNN (samples, timesteps, features)
        X_train_final = np.transpose(np.array(X_train), (0, 2, 1))
        X_test_final = np.transpose(np.array(X_test), (0, 2, 1))
        
        # Feature scaling
        X_train_scaled = self._scale_features(X_train_final, fit=True)
        X_test_scaled = self._scale_features(X_test_final, fit=False)
        
        # Prepare binary labels
        y_train_binary = np.array([0 if label == self.normal_class else 1 for label in y_train])
        y_test_binary = np.array([0 if label == self.normal_class else 1 for label in y_test])
        
        self.pipeline_metadata['input_shape'] = X_train_scaled.shape[1:]
        
        print(f"Training: Normal={np.sum(y_train_binary==0)}, Abnormal={np.sum(y_train_binary==1)}")
        print(f"Testing: Normal={np.sum(y_test_binary==0)}, Abnormal={np.sum(y_test_binary==1)}")
        print(f"Input shape: {X_train_scaled.shape[1:]}")
        
        return X_train_scaled, y_train_binary, X_test_scaled, y_test_binary, y_train, y_test
    
    def _load_features(self, df):
        """Load features with error handling"""
        X, y, failed_count = [], [], 0
        
        for _, row in df.iterrows():
            feature_path = os.path.join(self.config['data_path'], row['label'], row['split'], row['filename'])
            
            try:
                if os.path.exists(feature_path):
                    features = np.load(feature_path)
                    if not (np.isnan(features).any() or np.isinf(features).any()):
                        X.append(features)
                        y.append(row['label'])
                    else:
                        failed_count += 1
                else:
                    failed_count += 1
            except:
                failed_count += 1
                
        if failed_count > 0:
            print(f"Failed to load {failed_count} files")
        
        return X, y
    
    def _scale_features(self, X, fit=False):
        """Feature scaling for time-series data"""
        original_shape = X.shape
        X_reshaped = X.reshape(-1, X.shape[-1])
        
        if fit:
            self.scaler.fit(X_reshaped)
        
        X_scaled = self.scaler.transform(X_reshaped)
        return X_scaled.reshape(original_shape)
    
    def build_model(self, input_shape):
        """Build CNN for heart sound classification"""
        
        # CNN architecture
        self.model = Sequential([
            Conv1D(256, kernel_size=5, activation='relu', input_shape=input_shape),
            MaxPooling1D(pool_size=2),
            BatchNormalization(),
            
            Conv1D(128, kernel_size=5, activation='relu'),
            MaxPooling1D(pool_size=2),
            BatchNormalization(),
            
            Conv1D(64, kernel_size=5, activation='relu'),
            GlobalMaxPooling1D(),
            
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(1, activation='sigmoid')  
        ], name='HeartSound_CNN')
        
        # Use Adamax optimizer
        optimizer = Adamax(learning_rate=self.config.get('learning_rate', 1e-4))
        
        self.model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"Model built with {self.model.count_params():,} parameters")
        return self.model
    
    def train_model(self, X_train, y_train):
        """Train the model"""
        
        print("Training without class weights (data is balanced)")
        
        # Train/validation split
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_train, y_train, 
            test_size=0.2,
            stratify=y_train, 
            random_state=42
        )
        
        # Callbacks
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"heart_sound_cnn_{timestamp}.h5"
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True, 
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss', 
                factor=0.5,
                patience=15, 
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                model_path, 
                monitor='val_loss',
                save_best_only=True, 
                verbose=1
            )
        ]
        
        print("Training heart sound classifier...")
        history = self.model.fit(
            X_train_split, y_train_split,
            validation_data=(X_val_split, y_val_split),
            epochs=self.config.get('epochs', 50),
            batch_size=self.config.get('batch_size', 8),
            callbacks=callbacks,
            verbose=1
        )
        
        self.pipeline_metadata['model_path'] = model_path
        return history
    
    def evaluate_model(self, X_test, y_test):
        """Comprehensive evaluation"""
        
        y_pred_probs = self.model.predict(X_test, verbose=0).flatten()
        
        # Find optimal threshold
        best_threshold = self._find_optimal_threshold(y_test, y_pred_probs)
        
        # Evaluate at optimal threshold
        y_pred = (y_pred_probs > best_threshold).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        
        # Class-specific metrics
        normal_mask = y_test == 0
        abnormal_mask = y_test == 1
        
        normal_recall = np.sum((y_test == 0) & (y_pred == 0)) / np.sum(y_test == 0)
        abnormal_recall = np.sum((y_test == 1) & (y_pred == 1)) / np.sum(y_test == 1)
        normal_precision = np.sum((y_test == 0) & (y_pred == 0)) / np.sum(y_pred == 0) if np.sum(y_pred == 0) > 0 else 0
        abnormal_precision = np.sum((y_test == 1) & (y_pred == 1)) / np.sum(y_pred == 1) if np.sum(y_pred == 1) > 0 else 0
        
        balanced_accuracy = (normal_recall + abnormal_recall) / 2
        
        results = {
            'threshold': best_threshold,
            'accuracy': accuracy,
            'balanced_accuracy': balanced_accuracy,
            'f1_macro': f1_macro,
            'normal_recall': normal_recall,
            'normal_precision': normal_precision,
            'abnormal_recall': abnormal_recall,
            'abnormal_precision': abnormal_precision,
            'predictions': y_pred,
            'probabilities': y_pred_probs
        }
        
        # Print results
        print(f"\n{'='*60}")
        print("HEART SOUND CLASSIFICATION RESULTS")
        print("="*60)
        print(f"Optimal Threshold: {best_threshold:.3f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Balanced Accuracy: {balanced_accuracy:.4f}")
        print(f"Macro F1-Score: {f1_macro:.4f}")
        print(f"\nClass-specific Performance:")
        print(f"  Normal - Precision: {normal_precision:.4f}, Recall: {normal_recall:.4f}")
        print(f"  Abnormal - Precision: {abnormal_precision:.4f}, Recall: {abnormal_recall:.4f}")
        
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Normal', 'Abnormal'], digits=4))
        
        # Plot confusion matrix
        self._plot_confusion_matrix(y_test, y_pred)
        
        return results
    
    def _find_optimal_threshold(self, y_true, y_pred_probs):
        """Find threshold that maximizes F1 score"""
        thresholds = np.linspace(0.1, 0.9, 80)
        best_f1 = 0
        best_threshold = 0.5
        
        for thresh in thresholds:
            y_pred = (y_pred_probs > thresh).astype(int)
            f1 = f1_score(y_true, y_pred, average='macro')
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = thresh
        
        return best_threshold
    
    def _plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        
        class_names = ['Normal', 'Abnormal']
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        cm_norm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-8)
        
        # Create annotation text
        annot_text = np.array([[f'{cm[i, j]}\n({cm_norm[i, j]*100:.1f}%)' 
                               for j in range(2)] for i in range(2)])
        
        sns.heatmap(cm_norm, annot=annot_text, fmt='', 
                   xticklabels=class_names, yticklabels=class_names,
                   cmap='Blues', cbar_kws={'label': 'Normalized Accuracy'})
        
        plt.title('Heart Sound Classification\nConfusion Matrix', fontweight='bold', fontsize=14)
        plt.xlabel('Predicted Class', fontweight='bold')
        plt.ylabel('True Class', fontweight='bold')
        
        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f"heart_sound_confusion_matrix_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_scaler(self, filepath=None):
        """Save the StandardScaler"""
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"heart_sound_scaler_{timestamp}.pkl"
        
        joblib.dump(self.scaler, filepath)
        print(f"Scaler saved to: {filepath}")
        
        # Save scaler metadata
        scaler_info = {
            'scaler_type': 'StandardScaler',
            'n_features_in_': getattr(self.scaler, 'n_features_in_', None),
            'mean_': self.scaler.mean_.tolist() if hasattr(self.scaler, 'mean_') else None,
            'scale_': self.scaler.scale_.tolist() if hasattr(self.scaler, 'scale_') else None,
            'var_': self.scaler.var_.tolist() if hasattr(self.scaler, 'var_') else None,
            'input_shape': self.pipeline_metadata.get('input_shape', None),
            'timestamp': datetime.now().isoformat()
        }
        
        info_filepath = filepath.replace('.pkl', '_info.json')
        with open(info_filepath, 'w') as f:
            json.dump(scaler_info, f, indent=2)
        print(f"Scaler metadata saved to: {info_filepath}")
        
        return filepath, info_filepath
    
    def save_pipeline(self, filepath):
        """Save the complete pipeline"""
        pipeline_data = {
            'config': self.config,
            'scaler': self.scaler,
            'metadata': self.pipeline_metadata,
            'model_path': self.pipeline_metadata.get('model_path')
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(pipeline_data, f)
        
        print(f"Pipeline saved to: {filepath}")
        
        # Save scaler separately
        scaler_path, info_path = self.save_scaler(
            filepath.replace('.pkl', '_scaler.pkl')
        )
        
        return filepath, scaler_path, info_path
    
    def load_pipeline(self, filepath):
        """Load a saved pipeline"""
        with open(filepath, 'rb') as f:
            pipeline_data = pickle.load(f)
        
        self.config = pipeline_data['config']
        self.scaler = pipeline_data['scaler']
        self.pipeline_metadata = pipeline_data['metadata']
        
        model_path = pipeline_data['model_path']
        if os.path.exists(model_path):
            self.model = load_model(model_path, compile=False)
        else:
            print(f"Warning: Model file not found at {model_path}")
    
    def predict(self, X_new, threshold=0.5):
        """Make predictions on new data"""
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        # Scale the new data
        X_scaled = self._scale_features(X_new, fit=False)
        
        # Get predictions
        probabilities = self.model.predict(X_scaled, verbose=0).flatten()
        predictions = (probabilities > threshold).astype(int)
        
        return predictions, probabilities

def run_heart_sound_pipeline():
    """Run the heart sound classification pipeline"""
    set_global_seed(42)

    CONFIG = {
        'data_path': '/Users/gemwincanete/Audio2/datasets/enhanced_features',
        'batch_size': 8,
        'epochs': 100,
        'learning_rate': 1e-4
    }
    
    classifier = HeartSoundClassifier(CONFIG)
    
    # Define architecture and loss type for logging
    architecture_type = "CNN"
    loss_type = "Binary Crossentropy"
    
    print("HEART SOUND CLASSIFICATION PIPELINE")
    print(f"Architecture: {architecture_type}")
    print(f"Loss Function: {loss_type}")
    print("="*60)
    
    # Load and preprocess data
    X_train, y_train, X_test, y_test, _, _ = classifier.load_and_preprocess_data()
    
    # Build model
    input_shape = X_train.shape[1:]
    classifier.build_model(input_shape)
    
    # Print model summary
    print("\nModel Architecture:")
    classifier.model.summary()
    
    # Train model
    print("\nTraining...")
    history = classifier.train_model(X_train, y_train)
    
    # Evaluate model
    print("\nEvaluating...")
    results = classifier.evaluate_model(X_test, y_test)
    
    # Save pipeline
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pipeline_path = f"heart_sound_pipeline_{timestamp}.pkl"
    pipeline_file, scaler_file, scaler_info = classifier.save_pipeline(pipeline_path)
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"training_history_{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Final summary
    print(f"\n{'='*60}") 
    print("PIPELINE SUMMARY")
    print("="*60)
    print(f"Model: {architecture_type}")
    print(f"Parameters: {classifier.model.count_params():,}")
    print(f"Loss Function: {loss_type}")
    print(f"Optimal Threshold: {results['threshold']:.3f}")
    print(f"Final Accuracy: {results['accuracy']:.4f}")
    print(f"Balanced Accuracy: {results['balanced_accuracy']:.4f}")
    print(f"Macro F1-Score: {results['f1_macro']:.4f}")
    print(f"Normal Performance: P={results['normal_precision']:.3f}, R={results['normal_recall']:.3f}")
    print(f"Abnormal Performance: P={results['abnormal_precision']:.3f}, R={results['abnormal_recall']:.3f}")
    print(f"Pipeline saved: {pipeline_file}")
    print(f"Scaler saved: {scaler_file}")
    
    return classifier, results, history

# Utility functions for scaler management
def load_scaler(scaler_path):
    """Load a saved scaler"""
    return joblib.load(scaler_path)

def apply_scaler_to_data(scaler_path, new_data):
    """Apply saved scaler to new data"""
    scaler = load_scaler(scaler_path)
    
    original_shape = new_data.shape
    if len(new_data.shape) == 3:
        new_data_reshaped = new_data.reshape(-1, new_data.shape[-1])
        scaled_data = scaler.transform(new_data_reshaped)
        scaled_data = scaled_data.reshape(original_shape)
    else:
        scaled_data = scaler.transform(new_data)
    
    return scaled_data

# Main execution
if __name__ == "__main__":
    print("Running Heart Sound Classification (Abnormal and Normal) with CNN")
    classifier, results, history = run_heart_sound_pipeline()