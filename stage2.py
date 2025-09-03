import os
import json
import pickle
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import (LSTM, Conv1D, MaxPooling1D, GlobalAveragePooling1D, 
                                   Dense, Dropout, BatchNormalization, Input,
                                   Multiply, Reshape, Softmax, Lambda, Bidirectional,
                                   LayerNormalization, Add)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def set_global_seed(seed: int = 42):
    """Set seeds for reproducibility across Python, NumPy, and TensorFlow."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

class AbnormalityClassificationPipeline:
    """
    Direct Multi-class Classification Pipeline for Abnormal Heart Sound Types
    Classifies specific abnormality types directly without hierarchical structure
    """
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.pipeline_metadata = {}
        
    def load_and_preprocess_data(self):
        """Load and preprocess data for direct multi-class classification"""
        
        # Load metadata
        metadata_path = os.path.join(self.config['data_path'], self.config['metadata_file'])
        metadata_df = pd.read_csv(metadata_path)
        
        # Filter out normal class - only keep abnormal classes
        abnormal_df = metadata_df[metadata_df['label'] != 'normal']
        self.abnormal_classes = abnormal_df['label'].unique().tolist()
        
        print(f"Abnormal classes to classify: {self.abnormal_classes}")
        print(f"Class distribution: {abnormal_df['label'].value_counts().to_dict()}")
        
        # Verify class folders exist
        for class_name in self.abnormal_classes:
            class_path = os.path.join(self.config['data_path'], class_name)
            if not os.path.exists(class_path):
                print(f"WARNING: Class folder '{class_name}' not found at {class_path}")
            else:
                train_path = os.path.join(class_path, 'train')
                test_path = os.path.join(class_path, 'test')
                train_files = len(os.listdir(train_path)) if os.path.exists(train_path) else 0
                test_files = len(os.listdir(test_path)) if os.path.exists(test_path) else 0
                print(f"  {class_name}: {train_files} train files, {test_files} test files")
        
        # Store metadata
        self.pipeline_metadata.update({
            'abnormal_classes': self.abnormal_classes,
            'total_samples': len(abnormal_df),
            'class_distribution': abnormal_df['label'].value_counts().to_dict()
        })
        
        # Load data - only abnormal samples
        train_df = abnormal_df[abnormal_df['split'] == 'train']
        test_df = abnormal_df[abnormal_df['split'] == 'test']
        
        X_train, y_train = self._load_features(train_df)
        X_test, y_test = self._load_features(test_df)
        
        # Prepare data arrays
        X_train_array = np.array(X_train)
        X_test_array = np.array(X_test)
        
        # Transpose for models: (samples, timesteps, features)
        X_train_final = np.transpose(X_train_array, (0, 2, 1))
        X_test_final = np.transpose(X_test_array, (0, 2, 1))
        
        # Feature scaling
        X_train_scaled = self._scale_features(X_train_final, fit=True)
        X_test_scaled = self._scale_features(X_test_final, fit=False)
        
        # Encode labels
        self.label_encoder.fit(y_train)
        y_train_encoded = self.label_encoder.transform(y_train)
        y_test_encoded = self.label_encoder.transform(y_test)
        y_train_categorical = to_categorical(y_train_encoded, len(self.abnormal_classes))
        y_test_categorical = to_categorical(y_test_encoded, len(self.abnormal_classes))
        
        # Store preprocessing info
        self.pipeline_metadata['input_shape'] = X_train_scaled.shape[1:]
        self.pipeline_metadata['n_classes'] = len(self.abnormal_classes)
        
        return (X_train_scaled, y_train_categorical, X_test_scaled, y_test_categorical,
                y_train, y_test)  # Return original labels for evaluation
    
    def _load_features(self, df):
        """Load features with error handling"""
        X, y = [], []
        failed_count = 0
        
        for _, row in df.iterrows():
            try:
                feature_path = os.path.join(
                    self.config['data_path'], 
                    row['label'], 
                    row['split'], 
                    row['filename']
                )
                
                if os.path.exists(feature_path):
                    features = np.load(feature_path)
                    
                    # Basic feature validation
                    if not np.isnan(features).any() and not np.isinf(features).any():
                        X.append(features)
                        y.append(row['label'])
                    else:
                        failed_count += 1
                else:
                    failed_count += 1
            except Exception as e:
                failed_count += 1
                
        print(f"Failed to load {failed_count} files")
        return X, y
    
    def _scale_features(self, X, fit=False):
        """Feature scaling"""
        n_samples, n_timesteps, n_features = X.shape
        X_reshaped = X.reshape(-1, n_features)
        
        if fit:
            self.scaler.fit(X_reshaped)
        
        X_scaled = self.scaler.transform(X_reshaped)
        X_scaled = X_scaled.reshape(n_samples, n_timesteps, n_features)
        
        return X_scaled
    
    def build_model(self, input_shape):
        """
        Build LSTM-CNN model for direct abnormality classification
        """
        
        inputs = Input(shape=input_shape, name='input')
        
        # LSTM Layers - Capture temporal dependencies
        lstm_1 = Bidirectional(
            LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.2,
                 kernel_regularizer=l2(1e-4), recurrent_regularizer=l2(1e-4)),
            name='bilstm_1'
        )(inputs)
        lstm_1 = LayerNormalization(name='ln_1')(lstm_1)
        
        lstm_2 = Bidirectional(
            LSTM(96, return_sequences=True, dropout=0.35, recurrent_dropout=0.25,
                 kernel_regularizer=l2(1e-4), recurrent_regularizer=l2(1e-4)),
            name='bilstm_2'
        )(lstm_1)
        lstm_2 = LayerNormalization(name='ln_2')(lstm_2)
        
        # Skip connection
        skip_1 = Dense(192, activation='linear', name='skip_projection_1')(lstm_1)
        lstm_2_skip = Add(name='skip_add_1')([lstm_2, skip_1])
        
        lstm_3 = Bidirectional(
            LSTM(64, return_sequences=True, dropout=0.4, recurrent_dropout=0.3,
                 kernel_regularizer=l2(1e-4), recurrent_regularizer=l2(1e-4)),
            name='bilstm_3'
        )(lstm_2_skip)
        lstm_3 = LayerNormalization(name='ln_3')(lstm_3)
        
        # CNN Layers
        x = Conv1D(128, kernel_size=7, activation='relu', padding='same',
                   kernel_regularizer=l2(1e-4), name='conv1d_1')(lstm_3)
        x = BatchNormalization(name='bn_1')(x)
        x = MaxPooling1D(pool_size=2, name='pool_1')(x)
        x = Dropout(0.25, name='dropout_1')(x)
        
        x = Conv1D(256, kernel_size=5, activation='relu', padding='same',
                   kernel_regularizer=l2(1e-4), name='conv1d_2')(x)
        x = BatchNormalization(name='bn_2')(x)
        x = MaxPooling1D(pool_size=2, name='pool_2')(x)
        x = Dropout(0.3, name='dropout_2')(x)
        
        x = Conv1D(512, kernel_size=3, activation='relu', padding='same',
                   kernel_regularizer=l2(1e-4), name='conv1d_3')(x)
        x = BatchNormalization(name='bn_3')(x)
        x = Dropout(0.35, name='dropout_3')(x)
        
        # Global pooling
        x = GlobalAveragePooling1D(name='global_avg_pool')(x)
        
        # Classification head
        x = Dense(256, activation='relu', kernel_regularizer=l2(1e-4), name='fc_1')(x)
        x = BatchNormalization(name='bn_fc_1')(x)
        x = Dropout(0.5, name='dropout_fc_1')(x)
        
        x = Dense(128, activation='relu', kernel_regularizer=l2(1e-4), name='fc_2')(x)
        x = BatchNormalization(name='bn_fc_2')(x)
        x = Dropout(0.5, name='dropout_fc_2')(x)
        
        x = Dense(64, activation='relu', kernel_regularizer=l2(1e-4), name='fc_3')(x)
        x = Dropout(0.4, name='dropout_fc_3')(x)
        
        # Output layer
        outputs = Dense(len(self.abnormal_classes), activation='softmax', name='output')(x)
        
        # Create model
        self.model = Model(inputs=inputs, outputs=outputs, name='Abnormality_Classification')
        
        # Focal loss for imbalanced classes
        def focal_loss(gamma=2.0):
            def focal_loss_fn(y_true, y_pred):
                y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
                ce = -y_true * tf.math.log(y_pred)
                p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
                focal_weight = tf.pow(1 - p_t, gamma)
                focal_loss = focal_weight * ce
                return tf.reduce_sum(focal_loss, axis=-1)
            return focal_loss_fn
        
        # Compile model
        optimizer = Adam(learning_rate=self.config.get('learning_rate', 1e-3), clipnorm=1.0)
        
        self.model.compile(
            optimizer=optimizer,
            loss=focal_loss(gamma=2.5),
            metrics=['accuracy', 'precision', 'recall']
        )
        
        print("Abnormality Classification Model:")
        self.model.summary()
        
        return self.model
    
    def train_model(self, X_train, y_train):
        """Train the abnormality classification model"""
        
        # Calculate class weights for imbalanced classes
        y_train_classes = np.argmax(y_train, axis=1)
        class_counts = np.bincount(y_train_classes)
        total_samples = len(y_train_classes)
        n_classes = len(class_counts)
        
        class_weights = {}
        for i in range(n_classes):
            base_weight = total_samples / (n_classes * class_counts[i])
            # Apply class emphasis from config
            class_name = self.label_encoder.inverse_transform([i])[0]
            emphasis = self.config.get('class_emphasis', {}).get(class_name, 1.0)
            class_weights[i] = base_weight * emphasis
        
        print(f"Class weights: {class_weights}")
        
        # Train/validation split
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_train, y_train, 
            test_size=self.config['validation_split'],
            stratify=np.argmax(y_train, axis=1), 
            random_state=42
        )
        
        # Callbacks
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"abnormality_classification_{timestamp}.h5"
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1),
            ModelCheckpoint(model_path, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
        ]
        
        # Train model
        history = self.model.fit(
            X_train_split, y_train_split,
            validation_data=(X_val_split, y_val_split),
            epochs=self.config['epochs'],
            batch_size=self.config['batch_size'],
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1
        )
        
        self.pipeline_metadata['model_path'] = model_path
        return history
    
    def predict(self, X):
        """Make predictions on input data"""
        probabilities = self.model.predict(X, verbose=0)
        predictions_encoded = np.argmax(probabilities, axis=1)
        predictions = self.label_encoder.inverse_transform(predictions_encoded)
        
        return predictions, probabilities
    
    def evaluate_model(self, X_test, y_test_categorical, y_test_original):
        """Evaluate the model and plot confusion matrix"""
        
        # Get predictions
        predictions, probabilities = self.predict(X_test)
        y_test_classes = np.argmax(y_test_categorical, axis=1)
        predictions_encoded = np.argmax(probabilities, axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test_classes, predictions_encoded)
        f1_macro = f1_score(y_test_classes, predictions_encoded, average='macro')
        f1_weighted = f1_score(y_test_classes, predictions_encoded, average='weighted')
        f1_per_class = f1_score(y_test_classes, predictions_encoded, average=None)
        
        print(f"\n" + "="*50)
        print(f"ABNORMALITY CLASSIFICATION EVALUATION")
        print(f"="*50)
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1-Score (Macro): {f1_macro:.4f}")
        print(f"F1-Score (Weighted): {f1_weighted:.4f}")
        
        # Per-class F1 scores
        print(f"\nPer-class F1 scores:")
        for i, class_name in enumerate(self.label_encoder.classes_):
            print(f"  {class_name}: {f1_per_class[i]:.4f}")
        
        # Detailed classification report
        print(f"\nDetailed Classification Report:")
        print(classification_report(y_test_original, predictions, digits=4))
        
        # Plot confusion matrix
        self._plot_confusion_matrix(y_test_original, predictions)
        
        results = {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'f1_per_class': dict(zip(self.label_encoder.classes_, f1_per_class)),
            'predictions': predictions,
            'probabilities': probabilities
        }
        
        return results
    
    def plot_training_history(self, history):
        """Plot training and validation loss and accuracy curves"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
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
    
    def _plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix"""
        plt.figure(figsize=(10, 8))
        
        # Get unique classes and sort them for consistent ordering
        unique_classes = sorted(list(set(y_true) | set(y_pred)))
        cm = confusion_matrix(y_true, y_pred, labels=unique_classes)
        cm_norm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-8)
        
        # Create annotation text
        annot_text = []
        for i in range(len(unique_classes)):
            row_text = []
            for j in range(len(unique_classes)):
                count = cm[i, j]
                percentage = cm_norm[i, j] * 100
                row_text.append(f'{count}\n({percentage:.1f}%)')
            annot_text.append(row_text)
        
        sns.heatmap(cm_norm, annot=annot_text, fmt='', 
                   xticklabels=unique_classes, yticklabels=unique_classes,
                   cmap='Blues', cbar_kws={'label': 'Normalized Accuracy'})
        plt.title('Abnormality Classification Confusion Matrix', fontweight='bold', fontsize=14)
        plt.xlabel('Predicted Class', fontweight='bold')
        plt.ylabel('True Class', fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f"abnormality_classification_cm_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_pipeline(self, filepath):
        """Save the trained pipeline"""
        # Create timestamped filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pkl_path = f"{filepath}_{timestamp}.pkl"
        model_path = f"{filepath}_model_{timestamp}.h5"
        
        # Save pipeline metadata and preprocessing objects
        pipeline_data = {
            'config': self.config,
            'pipeline_metadata': self.pipeline_metadata,
            'label_encoder': self.label_encoder,
            'scaler': self.scaler,
            'abnormal_classes': self.abnormal_classes
        }
        
        with open(pkl_path, 'wb') as f:
            pickle.dump(pipeline_data, f)
        
        # Save the model separately
        if self.model is not None:
            self.model.save(model_path)
        
        print(f"Pipeline saved to {pkl_path}")
        print(f"Model saved to {model_path}")
        
        return pkl_path, model_path
    
    def load_pipeline(self, filepath):
        """Load a saved pipeline"""
        with open(filepath, 'rb') as f:
            pipeline_data = pickle.load(f)
        
        self.config = pipeline_data['config']
        self.pipeline_metadata = pipeline_data['pipeline_metadata']
        self.label_encoder = pipeline_data['label_encoder']
        self.scaler = pipeline_data['scaler']
        self.abnormal_classes = pipeline_data['abnormal_classes']
        
        print(f"Pipeline loaded from {filepath}")

def run_abnormality_classification_pipeline():
    """Run the abnormality classification pipeline"""
    set_global_seed(42)

    CONFIG = {
        'data_path': '/Users/gemwincanete/Audio2/datasets/enhanced_features',
        'metadata_file': '/Users/gemwincanete/Audio2/datasets/metadata.csv',
        'validation_split': 0.2,
        'batch_size': 32,
        'epochs': 200,
        'learning_rate': 1e-3,
        'class_emphasis': {
            'extrasystole': 2.5,
            'extrahls': 2.0,
            'murmur': 2.0,
            'artifact': 1.4
        }
    }
    
    # Initialize pipeline
    pipeline = AbnormalityClassificationPipeline(CONFIG)
    
    print("="*60)
    print("ABNORMALITY CLASSIFICATION PIPELINE")
    print("Direct Multi-class Classification of Abnormal Heart Sounds")
    print("="*60)
    
    # Load and preprocess data
    (X_train, y_train_categorical, X_test, y_test_categorical,
     y_train_original, y_test_original) = pipeline.load_and_preprocess_data()
    
    # Build model
    input_shape = (X_train.shape[1], X_train.shape[2])
    pipeline.build_model(input_shape)
    
    # Train model
    print("\n" + "="*40)
    print("TRAINING MODEL")
    print("="*40)
    history = pipeline.train_model(X_train, y_train_categorical)
    
    # Plot training history
    print("\n" + "="*40)
    print("TRAINING HISTORY VISUALIZATION")
    print("="*40)
    pipeline.plot_training_history(history)
    
    # Evaluate model
    print("\n" + "="*40)
    print("MODEL EVALUATION")
    print("="*40)
    results = pipeline.evaluate_model(X_test, y_test_categorical, y_test_original)
    
    # Final summary
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)
    print(f"Overall Accuracy: {results['accuracy']:.4f}")
    print(f"F1-Score (Macro): {results['f1_macro']:.4f}")
    print(f"F1-Score (Weighted): {results['f1_weighted']:.4f}")
    
    print(f"\nPer-class Performance:")
    for class_name, f1_score in results['f1_per_class'].items():
        print(f"  {class_name}: F1={f1_score:.4f}")
    
    # Save complete pipeline
    pipeline_path = "abnormality_classification_pipeline"
    pkl_path, model_path = pipeline.save_pipeline(pipeline_path)
    
    print(f"\nPipeline files saved:")
    print(f"  Metadata & Preprocessing: {pkl_path}")
    print(f"  Model: {model_path}")
    
    return pipeline, results, history

if __name__ == "__main__":
    pipeline, results, history = run_abnormality_classification_pipeline()