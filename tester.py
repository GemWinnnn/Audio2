import os
import numpy as np
import joblib
import librosa
from tensorflow.keras.models import load_model
import warnings
warnings.filterwarnings('ignore')

class HeartSoundTester:
    """
    Simple tester for heart sound classification model
    """
    
    def __init__(self, model_path, scaler_path):
        """
        Initialize the tester with model and scaler paths
        
        Args:
            model_path (str): Path to the saved .h5 model file
            scaler_path (str): Path to the saved scaler .pkl file
        """
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model = None
        self.scaler = None
        self.load_components()
    
    def load_components(self):
        """Load the model and scaler"""
        try:
            # Load model
            print(f"Loading model from: {self.model_path}")
            self.model = load_model(self.model_path)
            print("✓ Model loaded successfully")
            
            # Load scaler
            print(f"Loading scaler from: {self.scaler_path}")
            self.scaler = joblib.load(self.scaler_path)
            print("✓ Scaler loaded successfully")
            
        except Exception as e:
            print(f"Error loading components: {e}")
            raise
    
    def extract_features(self, audio_path):
        """
        Extract comprehensive features from audio file to match training pipeline
        Total: 181 feature dimensions
        
        Args:
            audio_path (str): Path to the audio file
            
        Returns:
            np.array: Extracted features (181 x time_frames)
        """
        # Configuration to match training
        n_mfcc = 13
        target_sr = 12000
        n_fft = 512
        duration = 12.0
        max_len = int(target_sr * duration)
        hop_length = 256
        n_chroma = 12
        n_mels = 128
        
        try:
            # Load audio with target sample rate
            signal, _ = librosa.load(audio_path, sr=target_sr)
            
            # Standardize audio length
            if len(signal) > max_len:
                signal = signal[:max_len]
            else:
                signal = np.pad(signal, (0, max_len - len(signal)))
            
            # 1. MFCC Features (39 dimensions: 13 base + 13 delta + 13 delta-delta)
            base_mfcc = librosa.feature.mfcc(y=signal, sr=target_sr, n_mfcc=n_mfcc, 
                                             n_fft=n_fft, hop_length=hop_length)
            delta_mfcc = librosa.feature.delta(base_mfcc)
            delta2_mfcc = librosa.feature.delta(base_mfcc, order=2)
            mfcc_features = np.vstack([base_mfcc, delta_mfcc, delta2_mfcc])
            
            # 2. Zero Crossing Rate (1 dimension)
            zcr = librosa.feature.zero_crossing_rate(signal, hop_length=hop_length)
            
            # 3. Chroma Features (12 dimensions)
            chroma = librosa.feature.chroma_stft(y=signal, sr=target_sr, 
                                                 n_fft=n_fft, hop_length=hop_length, n_chroma=n_chroma)
            
            # 4. RMS Energy (1 dimension)
            rms = librosa.feature.rms(y=signal, hop_length=hop_length)
            
            # 5. Melspectrogram (128 dimensions)
            melspec = librosa.feature.melspectrogram(y=signal, sr=target_sr, 
                                                     n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
            # Convert to log scale (dB)
            melspec_db = librosa.power_to_db(melspec, ref=np.max)
            
            # Combine all features
            all_features = np.vstack([
                mfcc_features,      # 39 dimensions
                zcr,                # 1 dimension
                chroma,             # 12 dimensions
                rms,                # 1 dimension
                melspec_db          # 128 dimensions
            ])
            # Total: 181 dimensions
            
            # Normalize features (per coefficient across time)
            normalized_features = (all_features - np.mean(all_features, axis=1, keepdims=True)) / \
                                 (np.std(all_features, axis=1, keepdims=True) + 1e-8)
            
            return normalized_features
            
        except Exception as e:
            print(f"Error extracting features from {audio_path}: {e}")
            return None
    
    def preprocess_features(self, features):
        """
        Preprocess features to match training format
        
        Args:
            features (np.array): Raw features
            
        Returns:
            np.array: Preprocessed features ready for model
        """
        # Add batch dimension and transpose for 1D CNN (samples, timesteps, features)
        features_batch = features.T[np.newaxis, :, :]  # Shape: (1, timesteps, features)
        
        # Scale features
        original_shape = features_batch.shape
        features_reshaped = features_batch.reshape(-1, features_batch.shape[-1])
        features_scaled = self.scaler.transform(features_reshaped)
        features_final = features_scaled.reshape(original_shape)
        
        return features_final
    
    def classify_audio(self, audio_path, threshold=0.5):
        """
        Classify a single audio file
        
        Args:
            audio_path (str): Path to the audio file
            threshold (float): Classification threshold (default: 0.5)
            
        Returns:
            dict: Classification results
        """
        if not os.path.exists(audio_path):
            return {"error": f"Audio file not found: {audio_path}"}
        
        try:
            print(f"\nProcessing: {audio_path}")
            
            # Extract features
            features = self.extract_features(audio_path)
            if features is None:
                return {"error": "Failed to extract features"}
            
            print(f"✓ Features extracted. Shape: {features.shape} (181 x time_frames)")
            
            # Preprocess features
            processed_features = self.preprocess_features(features)
            print(f"✓ Features preprocessed. Shape: {processed_features.shape}")
            
            # Make prediction
            probability = self.model.predict(processed_features, verbose=0)[0][0]
            prediction = 1 if probability > threshold else 0
            
            # Interpret results
            class_name = "Abnormal" if prediction == 1 else "Normal"
            confidence = probability if prediction == 1 else (1 - probability)
            
            results = {
                "audio_path": audio_path,
                "prediction": prediction,
                "class_name": class_name,
                "probability": float(probability),
                "confidence": float(confidence),
                "threshold": threshold
            }
            
            return results
            
        except Exception as e:
            return {"error": f"Classification failed: {e}"}
    
    def print_results(self, results):
        """Print classification results in a nice format"""
        if "error" in results:
            print(f"❌ Error: {results['error']}")
            return
        
        print(f"\n{'='*50}")
        print("HEART SOUND CLASSIFICATION RESULT")
        print("="*50)
        print(f"File: {os.path.basename(results['audio_path'])}")
        print(f"Classification: {results['class_name']}")
        print(f"Confidence: {results['confidence']:.2%}")
        print(f"Raw Probability: {results['probability']:.4f}")
        print(f"Threshold Used: {results['threshold']}")
        
        # Add interpretation with feature info
        feature_breakdown = f"""
Feature Breakdown (181 total dimensions):
• MFCC: 39 features (13 base + 13 delta + 13 delta-delta)
• Zero Crossing Rate: 1 feature  
• Chroma: 12 features (pitch class profiles)
• RMS Energy: 1 feature
• Mel-spectrogram: 128 features (log-scale)"""
        
        if results['class_name'] == 'Normal':
            print(f"✅ The heart sound appears to be NORMAL")
        else:
            print(f"⚠️  The heart sound appears to be ABNORMAL")
        
        print(feature_breakdown)
        
        print("="*50)

# Example usage function
def test_single_audio(model_path, scaler_path, audio_path, threshold=0.5):
    """
    Quick function to test a single audio file
    
    Args:
        model_path (str): Path to .h5 model file
        scaler_path (str): Path to .pkl scaler file  
        audio_path (str): Path to audio file to classify
        threshold (float): Classification threshold
    """
    
    # Create tester
    tester = HeartSoundTester(model_path, scaler_path)
    
    # Classify audio
    results = tester.classify_audio(audio_path, threshold)
    
    # Print results
    tester.print_results(results)
    
    return results

# Interactive testing function
def interactive_test(model_path, scaler_path):
    """
    Interactive testing - keeps asking for audio paths
    
    Args:
        model_path (str): Path to .h5 model file
        scaler_path (str): Path to .pkl scaler file
    """
    
    print("Heart Sound Classification Tester")
    print("="*40)
    print("Enter audio file paths to classify (type 'quit' to exit)")
    
    # Create tester
    tester = HeartSoundTester(model_path, scaler_path)
    
    while True:
        audio_path = input("\nEnter audio file path: ").strip()
        
        if audio_path.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not audio_path:
            continue
            
        # Classify the audio
        results = tester.classify_audio(audio_path)
        tester.print_results(results)

if __name__ == "__main__":
    # Example usage - replace with your actual file paths
    MODEL_PATH = "heart_sound_cnn_20250901_231047.h5"  # Replace with your model path
    SCALER_PATH = "heart_sound_pipeline_20250901_232643_scaler.pkl"  # Replace with your scaler path
    
    # Check if files exist
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model file not found: {MODEL_PATH}")
        print("Please update MODEL_PATH with the correct path to your .h5 file")
    elif not os.path.exists(SCALER_PATH):
        print(f"❌ Scaler file not found: {SCALER_PATH}")
        print("Please update SCALER_PATH with the correct path to your .pkl file")
    else:
        # Run interactive testing
        interactive_test(MODEL_PATH, SCALER_PATH)
        
        # Or test a single file:
        # AUDIO_PATH = "path/to/your/audio/file.wav"
        # results = test_single_audio(MODEL_PATH, SCALER_PATH, AUDIO_PATH)