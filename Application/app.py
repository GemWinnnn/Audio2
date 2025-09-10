from flask import Flask, render_template, request, jsonify, send_file, url_for
import os
import librosa
import numpy as np
import joblib
import io
import base64
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import soundfile as sf
from tensorflow.keras.models import load_model
import json
from datetime import datetime
import tempfile
from werkzeug.utils import secure_filename
import pickle

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'

# Create directories if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

class HeartSoundAnalyzer:
    def __init__(self):
        self.denoise_model = None
        self.stage1_model = None
        self.stage2_model = None
        self.stage1_scaler = None
        self.stage2_scaler = None
        self.label_mappings = None
        self.models_loaded = False
        
        # Audio processing parameters (matching your feature extraction)
        self.target_sr = 12000
        self.duration = 12.0
        self.max_len = int(self.target_sr * self.duration)
        self.n_fft = 512
        self.hop_length = 256
        self.n_mfcc = 13
        self.n_chroma = 12
        self.n_mels = 128
        
        # Denoise model parameters
        self.denoise_sr = 1000
        self.denoise_window = 12000
        
    def load_models(self):
        """Load all models and scalers"""
        try:
            # Get the base directory (parent of Application folder)
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            # Get the Application directory
            app_dir = os.path.dirname(os.path.abspath(__file__))
            
            print(f"🔍 Base directory: {base_dir}")
            print(f"🔍 App directory: {app_dir}")
            print(f"🔍 Current working directory: {os.getcwd()}")
            print(f"🔍 Files in app directory: {os.listdir(app_dir)}")
            
            # Check if we're in a different directory structure (e.g., on Render)
            if not os.path.exists(os.path.join(base_dir, 'Denoise')):
                print("🔍 Checking alternative directory structure...")
                # Maybe the models are in the same directory as the app
                if os.path.exists(os.path.join(app_dir, 'Denoise')):
                    print("🔍 Found Denoise in app directory, using that instead")
                    base_dir = app_dir
            
            print("📄 Loading denoise model...")
            # Load denoise model
            denoise_path = os.path.join(base_dir, 'Denoise', 'best_model_denoise.h5')
            print(f"🔍 Denoise model path: {denoise_path}")
            print(f"🔍 Denoise model exists: {os.path.exists(denoise_path)}")
            
            if not os.path.exists(denoise_path):
                print(f"⚠️ Denoise model not found at: {denoise_path}")
                print(f"🔍 Checking if Denoise directory exists: {os.path.exists(os.path.join(base_dir, 'Denoise'))}")
                if os.path.exists(os.path.join(base_dir, 'Denoise')):
                    print(f"🔍 Files in Denoise directory: {os.listdir(os.path.join(base_dir, 'Denoise'))}")
                return False
            
            try:
                self.denoise_model = load_model(denoise_path, compile=False)
                print("✅ Denoise model loaded")
            except Exception as e:
                print(f"❌ Error loading denoise model: {e}")
                return False
            
            print("📄 Loading Stage 1 model...")
            # Load Stage 1 model (Normal vs Abnormal) - models are in Application directory
            stage1_model_path = os.path.join(app_dir, 'stage1_results', 'heart_sound_cnn_20250901_231047.h5')
            stage1_scaler_path = os.path.join(app_dir, 'stage1_results', 'heart_sound_pipeline_20250901_232643_scaler.pkl')
            
            print(f"🔍 Stage 1 model path: {stage1_model_path}")
            print(f"🔍 Stage 1 scaler path: {stage1_scaler_path}")
            print(f"🔍 Stage 1 model exists: {os.path.exists(stage1_model_path)}")
            print(f"🔍 Stage 1 scaler exists: {os.path.exists(stage1_scaler_path)}")
            
            if not os.path.exists(stage1_model_path) or not os.path.exists(stage1_scaler_path):
                print(f"⚠️ Stage 1 files not found:")
                print(f"   Model: {stage1_model_path}")
                print(f"   Scaler: {stage1_scaler_path}")
                print(f"🔍 Checking if stage1_results directory exists: {os.path.exists(os.path.join(app_dir, 'stage1_results'))}")
                if os.path.exists(os.path.join(app_dir, 'stage1_results')):
                    print(f"🔍 Files in stage1_results directory: {os.listdir(os.path.join(app_dir, 'stage1_results'))}")
                return False
            
            try:
                self.stage1_model = load_model(stage1_model_path, compile=False)
                self.stage1_scaler = joblib.load(stage1_scaler_path)
                print("✅ Stage 1 model and scaler loaded")
            except Exception as e:
                print(f"❌ Error loading Stage 1 model: {e}")
                return False
            
            print("📄 Loading Stage 2 model...")
            # Load Stage 2 model (Abnormality classification) - models are in Application directory
            stage2_model_path = os.path.join(app_dir, 'stage2_results', 'abnormality_classification_20250903_140057.h5')
            stage2_scaler_path = os.path.join(app_dir, 'stage2_results', 'abnormality_classification_pipeline_20250903_144628.pkl')
            
            print(f"🔍 Stage 2 model path: {stage2_model_path}")
            print(f"🔍 Stage 2 scaler path: {stage2_scaler_path}")
            print(f"🔍 Stage 2 model exists: {os.path.exists(stage2_model_path)}")
            print(f"🔍 Stage 2 scaler exists: {os.path.exists(stage2_scaler_path)}")
            
            if not os.path.exists(stage2_model_path) or not os.path.exists(stage2_scaler_path):
                print(f"⚠️ Stage 2 files not found:")
                print(f"   Model: {stage2_model_path}")
                print(f"   Scaler: {stage2_scaler_path}")
                print(f"🔍 Checking if stage2_results directory exists: {os.path.exists(os.path.join(app_dir, 'stage2_results'))}")
                if os.path.exists(os.path.join(app_dir, 'stage2_results')):
                    print(f"🔍 Files in stage2_results directory: {os.listdir(os.path.join(app_dir, 'stage2_results'))}")
                return False
            
            try:
                self.stage2_model = load_model(stage2_model_path, compile=False)
                
                # Load stage 2 scaler from pipeline
                with open(stage2_scaler_path, 'rb') as f:
                    stage2_pipeline = pickle.load(f)
                    if isinstance(stage2_pipeline, dict) and 'scaler' in stage2_pipeline:
                        self.stage2_scaler = stage2_pipeline['scaler']
                    else:
                        # If it's just the scaler itself
                        self.stage2_scaler = stage2_pipeline
                print("✅ Stage 2 model and scaler loaded")
            except Exception as e:
                print(f"❌ Error loading Stage 2 model: {e}")
                return False
            
            # Define label mappings
            self.label_mappings = {
                'stage1': {0: 'Normal', 1: 'Abnormal'},
                'stage2': {
                    0: 'artifact', 
                    1: 'extra_heart_audio', 
                    2: 'extra_systole', 
                    3: 'murmur'
                }
            }
            
            self.models_loaded = True
            return True
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            return False
    
    def denoise_audio(self, audio_data, sr):
        """Denoise audio using the denoise model"""
        try:
            # Resample to denoise model's expected sample rate
            audio_resampled = librosa.resample(audio_data, orig_sr=sr, target_sr=self.denoise_sr)
            
            # Pad or trim to expected window size
            if len(audio_resampled) < self.denoise_window:
                audio_padded = np.pad(audio_resampled, (0, self.denoise_window - len(audio_resampled)))
            else:
                audio_padded = audio_resampled[:self.denoise_window]
            
            # Reshape for model: (1, window_size, 1)
            model_input = audio_padded.reshape(1, self.denoise_window, 1)
            
            # Denoise
            denoised = self.denoise_model.predict(model_input, verbose=0)
            
            # Remove batch and channel dimensions
            if denoised.shape[0] == 1:
                denoised = denoised[0]
            if len(denoised.shape) > 1 and denoised.shape[-1] == 1:
                denoised = denoised[:, 0]
            
            # Ensure denoised is 1D
            denoised = np.squeeze(denoised)
            
            # Resample back to target sample rate for feature extraction
            denoised_resampled = librosa.resample(denoised, orig_sr=self.denoise_sr, target_sr=self.target_sr)
            
            return denoised_resampled, audio_padded
            
        except Exception as e:
            print(f"❌ Error in denoising: {e}")
            raise
    
    def extract_features(self, signal):
        """Extract comprehensive audio features (matching your feature.py)"""
        try:
            # Standardize audio length
            if len(signal) > self.max_len:
                signal = signal[:self.max_len]
            else:
                signal = np.pad(signal, (0, self.max_len - len(signal)))
            
            # 1. MFCC Features (39 dimensions: 13 base + 13 delta + 13 delta-delta)
            base_mfcc = librosa.feature.mfcc(y=signal, sr=self.target_sr, n_mfcc=self.n_mfcc, 
                                           n_fft=self.n_fft, hop_length=self.hop_length)
            delta_mfcc = librosa.feature.delta(base_mfcc)
            delta2_mfcc = librosa.feature.delta(base_mfcc, order=2)
            mfcc_features = np.vstack([base_mfcc, delta_mfcc, delta2_mfcc])
            
            # 2. Zero Crossing Rate (1 dimension)
            zcr = librosa.feature.zero_crossing_rate(signal, hop_length=self.hop_length)
            
            # 3. Chroma Features (12 dimensions)
            chroma = librosa.feature.chroma_stft(y=signal, sr=self.target_sr, 
                                               n_fft=self.n_fft, hop_length=self.hop_length, 
                                               n_chroma=self.n_chroma)
            
            # 4. RMS Energy (1 dimension)
            rms = librosa.feature.rms(y=signal, hop_length=self.hop_length)
            
            # 5. Melspectrogram (128 dimensions)
            melspec = librosa.feature.melspectrogram(y=signal, sr=self.target_sr, 
                                                   n_fft=self.n_fft, hop_length=self.hop_length, 
                                                   n_mels=self.n_mels)
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
            print(f"❌ Error in feature extraction: {e}")
            raise
    
    def classify_stage1(self, features):
        """Stage 1: Normal vs Abnormal classification"""
        try:
            # Transpose for CNN input (samples, timesteps, features)
            features_transposed = np.transpose(features[np.newaxis, :, :], (0, 2, 1))
            
            # Scale features
            original_shape = features_transposed.shape
            features_reshaped = features_transposed.reshape(-1, features_transposed.shape[-1])
            features_scaled = self.stage1_scaler.transform(features_reshaped)
            features_final = features_scaled.reshape(original_shape)
            
            # Predict
            prediction_prob = self.stage1_model.predict(features_final, verbose=0)[0][0]
            prediction = int(prediction_prob > 0.5)
            
            return {
                'prediction': prediction,
                'probability': float(prediction_prob),
                'label': self.label_mappings['stage1'][prediction],
                'confidence': float(max(prediction_prob, 1 - prediction_prob))
            }
            
        except Exception as e:
            print(f"❌ Error in Stage 1 classification: {e}")
            raise
    
    def classify_stage2(self, features):
        """Stage 2: Abnormality type classification"""
        try:
            # Transpose for CNN input (samples, timesteps, features)
            features_transposed = np.transpose(features[np.newaxis, :, :], (0, 2, 1))
            
            # Scale features
            original_shape = features_transposed.shape
            features_reshaped = features_transposed.reshape(-1, features_transposed.shape[-1])
            features_scaled = self.stage2_scaler.transform(features_reshaped)
            features_final = features_scaled.reshape(original_shape)
            
            # Predict
            prediction_probs = self.stage2_model.predict(features_final, verbose=0)[0]
            prediction = int(np.argmax(prediction_probs))
            
            return {
                'prediction': prediction,
                'probabilities': prediction_probs.tolist(),
                'label': self.label_mappings['stage2'][prediction],
                'confidence': float(np.max(prediction_probs))
            }
            
        except Exception as e:
            print(f"❌ Error in Stage 2 classification: {e}")
            raise
    
    def analyze_audio(self, file_path):
        """Complete audio analysis pipeline"""
        try:
            print(f"📄 Analyzing audio file: {file_path}")
            
            # Load original audio
            original_audio, sr = librosa.load(file_path)
            print(f"🔊 Original audio: {len(original_audio)} samples at {sr} Hz")
            
            # Denoise audio
            print("📄 Denoising audio...")
            denoised_audio, raw_denoised = self.denoise_audio(original_audio, sr)
            
            # Extract features from denoised audio
            print("📄 Extracting features...")
            features = self.extract_features(denoised_audio)
            print(f"🔊 Features extracted: {features.shape}")
            
            # Stage 1 classification
            print("📄 Running Stage 1 classification...")
            stage1_result = self.classify_stage1(features)
            print(f"🔊 Stage 1 result: {stage1_result['label']} ({stage1_result['confidence']:.2%})")
            
            # Stage 2 classification (only if abnormal)
            stage2_result = None
            if stage1_result['prediction'] == 1:  # Abnormal
                print("📄 Running Stage 2 classification...")
                stage2_result = self.classify_stage2(features)
                print(f"🔊 Stage 2 result: {stage2_result['label']} ({stage2_result['confidence']:.2%})")
            
            # Generate visualizations
            print("📄 Generating visualizations...")
            plots = self.generate_plots(original_audio, denoised_audio, features, sr)
            
            # Save denoised audio
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            denoised_filename = f"denoised_{timestamp}.wav"
            denoised_path = os.path.join(app.config['RESULTS_FOLDER'], denoised_filename)
            
            # Save at original sample rate for playback
            denoised_playback = librosa.resample(raw_denoised, orig_sr=self.denoise_sr, target_sr=sr)
            sf.write(denoised_path, denoised_playback, sr)
            
            # Verify file was saved correctly
            if os.path.exists(denoised_path):
                file_size = os.path.getsize(denoised_path)
                print(f"💾 Denoised audio saved: {denoised_filename} ({file_size} bytes)")
            else:
                print(f"❌ Failed to save denoised audio: {denoised_path}")
            
            return {
                'success': True,
                'original_duration': len(original_audio) / sr,
                'denoised_audio_path': denoised_filename,
                'stage1': stage1_result,
                'stage2': stage2_result,
                'plots': plots,
                'timestamp': timestamp
            }
            
        except Exception as e:
            print(f"❌ Error in analysis: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def generate_plots(self, original, denoised, features, sr):
        """Generate visualization plots"""
        try:
            plots = {}
            
            # Set style for better plots
            plt.style.use('default')
            
            # 1. Waveform comparison
            plt.figure(figsize=(14, 8))
            
            plt.subplot(2, 1, 1)
            time_orig = np.linspace(0, len(original)/sr, len(original))
            plt.plot(time_orig[:sr*5], original[:sr*5], color='#e74c3c', linewidth=0.8)
            plt.title('Original Audio Waveform (First 5 seconds)', fontsize=14, fontweight='bold')
            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude')
            plt.grid(True, alpha=0.3)
            
            plt.subplot(2, 1, 2)
            denoised_display = librosa.resample(denoised, orig_sr=self.target_sr, target_sr=sr)
            time_den = np.linspace(0, len(denoised_display)/sr, len(denoised_display))
            plt.plot(time_den[:sr*5], denoised_display[:sr*5], color='#27ae60', linewidth=0.8)
            plt.title('Denoised Audio Waveform (First 5 seconds)', fontsize=14, fontweight='bold')
            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight', facecolor='white')
            buffer.seek(0)
            plots['waveform'] = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            # 2. Feature visualization
            plt.figure(figsize=(16, 10))
            
            # MFCC
            plt.subplot(2, 2, 1)
            librosa.display.specshow(features[:13], x_axis='time', y_axis='mel', sr=self.target_sr, 
                                   hop_length=self.hop_length, cmap='viridis')
            plt.title('MFCC Features', fontsize=12, fontweight='bold')
            plt.colorbar(format='%+2.0f dB')
            
            # Chroma
            plt.subplot(2, 2, 2)
            librosa.display.specshow(features[40:52], x_axis='time', y_axis='chroma', sr=self.target_sr,
                                   hop_length=self.hop_length, cmap='plasma')
            plt.title('Chroma Features', fontsize=12, fontweight='bold')
            plt.colorbar()
            
            # Zero Crossing Rate
            plt.subplot(2, 2, 3)
            frames = range(len(features[39]))
            times = librosa.frames_to_time(frames, sr=self.target_sr, hop_length=self.hop_length)
            plt.plot(times, features[39], color='#f39c12', linewidth=1)
            plt.title('Zero Crossing Rate', fontsize=12, fontweight='bold')
            plt.xlabel('Time (s)')
            plt.ylabel('ZCR')
            plt.grid(True, alpha=0.3)
            
            # Melspectrogram (subset)
            plt.subplot(2, 2, 4)
            librosa.display.specshow(features[53:181][:64], x_axis='time', y_axis='mel', sr=self.target_sr,
                                   hop_length=self.hop_length, cmap='magma')
            plt.title('Mel-spectrogram (First 64 bins)', fontsize=12, fontweight='bold')
            plt.colorbar(format='%+2.0f dB')
            
            plt.tight_layout()
            
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight', facecolor='white')
            buffer.seek(0)
            plots['features'] = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return plots
            
        except Exception as e:
            print(f"❌ Error generating plots: {e}")
            return {}

# Initialize analyzer
analyzer = HeartSoundAnalyzer()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': analyzer.models_loaded,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/test')
def test_endpoint():
    """Simple test endpoint to verify server is working"""
    return jsonify({
        'message': 'Server is working',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        # Check if models are loaded
        if not analyzer.models_loaded:
            return jsonify({'success': False, 'error': 'Models not loaded. Please restart the server.'})
        
        if 'audio_file' not in request.files:
            return jsonify({'success': False, 'error': 'No file selected'})
        
        file = request.files['audio_file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'})
        
        # Check file size (additional check)
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)
        
        if file_size > app.config['MAX_CONTENT_LENGTH']:
            return jsonify({'success': False, 'error': f'File too large. Maximum size is {app.config["MAX_CONTENT_LENGTH"] // (1024*1024)}MB'})
        
        if file:
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{filename}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            print(f"📁 File uploaded: {filename} ({file_size} bytes)")
            
            # Analyze the audio
            result = analyzer.analyze_audio(file_path)
            
            # Clean up uploaded file
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"🗑️ Cleaned up: {filename}")
                else:
                    print(f"⚠️ File already cleaned up or not found: {file_path}")
            except Exception as cleanup_error:
                print(f"⚠️ Error during cleanup (non-critical): {cleanup_error}")
                pass
            
            return jsonify(result)
    
    except Exception as e:
        print(f"❌ Error in upload endpoint: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/download/<filename>')
def download_file(filename):
    try:
        file_path = os.path.join(app.config['RESULTS_FOLDER'], filename)
        print(f"🔍 Download request for: {filename}")
        print(f"🔍 File path: {file_path}")
        print(f"🔍 File exists: {os.path.exists(file_path)}")
        
        if os.path.exists(file_path):
            # Get file size for debugging
            file_size = os.path.getsize(file_path)
            print(f"🔍 File size: {file_size} bytes")
            
            from flask import Response
            import mimetypes
            
            # Get the correct MIME type
            mime_type, _ = mimetypes.guess_type(file_path)
            if not mime_type:
                mime_type = 'audio/wav'
            
            # Read the file and serve with proper headers
            with open(file_path, 'rb') as f:
                file_data = f.read()
            
            response = Response(
                file_data,
                mimetype=mime_type,
                headers={
                    'Content-Disposition': f'inline; filename="{filename}"',
                    'Accept-Ranges': 'bytes',
                    'Content-Length': str(len(file_data))
                }
            )
            return response
        else:
            print(f"❌ File not found: {file_path}")
            return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        print(f"❌ Download error: {e}")
        return jsonify({'error': str(e)}), 500

@app.errorhandler(413)
def too_large(e):
    return jsonify({'success': False, 'error': 'File too large'}), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(Exception)
def handle_exception(e):
    """Handle all unhandled exceptions and return JSON"""
    print(f"❌ Unhandled exception: {e}")
    return jsonify({'success': False, 'error': 'An unexpected error occurred'}), 500

if __name__ == '__main__':
    print("🎵 Heart Sound Analyzer - Starting Server...")
    print("=" * 50)
    
    # Load models FIRST
    print("📄 Loading models...")
    if analyzer.load_models():
        print("✅ All models loaded successfully!")
        print("=" * 50)
        print("🚀 Starting Flask server...")
        
        # Get port from environment variable (Render sets this)
        port = int(os.environ.get('PORT', 5001))
        debug = os.environ.get('FLASK_ENV') == 'development'
        
        print(f"🌐 Access the application at: http://localhost:{port}")
        print("=" * 50)
        
        # Run the app only once, after models are loaded
        app.run(debug=debug, host='0.0.0.0', port=port)
    else:
        print("❌ Failed to load models. Please check file paths.")
        print("\nExpected files:")
        print("1. Denoise model: ../Denoise/best_model_denoise.h5")
        print("2. Stage 1 model: stage1_results/heart_sound_cnn_20250901_231047.h5")
        print("3. Stage 1 scaler: stage1_results/heart_sound_pipeline_20250901_232643_scaler.pkl")
        print("4. Stage 2 model: stage2_results/abnormality_classification_20250903_140057.h5")
        print("5. Stage 2 scaler: stage2_results/abnormality_classification_pipeline_20250903_144628.pkl")
        print("\n⚠️ Starting server anyway - uploads will fail until models are loaded")
        
        # Start the server anyway so you can at least see the error page
        app.run(debug=debug, host='0.0.0.0', port=port)