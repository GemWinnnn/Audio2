import os
import librosa
import numpy as np
import json
import pandas as pd
from tqdm import tqdm
import shutil

# Feature extraction configuration
n_mfcc = 13  # Base MFCC coefficients (reduced from 26 to balance with other features)
target_sr = 12000  # Standardized sample rate
n_fft = 512  # FFT window size
duration = 12.0
max_len = int(target_sr * duration)
hop_length = 256
n_chroma = 12  # Number of chroma features
n_mels = 128  # Number of mel bands for melspectrogram

# Dataset paths
BASE_DIR = "/Users/gemwincanete/Audio2/datasets/merged_data"
OUTPUT_BASE = "/Users/gemwincanete/Audio2/datasets"

# Training limits per class
class_limits = {
    'normal': 1300,
    'murmur': 400,
    'extra_systole': 400,
    'extra_heart_audio': 400,
    'artifact': 400
}


# Initialize label mapping and metadata
label_map = {}
label_index = 0
metadata = []

def extract_comprehensive_features(signal):
    """Extract comprehensive audio features including MFCC, ZCR, Chroma, RMS, and Melspectrogram"""
    
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

def get_feature_info():
    """Return information about the feature structure"""
    feature_info = {
        'mfcc_base': {'start': 0, 'end': 13, 'description': 'Base MFCC coefficients'},
        'mfcc_delta': {'start': 13, 'end': 26, 'description': 'MFCC delta coefficients'},
        'mfcc_delta2': {'start': 26, 'end': 39, 'description': 'MFCC delta-delta coefficients'},
        'zcr': {'start': 39, 'end': 40, 'description': 'Zero Crossing Rate'},
        'chroma': {'start': 40, 'end': 52, 'description': 'Chroma features (pitch class profiles)'},
        'rms': {'start': 52, 'end': 53, 'description': 'Root Mean Square energy'},
        'melspectrogram': {'start': 53, 'end': 181, 'description': 'Mel-scale spectrogram (dB)'},
        'total_dimensions': 181
    }
    return feature_info

def process_split(split_name):
    """Process train or test split"""
    global label_index
    
    print(f"\n🔄 Processing {split_name.upper()} split...")
    
    for class_name in sorted(os.listdir(BASE_DIR)):
        class_path = os.path.join(BASE_DIR, class_name)
        if not os.path.isdir(class_path):
            continue
            
        split_path = os.path.join(class_path, split_name)
        if not os.path.isdir(split_path):
            continue
        
        # Create label mapping
        if class_name not in label_map:
            label_map[class_name] = label_index
            label_index += 1
        
        # Create output directory for this class
        output_class_dir = os.path.join(OUTPUT_BASE, class_name, split_name)
        os.makedirs(output_class_dir, exist_ok=True)
        
        # Get all wav files
        wav_files = [f for f in os.listdir(split_path) if f.endswith('.wav')]
        
        # Apply class limits for training data
        if split_name == "train" and class_name in class_limits:
            wav_files = wav_files[:class_limits[class_name]]
        
        print(f"  📁 {class_name}: Processing {len(wav_files)} files")
        
        # Process each audio file
        for wav_file in tqdm(wav_files, desc=f'  Extracting {class_name}', leave=False):
            try:
                # Load audio with target sample rate
                audio_path = os.path.join(split_path, wav_file)
                signal, _ = librosa.load(audio_path, sr=target_sr)
                
                # Standardize audio length
                if len(signal) > max_len:
                    signal = signal[:max_len]
                else:
                    signal = np.pad(signal, (0, max_len - len(signal)))
                
                # Extract comprehensive features
                combined_features = extract_comprehensive_features(signal)
                
                # Save features
                feature_filename = f"{os.path.splitext(wav_file)[0]}.npy"
                feature_path = os.path.join(output_class_dir, feature_filename)
                np.save(feature_path, combined_features)
                
                # Record metadata
                metadata.append({
                    'filename': feature_filename,
                    'original_file': wav_file,
                    'label': class_name,
                    'label_id': label_map[class_name],
                    'split': split_name,
                    'feature_shape': str(combined_features.shape),
                    'audio_length': len(signal),
                    'sample_rate': target_sr,
                    'feature_dimensions': combined_features.shape[0],
                    'time_frames': combined_features.shape[1]
                })
                
            except Exception as e:
                print(f"    ❌ Error processing {wav_file}: {e}")

# Process both splits
process_split("train")
process_split("test")

# Save metadata to CSV
metadata_df = pd.DataFrame(metadata)
metadata_df.to_csv(os.path.join(OUTPUT_BASE, 'metadata.csv'), index=False)

# Save label mapping
with open(os.path.join(OUTPUT_BASE, "label_map.json"), "w") as f:
    json.dump(label_map, f, indent=2)

# Save feature information
feature_info = get_feature_info()
with open(os.path.join(OUTPUT_BASE, "feature_info.json"), "w") as f:
    json.dump(feature_info, f, indent=2)

# Print summary statistics
print(f"\n✅ Enhanced feature extraction complete!")
print(f"📊 Summary:")
print(f"   • Output directory: {OUTPUT_BASE}")
print(f"   • Total files processed: {len(metadata)}")
print(f"   • Classes: {list(label_map.keys())}")
print(f"   • Feature dimensions: {feature_info['total_dimensions']}")
print(f"   • Sample rate: {target_sr} Hz")
print(f"   • Audio duration: {duration} seconds")

print(f"\n🔧 Feature breakdown:")
for feature_name, info in feature_info.items():
    if feature_name != 'total_dimensions':
        dim_count = info['end'] - info['start']
        print(f"   • {feature_name}: {dim_count} dimensions ({info['start']}-{info['end']-1}) - {info['description']}")

# Display class distribution
print(f"\n📈 Class distribution:")
if not metadata_df.empty and 'label' in metadata_df.columns and 'split' in metadata_df.columns:
    class_counts = metadata_df.groupby(['label', 'split']).size().unstack(fill_value=0)
    print(class_counts)
else:
    print("   ⚠️  No data processed or missing required columns in metadata")

print(f"\n💾 Files saved:")
print(f"   • Metadata: {OUTPUT_BASE}/metadata.csv")
print(f"   • Label mapping: {OUTPUT_BASE}/label_map.json")
print(f"   • Feature info: {OUTPUT_BASE}/feature_info.json")
print(f"   • Enhanced features: {OUTPUT_BASE}/[class]/[split]/[filename].npy")

print(f"\n📝 Feature extraction notes:")
print(f"   • MFCC: 39 features (13 base + 13 delta + 13 delta-delta)")
print(f"   • ZCR: 1 feature (signal transition rate)")
print(f"   • Chroma: 12 features (pitch class profiles)")
print(f"   • RMS: 1 feature (audio loudness measure)")
print(f"   • Melspectrogram: 128 features (mel-scale frequency representation)")
print(f"   • Total: 181 feature dimensions per time frame")