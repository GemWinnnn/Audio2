import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import hilbert, butter, filtfilt, spectrogram, resample, find_peaks

# --- Springer-like segmentation for denoised PCG files ---

# Configuration - Change this to process different datasets
DATASET_TYPE = 'extra_systole'  # Options: 'normal', 'murmur', 'extra_heart_audio', 'artifact', 'extra_systole'

# Base directory for denoised datasets
BASE_DIR = '/Users/gemwincanete/Audio2/datasets/Denoised'

# Construct input and output paths
INPUT_DIR = os.path.join(BASE_DIR, DATASET_TYPE, 'Train')
OUTPUT_DIR = os.path.join(INPUT_DIR, 'springer_segmentation_output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

STATE_LABELS = {1: 'S1', 2: 'systole', 3: 'S2', 4: 'diastole'}

# --- Feature extraction ---
def homomorphic_envelope(signal, fs, lpf_frequency=8):
    """Extract homomorphic envelope from PCG signal"""
    analytic = hilbert(signal)
    envelope = np.abs(analytic)
    log_env = np.log(envelope + 1e-10)
    # 1st order Butterworth lowpass
    b, a = butter(1, 2 * lpf_frequency / fs, btype='low')
    hom_env = np.exp(filtfilt(b, a, log_env))
    hom_env[0] = hom_env[1]  # Remove spike
    return hom_env

def hilbert_envelope(signal):
    """Extract Hilbert envelope"""
    return np.abs(hilbert(signal))

def psd_feature(signal, fs, low=40, high=60, feature_len=None):
    """Extract power spectral density feature in frequency band"""
    nperseg = min(int(fs / 40), len(signal) // 4)  # Ensure nperseg is not too large
    noverlap = nperseg // 2
    
    f, t, Sxx = spectrogram(signal, fs, nperseg=nperseg, noverlap=noverlap, scaling='spectrum')
    
    # Handle case where frequency range is outside signal spectrum
    if high > f[-1]:
        high = f[-1]
    if low > f[-1]:
        low = 0
        
    idx_low = np.argmin(np.abs(f - low))
    idx_high = np.argmin(np.abs(f - high))
    psd = np.mean(Sxx[idx_low:idx_high+1, :], axis=0)
    
    if feature_len is not None and len(psd) != feature_len:
        if len(psd) > 0:
            psd = resample(psd, feature_len)
        else:
            psd = np.zeros(feature_len)
    
    return psd

def normalise(x):
    """Normalize signal to zero mean, unit variance"""
    x = np.asarray(x)
    std_x = np.std(x)
    if std_x == 0:
        return np.zeros_like(x)
    return (x - np.mean(x)) / std_x

def extract_features(signal, fs, feature_fs=50):
    """Extract features from PCG signal"""
    print(f"  Signal length: {len(signal)}, sampling rate: {fs}")
    
    # Extract envelopes
    hom_env = homomorphic_envelope(signal, fs)
    hilb_env = hilbert_envelope(signal)
    
    # Calculate target length for downsampling
    target_len = int(len(signal) * feature_fs / fs)
    print(f"  Target feature length: {target_len}")
    
    # Downsample envelopes
    if target_len > 0:
        hom_env_ds = resample(hom_env, target_len)
        hilb_env_ds = resample(hilb_env, target_len)
    else:
        hom_env_ds = hom_env
        hilb_env_ds = hilb_env
        target_len = len(hom_env)
    
    # PSD feature
    psd = psd_feature(signal, fs, feature_len=target_len)
    
    # Normalise
    hom_env_ds = normalise(hom_env_ds)
    hilb_env_ds = normalise(hilb_env_ds)
    psd = normalise(psd)
    
    # Stack features
    features = np.stack([hom_env_ds, hilb_env_ds, psd], axis=1)
    print(f"  Features shape: {features.shape}")
    
    return features, feature_fs

# --- Heart rate estimation ---
def estimate_heart_rate(signal, fs):
    """Estimate heart rate using autocorrelation"""
    print("  Estimating heart rate...")
    
    hom_env = homomorphic_envelope(signal, fs)
    y = hom_env - np.mean(hom_env)
    
    # Autocorrelation
    c = np.correlate(y, y, mode='full')
    autocorr = c[len(y)-1:]
    
    # Look for peaks in reasonable heart rate range (30-200 bpm)
    min_idx = int(60 / 200 * fs)  # 200 bpm max
    max_idx = int(60 / 30 * fs)   # 30 bpm min
    
    if max_idx >= len(autocorr):
        max_idx = len(autocorr) - 1
    
    if min_idx >= max_idx:
        # Fallback values
        heart_rate = 75
        systolic_interval = 0.3
        print(f"  Using default HR: {heart_rate} bpm")
        return heart_rate, systolic_interval
    
    idx = np.argmax(autocorr[min_idx:max_idx]) + min_idx
    heart_rate = 60 / (idx / fs)
    
    # Systolic interval: max peak between 0.2s and half heart cycle
    max_sys = min(int((60 / heart_rate * fs) / 2), len(autocorr) - 1)
    min_sys = int(0.2 * fs)
    
    if min_sys >= max_sys:
        systolic_interval = 0.3
    else:
        pos = np.argmax(autocorr[min_sys:max_sys]) + min_sys
        systolic_interval = pos / fs
    
    print(f"  Estimated HR: {heart_rate:.1f} bpm, Systolic interval: {systolic_interval:.3f}s")
    return heart_rate, systolic_interval

# --- Heuristic segmentation (Springer-style, no HMM) ---
def segment_states(features, feature_fs, heart_rate, systolic_interval):
    """Segment PCG into cardiac states"""
    print("  Segmenting states...")
    
    # Use the homomorphic envelope to find peaks (S1/S2)
    env = features[:, 0]
    
    # Adaptive threshold based on signal statistics
    threshold = np.mean(env) + 0.5 * np.std(env)
    min_distance = max(int(feature_fs * 60 / heart_rate * 0.3), 1)  # Minimum distance between peaks
    
    print(f"  Peak detection threshold: {threshold:.3f}, min distance: {min_distance}")
    
    # Find all peaks
    peaks, properties = find_peaks(env, height=threshold, distance=min_distance)
    
    print(f"  Found {len(peaks)} peaks")
    
    if len(peaks) == 0:
        # No peaks found, create a simple alternating pattern
        print("  No peaks found, using simple alternating pattern")
        states = np.ones(len(env), dtype=int)
        cycle_len = int(feature_fs * 60 / heart_rate)
        for i in range(0, len(env), cycle_len):
            s1_len = int(0.122 * feature_fs)
            sys_len = int(systolic_interval * feature_fs)
            s2_len = int(0.092 * feature_fs)
            
            end_s1 = min(i + s1_len, len(env))
            end_sys = min(i + s1_len + sys_len, len(env))
            end_s2 = min(i + s1_len + sys_len + s2_len, len(env))
            end_dia = min(i + cycle_len, len(env))
            
            states[i:end_s1] = 1  # S1
            states[end_s1:end_sys] = 2  # systole
            states[end_sys:end_s2] = 3  # S2
            states[end_s2:end_dia] = 4  # diastole
        
        return states
    
    # Initialize states array
    states = np.ones(len(env), dtype=int)  # Default to S1
    
    # State durations (in samples)
    mean_S1 = max(int(0.122 * feature_fs), 1)
    mean_S2 = max(int(0.092 * feature_fs), 1)
    
    # Assign S1 and S2 at peaks, alternate starting with S1
    is_S1 = True
    last_idx = 0
    
    for i, peak in enumerate(peaks):
        if is_S1:
            # S1 state
            s1_start = max(peak - mean_S1//2, 0)
            s1_end = min(peak + mean_S1//2, len(env))
            states[s1_start:s1_end] = 1
            
            # Fill gap with diastole if there was a previous state
            if i > 0 and last_idx < s1_start:
                states[last_idx:s1_start] = 4  # diastole
            
            last_idx = s1_end
        else:
            # S2 state
            s2_start = max(peak - mean_S2//2, 0)
            s2_end = min(peak + mean_S2//2, len(env))
            states[s2_start:s2_end] = 3
            
            # Fill gap with systole
            if last_idx < s2_start:
                states[last_idx:s2_start] = 2  # systole
            
            last_idx = s2_end
        
        is_S1 = not is_S1
    
    # Fill the rest
    if last_idx < len(env):
        states[last_idx:] = 4 if is_S1 else 2
    
    # Count states
    unique, counts = np.unique(states, return_counts=True)
    print(f"  State distribution: {dict(zip([STATE_LABELS[u] for u in unique], counts))}")
    
    return states

# --- Output CSV ---
def write_state_csv(times, states, out_csv):
    """Write segmentation results to CSV"""
    print(f"  Writing CSV: {out_csv}")
    
    # Find transitions
    transitions = np.where(np.diff(states) != 0)[0] + 1  # +1 because diff reduces length by 1
    
    rows = []
    start_idx = 0
    
    for idx in transitions:
        if start_idx < len(states):
            rows.append({
                'start_time': times[start_idx],
                'end_time': times[idx-1] if idx < len(times) else times[-1],
                'state': STATE_LABELS[states[start_idx]]
            })
        start_idx = idx
    
    # Last segment
    if start_idx < len(states):
        rows.append({
            'start_time': times[start_idx],
            'end_time': times[-1],
            'state': STATE_LABELS[states[start_idx]]
        })
    
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"  Wrote {len(rows)} segments to CSV")

# --- Directory validation ---
def validate_input_directory(input_dir):
    """Validate that the input directory exists and contains WAV files"""
    if not os.path.exists(input_dir):
        print(f"ERROR: Input directory does not exist: {input_dir}")
        return False
    
    wav_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.wav')]
    if len(wav_files) == 0:
        print(f"ERROR: No WAV files found in directory: {input_dir}")
        return False
    
    print(f"Found {len(wav_files)} WAV files in {input_dir}")
    return True

# --- Main processing loop ---
def main():
    print(f"Processing dataset: {DATASET_TYPE}")
    print(f"Input directory: {INPUT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Validate input directory
    if not validate_input_directory(INPUT_DIR):
        return
    
    # Get list of WAV files
    wav_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith('.wav')]
    print(f"Found {len(wav_files)} WAV files")
    
    if len(wav_files) == 0:
        print("No WAV files found in input directory")
        return
    
    for fname in wav_files:
        wav_path = os.path.join(INPUT_DIR, fname)
        print(f'\nProcessing {fname}')
        
        try:
            # Load audio
            y, fs = librosa.load(wav_path, sr=None)
            print(f"  Loaded audio: {len(y)} samples at {fs} Hz ({len(y)/fs:.2f}s)")
            
            if len(y) == 0:
                print(f"  WARNING: Empty audio file, skipping")
                continue
            
            # Extract features
            features, feature_fs = extract_features(y, fs)
            
            # Estimate heart rate
            heart_rate, systolic_interval = estimate_heart_rate(y, fs)
            
            # Segment states
            states = segment_states(features, feature_fs, heart_rate, systolic_interval)
            
            # Create time vector for features
            times = np.linspace(0, len(y)/fs, len(states))
            
            # Plot results
            plt.figure(figsize=(16, 10))
            
            # Create time vectors
            t_audio = np.linspace(0, len(y)/fs, len(y))
            
            # Main plot: Audio with state overlay
            plt.subplot(3, 1, 1)
            plt.plot(t_audio, y, 'b-', alpha=0.8, linewidth=0.8, label='PCG Signal')
            
            # Color-code the background by cardiac state
            colors = {1: 'red', 2: 'orange', 3: 'purple', 4: 'lightblue'}
            state_names = {1: 'S1', 2: 'Systole', 3: 'S2', 4: 'Diastole'}
            
            # Interpolate states to match audio length for background coloring
            states_full = np.interp(t_audio, times, states)
            
            for state_val in [1, 2, 3, 4]:
                mask = np.abs(states_full - state_val) < 0.1
                if np.any(mask):
                    plt.fill_between(t_audio, plt.ylim()[0], plt.ylim()[1], 
                                   where=mask, alpha=0.2, color=colors[state_val], 
                                   label=f'{state_names[state_val]} regions')
            
            plt.ylabel('Amplitude')
            plt.title(f'{fname} - PCG Signal with Cardiac State Regions (HR: {heart_rate:.1f} bpm)')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            
            # Feature envelope with state markers
            plt.subplot(3, 1, 2)
            plt.plot(times, features[:, 0], 'darkred', linewidth=2, label='Homomorphic Envelope')
            
            # Mark state transitions with vertical lines
            state_changes = np.where(np.diff(states) != 0)[0]
            for change_idx in state_changes:
                if change_idx < len(times):
                    plt.axvline(x=times[change_idx], color='gray', linestyle='--', alpha=0.7)
            
            # Add state labels at the top
            current_state = states[0]
            start_idx = 0
            for i, state in enumerate(states):
                if state != current_state or i == len(states) - 1:
                    mid_time = times[start_idx + (i - start_idx) // 2]
                    plt.text(mid_time, max(features[:, 0]) * 0.9, 
                           state_names[current_state], 
                           ha='center', va='center', 
                           bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[current_state], alpha=0.7),
                           fontsize=9, fontweight='bold')
                    current_state = state
                    start_idx = i
            
            plt.ylabel('Envelope Amplitude')
            plt.title('Homomorphic Envelope with State Segmentation')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # State timeline as a cleaner visualization
            plt.subplot(3, 1, 3)
            
            # Create step plot for states
            for i in range(len(times)-1):
                state_val = states[i]
                plt.fill_between([times[i], times[i+1]], 0, 1, 
                               color=colors[state_val], alpha=0.8, 
                               edgecolor='black', linewidth=0.5)
                
                # Add text labels for longer segments
                segment_duration = times[i+1] - times[i]
                if segment_duration > 0.1:  # Only label segments longer than 0.1s
                    mid_time = (times[i] + times[i+1]) / 2
                    plt.text(mid_time, 0.5, state_names[state_val], 
                           ha='center', va='center', fontweight='bold', fontsize=10)
            
            plt.ylim(0, 1)
            plt.ylabel('Cardiac States')
            plt.xlabel('Time (s)')
            plt.title('Cardiac State Timeline')
            
            # Create custom legend
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor=colors[i], label=state_names[i]) 
                             for i in [1, 2, 3, 4]]
            plt.legend(handles=legend_elements, loc='upper right')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot
            plot_path = os.path.join(OUTPUT_DIR, fname.replace('.wav', '_segmentation.png'))
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  Saved plot: {plot_path}")
            
            # Save CSV
            out_csv = os.path.join(OUTPUT_DIR, fname.replace('.wav', '_states.csv'))
            write_state_csv(times, states, out_csv)
            
        except Exception as e:
            print(f"  ERROR processing {fname}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    print('\nDone!')

if __name__ == "__main__":
    main()