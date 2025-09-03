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
BASE_DIR = '/Users/gemwincanete/Audio/thesis/datasets/Denoised'

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

# --- Improved heuristic segmentation ---
def segment_states(features, feature_fs, heart_rate, systolic_interval):
    """Segment PCG into cardiac states with proper end handling"""
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
        return create_alternating_pattern(len(env), feature_fs, heart_rate, systolic_interval)
    
    # Calculate cardiac cycle duration
    cycle_duration = 60 / heart_rate  # in seconds
    cycle_samples = int(cycle_duration * feature_fs)
    
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
    
    # IMPROVED: Handle the end of the recording more intelligently
    if last_idx < len(env):
        remaining_samples = len(env) - last_idx
        remaining_time = remaining_samples / feature_fs
        
        print(f"  Handling end segment: {remaining_time:.3f}s remaining")
        
        # Determine expected state based on cardiac cycle timing
        if len(peaks) >= 2:
            # Calculate expected next peak timing based on previous peaks
            last_peak = peaks[-1]
            if len(peaks) >= 2:
                peak_interval = peaks[-1] - peaks[-2]
            else:
                peak_interval = cycle_samples
            
            expected_next_peak = last_peak + peak_interval
            time_to_next_peak = (expected_next_peak - last_idx) / feature_fs
            
            # Decide state based on timing and what the last peak was
            if is_S1:  # Next peak should be S1
                if time_to_next_peak <= 0.15:  # Very close to next S1
                    # Finish current diastole and start S1
                    s1_start = max(0, int(expected_next_peak - mean_S1//2))
                    if s1_start > last_idx:
                        states[last_idx:s1_start] = 4  # diastole
                        states[s1_start:] = 1  # S1
                    else:
                        states[last_idx:] = 1  # S1
                elif remaining_time > cycle_duration * 0.7:  # Long remaining segment
                    # Complete a full cycle pattern
                    states = complete_cycle_pattern(states, last_idx, remaining_samples, 
                                                  feature_fs, heart_rate, systolic_interval, start_with_diastole=True)
                else:
                    # Just diastole
                    states[last_idx:] = 4
            else:  # Next peak should be S2
                if time_to_next_peak <= 0.15:  # Very close to next S2
                    # Finish current systole and start S2
                    s2_start = max(0, int(expected_next_peak - mean_S2//2))
                    if s2_start > last_idx:
                        states[last_idx:s2_start] = 2  # systole
                        states[s2_start:] = 3  # S2
                    else:
                        states[last_idx:] = 3  # S2
                elif remaining_time > systolic_interval + 0.2:  # Long enough for systole + S2
                    # Complete systole -> S2 -> diastole pattern
                    sys_duration = int(systolic_interval * feature_fs)
                    s2_start = last_idx + sys_duration
                    s2_end = min(s2_start + mean_S2, len(env))
                    
                    states[last_idx:s2_start] = 2  # systole
                    states[s2_start:s2_end] = 3    # S2
                    if s2_end < len(env):
                        states[s2_end:] = 4        # diastole
                else:
                    # Just systole
                    states[last_idx:] = 2
        else:
            # Fallback: use simple pattern based on expected next state
            if remaining_time > 0.3:  # Significant remaining time
                states = complete_cycle_pattern(states, last_idx, remaining_samples, 
                                              feature_fs, heart_rate, systolic_interval, 
                                              start_with_diastole=is_S1)
            else:
                # Short segment, assign based on expected next state
                states[last_idx:] = 4 if is_S1 else 2
    
    # Count states
    unique, counts = np.unique(states, return_counts=True)
    print(f"  State distribution: {dict(zip([STATE_LABELS[u] for u in unique], counts))}")
    
    return states

def create_alternating_pattern(length, feature_fs, heart_rate, systolic_interval):
    """Create alternating S1-systole-S2-diastole pattern"""
    states = np.ones(length, dtype=int)
    cycle_len = int(feature_fs * 60 / heart_rate)
    
    for i in range(0, length, cycle_len):
        s1_len = int(0.122 * feature_fs)
        sys_len = int(systolic_interval * feature_fs)
        s2_len = int(0.092 * feature_fs)
        
        # Ensure we don't exceed array bounds
        end_s1 = min(i + s1_len, length)
        end_sys = min(i + s1_len + sys_len, length)
        end_s2 = min(i + s1_len + sys_len + s2_len, length)
        end_dia = min(i + cycle_len, length)
        
        # Only assign if indices are valid
        if i < length:
            states[i:end_s1] = 1  # S1
        if end_s1 < length:
            states[end_s1:end_sys] = 2  # systole
        if end_sys < length:
            states[end_sys:end_s2] = 3  # S2
        if end_s2 < length:
            states[end_s2:end_dia] = 4  # diastole
    
    return states

def complete_cycle_pattern(states, start_idx, remaining_samples, feature_fs, heart_rate, systolic_interval, start_with_diastole=True):
    """Complete a cardiac cycle pattern for the remaining segment"""
    s1_len = max(int(0.122 * feature_fs), 1)
    sys_len = max(int(systolic_interval * feature_fs), 1)
    s2_len = max(int(0.092 * feature_fs), 1)
    
    current_idx = start_idx
    end_idx = start_idx + remaining_samples
    
    if start_with_diastole:
        # Start with diastole -> S1 -> systole -> S2 -> diastole...
        pattern = [4, 1, 2, 3]  # diastole, S1, systole, S2
        durations = [None, s1_len, sys_len, s2_len]  # diastole duration calculated
    else:
        # Start with systole -> S2 -> diastole -> S1...
        pattern = [2, 3, 4, 1]  # systole, S2, diastole, S1
        durations = [sys_len, s2_len, None, s1_len]  # diastole duration calculated
    
    pattern_idx = 0
    
    while current_idx < end_idx:
        state = pattern[pattern_idx]
        
        if durations[pattern_idx] is None:  # Diastole - fill remaining time in cycle
            # Calculate remaining time in current cycle
            cycle_duration = 60 / heart_rate
            cycle_samples = int(feature_fs * cycle_duration)
            used_samples = s1_len + sys_len + s2_len
            diastole_samples = max(cycle_samples - used_samples, 1)
            duration = min(diastole_samples, end_idx - current_idx)
        else:
            duration = min(durations[pattern_idx], end_idx - current_idx)
        
        next_idx = min(current_idx + duration, end_idx)
        states[current_idx:next_idx] = state
        
        current_idx = next_idx
        pattern_idx = (pattern_idx + 1) % len(pattern)
    
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

# --- Audio activity detection ---
def detect_audio_end(signal, fs, window_size=0.5, threshold_factor=0.1, min_silence_duration=1.0):
    """
    Detect where meaningful audio activity ends in the signal
    
    Args:
        signal: Audio signal
        fs: Sampling rate
        window_size: Window size in seconds for energy calculation
        threshold_factor: Fraction of max energy to use as threshold
        min_silence_duration: Minimum duration of silence to consider as end
    
    Returns:
        end_idx: Index where audio activity ends, or len(signal) if no clear end
    """
    print("  Detecting end of audio activity...")
    
    # Calculate windowed energy
    window_samples = int(window_size * fs)
    hop_samples = window_samples // 4
    
    energies = []
    times = []
    
    for i in range(0, len(signal) - window_samples + 1, hop_samples):
        window = signal[i:i + window_samples]
        energy = np.sum(window**2)
        energies.append(energy)
        times.append(i / fs)
    
    energies = np.array(energies)
    times = np.array(times)
    
    if len(energies) == 0:
        return len(signal)
    
    # Smooth the energy curve
    if len(energies) > 5:
        from scipy.ndimage import gaussian_filter1d
        energies_smooth = gaussian_filter1d(energies, sigma=2)
    else:
        energies_smooth = energies
    
    # Find threshold based on maximum energy
    max_energy = np.max(energies_smooth)
    threshold = max_energy * threshold_factor
    
    print(f"  Energy threshold: {threshold:.2e} (max: {max_energy:.2e})")
    
    # Find the last point where energy is above threshold
    above_threshold = energies_smooth > threshold
    
    if not np.any(above_threshold):
        # No significant activity found
        print("  No significant audio activity detected")
        return len(signal)
    
    # Find last significant activity
    last_active_idx = np.where(above_threshold)[0][-1]
    last_active_time = times[last_active_idx]
    
    # Check if there's enough silence after this point
    silence_duration = (len(signal) / fs) - last_active_time
    
    if silence_duration >= min_silence_duration:
        # Convert back to sample index with some padding
        padding_time = 0.2  # 200ms padding after last activity
        end_time = min(last_active_time + padding_time, len(signal) / fs)
        end_idx = int(end_time * fs)
        
        print(f"  Audio activity ends at {end_time:.2f}s (silence duration: {silence_duration:.2f}s)")
        return end_idx
    else:
        print(f"  Audio activity continues until end (silence duration: {silence_duration:.2f}s)")
        return len(signal)

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
                
            # Detect where audio activity ends
            audio_end_idx = detect_audio_end(y, fs)
            
            # Truncate signal to where activity ends
            if audio_end_idx < len(y):
                y_active = y[:audio_end_idx]
                actual_duration = len(y_active) / fs
                print(f"  Truncated to active portion: {len(y_active)} samples ({actual_duration:.2f}s)")
            else:
                y_active = y
                actual_duration = len(y) / fs
                print(f"  Using full signal: {actual_duration:.2f}s")
            
            # Skip if the active portion is too short
            if actual_duration < 2.0:  # Less than 2 seconds of activity
                print(f"  WARNING: Active portion too short ({actual_duration:.2f}s), skipping")
                continue
            
            # Extract features from active portion only
            features, feature_fs = extract_features(y_active, fs)
            
            # Estimate heart rate from active portion
            heart_rate, systolic_interval = estimate_heart_rate(y_active, fs)
            
            # Segment states
            states = segment_states(features, feature_fs, heart_rate, systolic_interval)
            
            # Create time vector for features (based on active portion duration)
            times = np.linspace(0, len(y_active)/fs, len(states))
            
            # Plot results
            plt.figure(figsize=(16, 10))
            
            # Create time vectors
            t_audio_full = np.linspace(0, len(y)/fs, len(y))  # Full recording
            t_audio_active = np.linspace(0, len(y_active)/fs, len(y_active))  # Active portion
            
            # Main plot: Full audio with active portion highlighted
            plt.subplot(4, 1, 1)
            plt.plot(t_audio_full, y, 'lightgray', alpha=0.6, linewidth=0.5, label='Full Recording')
            plt.plot(t_audio_active, y_active, 'b-', alpha=0.8, linewidth=0.8, label='Active Portion')
            
            # Mark the end of active portion
            if audio_end_idx < len(y):
                plt.axvline(x=len(y_active)/fs, color='red', linestyle='--', linewidth=2, 
                           label=f'Activity End ({len(y_active)/fs:.2f}s)')
            
            plt.ylabel('Amplitude')
            plt.title(f'{fname} - Full Recording vs Active Portion (HR: {heart_rate:.1f} bpm)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Second plot: Active audio with state overlay
            plt.subplot(4, 1, 2)
            plt.plot(t_audio_active, y_active, 'b-', alpha=0.8, linewidth=0.8, label='PCG Signal (Active)')
            
            # Color-code the background by cardiac state
            colors = {1: 'red', 2: 'orange', 3: 'purple', 4: 'lightblue'}
            state_names = {1: 'S1', 2: 'Systole', 3: 'S2', 4: 'Diastole'}
            
            # Interpolate states to match audio length for background coloring
            states_full = np.interp(t_audio_active, times, states)
            
            for state_val in [1, 2, 3, 4]:
                mask = np.abs(states_full - state_val) < 0.1
                if np.any(mask):
                    plt.fill_between(t_audio_active, plt.ylim()[0], plt.ylim()[1], 
                                   where=mask, alpha=0.2, color=colors[state_val], 
                                   label=f'{state_names[state_val]} regions')
            
            plt.ylabel('Amplitude')
            plt.title('Active PCG Signal with Cardiac State Regions')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            
            # Feature envelope with state markers
            plt.subplot(4, 1, 3)
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
            plt.subplot(4, 1, 4)
            
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