import pandas as pd
import numpy as np
import os
import random
import soundfile as sf
from scipy import signal
from scipy.interpolate import CubicSpline
import librosa
import torch
import torch.nn.functional as F
from pathlib import Path
import shutil
import warnings

def load_audio_file(file_path):
    """Load audio file and return the signal"""
    try:
        audio, sr = sf.read(file_path)
        return audio, sr
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None

def save_audio_file(audio, sr, file_path):
    """Save audio file"""
    try:
        sf.write(file_path, audio, sr)
        return True
    except Exception as e:
        print(f"Error saving {file_path}: {e}")
        return False

def get_cardiac_segments(springer_df, filename):
    """
    Extract cardiac phase segments from springer segmentation data
    Returns dict with segments organized by cardiac phase
    """
    # Extract just the filename without extension from the source_filename
    base_filename = filename.replace('.wav', '')
    
    # Filter by source_filename (which contains the base filename)
    file_data = springer_df[springer_df['source_filename'].str.contains(base_filename, na=False)]
    
    if len(file_data) == 0:
        return None
    
    # Group segments by cardiac cycles
    cardiac_cycles = []
    current_cycle = {'S1': [], 'systole': [], 'S2': [], 'diastole': []}
    
    for _, row in file_data.iterrows():
        state = row['state']
        start_time = row['start_time']
        end_time = row['end_time']
        
        if state in current_cycle:
            current_cycle[state].append((start_time, end_time))
            
            # If we completed a cycle (found diastole), start a new cycle
            if state == 'diastole':
                cardiac_cycles.append(current_cycle.copy())
                current_cycle = {'S1': [], 'systole': [], 'S2': [], 'diastole': []}
    
    return cardiac_cycles

def optimal_displacement_max_sum(s1, s2, alpha):
    """
    Find optimal displacement for aligning two segments based on maximum sum
    """
    len_s1 = len(s1)
    len_s2 = len(s2)
    
    if len_s1 > len_s2:
        # Zero pad s2 to the length of s1
        s2_padded = np.pad(s2, (0, len_s1 - len_s2), 'constant')
        max_sum = float('-inf')
        opt_displacement = 0
        
        # Iterate over all possible displacements
        for displacement in range(len_s1 - len_s2 + 1):
            # Create the current shifted version of s2
            current_s2_shifted = np.roll(s2_padded, displacement)
            # Calculate the sum with mixing coefficient
            current_sum = (np.sum(s1[:displacement]) + 
                          np.sum(s1[displacement:displacement+len_s2]*alpha + 
                                current_s2_shifted[displacement:displacement+len_s2]*(1-alpha)) + 
                          np.sum(s1[displacement+len_s2:]))
            
            if current_sum > max_sum:
                max_sum = current_sum
                opt_displacement = displacement
                
    else:  # len_s1 <= len_s2
        # Zero pad s1 to the length of s2
        s1_padded = np.pad(s1, (0, len_s2 - len_s1), 'constant')
        max_sum = float('-inf')
        opt_displacement = 0
        
        # Iterate over all possible displacements
        for displacement in range(len_s2 - len_s1 + 1):
            # Create the current shifted version of s1
            current_s1_shifted = np.roll(s1_padded, displacement)
            # Calculate the sum with mixing coefficient
            current_sum = np.sum(current_s1_shifted[displacement:displacement+len_s1]*alpha + 
                                s2[displacement:displacement+len_s1]*(1-alpha))
            
            if current_sum > max_sum:
                max_sum = current_sum
                opt_displacement = displacement
    
    return opt_displacement

def cardiac_aware_mixup(audio1, audio2, cycles1, cycles2, sr, alpha=0.5, method='basic'):
    """
    Perform cardiac-aware mixup between two audio signals using segmentation
    """
    if cycles1 is None or cycles2 is None or len(cycles1) == 0 or len(cycles2) == 0:
        # Fallback to simple mixup if no segmentation available
        min_len = min(len(audio1), len(audio2))
        return alpha * audio1[:min_len] + (1 - alpha) * audio2[:min_len]
    
    # Start with audio1 as base
    mixed_audio = audio1.copy()
    
    # Mix corresponding cardiac cycles
    num_cycles = min(len(cycles1), len(cycles2))
    
    for cycle_idx in range(num_cycles):
        cycle1 = cycles1[cycle_idx]
        cycle2 = cycles2[cycle_idx]
        
        # Mix each cardiac phase
        for phase in ['S1', 'systole', 'S2', 'diastole']:
            if cycle1[phase] and cycle2[phase]:
                # Take first segment of each phase (in case of multiple segments)
                start1, end1 = cycle1[phase][0]
                start2, end2 = cycle2[phase][0]
                
                # Convert to sample indices
                start_idx1 = int(start1 * sr)
                end_idx1 = int(end1 * sr)
                start_idx2 = int(start2 * sr)
                end_idx2 = int(end2 * sr)
                
                # Extract segments
                segment1 = audio1[start_idx1:end_idx1]
                segment2 = audio2[start_idx2:end_idx2]
                
                if len(segment1) == 0 or len(segment2) == 0:
                    continue
                
                # Apply mixing based on method
                if method == 'optimal' and len(segment1) != len(segment2):
                    # Use optimal displacement for different length segments
                    displacement = optimal_displacement_max_sum(segment1, segment2, alpha)
                    
                    if len(segment1) > len(segment2):
                        # Mix with optimal displacement
                        mixed_segment = segment1.copy()
                        end_mix = min(displacement + len(segment2), len(segment1))
                        mix_len = end_mix - displacement
                        mixed_segment[displacement:displacement+mix_len] = (
                            alpha * segment1[displacement:displacement+mix_len] + 
                            (1-alpha) * segment2[:mix_len]
                        )
                    else:
                        # segment2 is longer
                        mixed_segment = (alpha * segment1 + 
                                       (1-alpha) * segment2[displacement:displacement+len(segment1)])
                        
                else:
                    # Basic mixing - use minimum length
                    min_len = min(len(segment1), len(segment2))
                    mixed_segment = (alpha * segment1[:min_len] + 
                                   (1-alpha) * segment2[:min_len])
                
                # Replace the segment in mixed_audio
                if len(mixed_segment) > 0:
                    end_replace = min(start_idx1 + len(mixed_segment), len(mixed_audio))
                    replace_len = end_replace - start_idx1
                    mixed_audio[start_idx1:end_replace] = mixed_segment[:replace_len]
    
    return mixed_audio

def magnitude_warp(audio, sigma=0.2, knot=4):
    """
    Apply magnitude warping to audio signal
    """
    if len(audio) == 0:
        return audio
        
    orig_steps = np.arange(len(audio))
    random_warps = np.random.normal(loc=1.0, scale=sigma, size=(knot+2,))
    warp_steps = np.linspace(0, len(audio)-1., num=knot+2)
    
    try:
        warper = CubicSpline(warp_steps, random_warps)(orig_steps)
        return audio * warper
    except:
        # Fallback if CubicSpline fails
        return audio

def cardiac_aware_durmixmagwarp(audio1, audio2, cycles1, cycles2, sr, alpha=0.5, sigma=0.2, knot=4, method='basic'):
    """
    Perform cardiac-aware duration-preserving mixup with magnitude warping
    """
    # First apply cardiac-aware mixup
    mixed_audio = cardiac_aware_mixup(audio1, audio2, cycles1, cycles2, sr, alpha, method)
    
    # Then apply magnitude warping
    warped_audio = magnitude_warp(mixed_audio, sigma, knot)
    
    return warped_audio

def find_audio_file(filename, search_dir="datasets/Denoised"):
    """
    Find the actual audio file path for a given filename
    """
    # Use absolute path to ensure we're looking in the right place
    if not os.path.isabs(search_dir):
        search_dir = os.path.join(os.getcwd(), search_dir)
    
    # Add .wav extension if not present
    if not filename.endswith('.wav'):
        filename = filename + '.wav'
    
    for root, dirs, files in os.walk(search_dir):
        for file in files:
            if file.endswith('.wav'):
                # Check for exact match first (filename.wav)
                if file == filename:
                    return os.path.join(root, file)
                # Then check if filename is contained in the file name
                elif filename.replace('.wav', '') in file.replace('.wav', ''):
                    return os.path.join(root, file)
    return None

def create_cardiac_aware_augmentations(category_targets=None):
    """
    Create cardiac-aware augmentations: PCGmix and PCGmix+ with segmentation
    Balanced augmentation version with specific targets per category
    
    Args:
        category_targets: Dict mapping category names to number of augmentations to create
                         If None, uses recommended targets based on dataset analysis
    """
    
    # Load the springer segmentation data
    springer_file = "/Users/gemwincanete/Audio2/merged_springer_segmentation_data.csv"
    if not os.path.exists(springer_file):
        print(f"Springer segmentation file not found: {springer_file}")
        return
    
    springer_df = pd.read_csv(springer_file)
    print(f"Loaded segmentation data with {len(springer_df)} entries")
    
    # Get unique categories from the CSV
    available_categories = springer_df['category'].unique()
    print(f"Available categories: {available_categories}")
    
    # Define augmentation targets per category based on dataset analysis
    if category_targets is None:
        category_targets = {
            'normal': 0,          
            'murmur': 80,           # Add ~80 augmentations (current: 88 train)
            'extra_systole': 80,    # Add ~80 augmentations (current: 32 train)
            'extra_heart_audio': 85, # Add ~85 augmentations (current: 13 train)
            'artifact': 60          # Add ~60 augmentations (current: 28 train)
        }
    
    print(f"\nAugmentation targets per category:")
    for category, target in category_targets.items():
        if target > 0:
            print(f"  {category}: {target} augmentations")
        else:
            print(f"  {category}: No augmentation (sufficient data)")
    
    # Filter to only categories that exist in CSV and need augmentation
    categories_to_augment = [cat for cat in category_targets.keys() 
                           if cat in available_categories and category_targets[cat] > 0]
    print(f"Categories to augment: {categories_to_augment}")
    
    # Create output directory
    output_dir = "Cardiac_Aware_Augmented_data"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create subdirectories for each augmentation type
    pcgmix_dir = os.path.join(output_dir, "PCGmix_cardiac_aware")
    pcgmix_plus_dir = os.path.join(output_dir, "PCGmix_plus_cardiac_aware")
    os.makedirs(pcgmix_dir, exist_ok=True)
    os.makedirs(pcgmix_plus_dir, exist_ok=True)
    
    # Create subdirectories for each category that will be augmented
    for category in categories_to_augment:
        os.makedirs(os.path.join(pcgmix_dir, category), exist_ok=True)
        os.makedirs(os.path.join(pcgmix_plus_dir, category), exist_ok=True)
    
    # Process each category
    total_stats = {}
    
    for category in categories_to_augment:
        print(f"\nProcessing category: {category}")
        
        # Get all unique filenames for this category
        category_data = springer_df[springer_df['category'] == category]
        category_files = category_data['source_filename'].unique().tolist()
        
        # Extract base filenames (remove _states.csv extension)
        category_files = [f.replace('_states.csv', '') for f in category_files]
        
        if len(category_files) < 2:
            print(f"Not enough files for category {category} (need at least 2, found {len(category_files)})")
            total_stats[category] = {'files': len(category_files), 'target': category_targets[category], 'attempts': 0, 'augmentations': 0, 'cardiac_aware': 0}
            continue
        
        print(f"Found {len(category_files)} files for {category}")
        
        # Get target number of augmentations for this category
        target_augmentations = category_targets[category]
        print(f"Target: {target_augmentations} augmentations")
        
        augmentations_created = 0
        successful_cardiac_mixing = 0
        attempts = 0
        max_attempts = target_augmentations * 5  # Allow multiple attempts for failures
        
        while augmentations_created < target_augmentations and attempts < max_attempts:
            try:
                attempts += 1
                
                if attempts % 50 == 0:  # Progress indicator
                    print(f"  Progress: {augmentations_created}/{target_augmentations} created, attempt {attempts}")
                
                # Randomly select two different files (allows repetition across pairs)
                file1, file2 = random.sample(category_files, 2)
                
                # Find the actual audio files
                audio_file1 = find_audio_file(file1)
                audio_file2 = find_audio_file(file2)
                
                if not audio_file1 or not audio_file2:
                    print(f"Could not find audio files for {file1} or {file2}")
                    continue
                
                # Load audio files
                audio1, sr1 = load_audio_file(audio_file1)
                audio2, sr2 = load_audio_file(audio_file2)
                
                if audio1 is None or audio2 is None:
                    continue
                
                # Ensure same sampling rate
                if sr1 != sr2:
                    audio2 = librosa.resample(audio2, orig_sr=sr2, target_sr=sr1)
                    sr2 = sr1
                
                # Get cardiac segments for both files
                cycles1 = get_cardiac_segments(springer_df, file1)
                cycles2 = get_cardiac_segments(springer_df, file2)
                
                # Random alpha between 0.3 and 0.7
                alpha = random.uniform(0.3, 0.7)
                
                # Choose mixing method (basic or optimal)
                method = 'optimal' if random.random() > 0.5 else 'basic'
                
                # Create PCGmix (cardiac-aware mixup) augmentation
                pcgmix_audio_result = cardiac_aware_mixup(
                    audio1, audio2, cycles1, cycles2, sr1, alpha, method
                )
                
                # Create PCGmix+ (cardiac-aware mixup + magnitude warping) augmentation
                pcgmix_plus_audio_result = cardiac_aware_durmixmagwarp(
                    audio1, audio2, cycles1, cycles2, sr1, alpha, sigma=0.2, knot=4, method=method
                )
                
                # Save PCGmix augmentation
                pcgmix_filename = f"{file1}_cardiac_pcgmix_{file2}_alpha{alpha:.2f}_{method}_{attempts}.wav"
                pcgmix_path = os.path.join(pcgmix_dir, category, pcgmix_filename)
                pcgmix_saved = save_audio_file(pcgmix_audio_result, sr1, pcgmix_path)
                
                # Save PCGmix+ augmentation (only if we still need more)
                pcgmix_plus_saved = True
                if augmentations_created < target_augmentations:
                    pcgmix_plus_filename = f"{file1}_cardiac_pcgmix_plus_{file2}_alpha{alpha:.2f}_{method}_{attempts}.wav"
                    pcgmix_plus_path = os.path.join(pcgmix_plus_dir, category, pcgmix_plus_filename)
                    pcgmix_plus_saved = save_audio_file(pcgmix_plus_audio_result, sr1, pcgmix_plus_path)
                
                if pcgmix_saved:
                    augmentations_created += 1
                    if pcgmix_plus_saved and augmentations_created < target_augmentations:
                        augmentations_created += 1
                    
                    # Check if cardiac segmentation was used
                    if cycles1 is not None and cycles2 is not None:
                        successful_cardiac_mixing += 1
                    
                    if augmentations_created <= 3:  # Show first 3 examples for verification
                        cardiac_status = "with cardiac segmentation" if (cycles1 and cycles2) else "fallback to simple mixing"
                        print(f"  Created: {file1} x {file2} (alpha={alpha:.2f}, {method}) -> {cardiac_status}")
                        
            except Exception as e:
                print(f"Error processing {file1} x {file2}: {e}")
                continue
        
        # Store statistics
        total_stats[category] = {
            'files': len(category_files),
            'target': target_augmentations,
            'attempts': attempts,
            'augmentations': augmentations_created,
            'cardiac_aware': successful_cardiac_mixing
        }
        
        print(f"Category {category} completed:")
        print(f"  Files: {len(category_files)}")
        print(f"  Target: {target_augmentations}")
        print(f"  Attempts: {attempts}")
        print(f"  Augmentations created: {augmentations_created}")
        print(f"  Success rate: {(augmentations_created/attempts*100):.1f}%")
        print(f"  Cardiac-aware augmentations: {successful_cardiac_mixing}")
    
    print(f"\nCardiac-aware augmentation complete! Results saved in {output_dir}/")
    
    # Print comprehensive summary
    print("\n" + "="*80)
    print("CARDIAC-AWARE AUGMENTATION SUMMARY")
    print("="*80)
    
    total_files = 0
    total_targets = 0
    total_attempts = 0
    total_augmentations = 0
    total_cardiac_aware = 0
    
    for category in categories_to_augment:
        if category in total_stats:
            stats = total_stats[category]
            pcgmix_count = len(os.listdir(os.path.join(pcgmix_dir, category))) if os.path.exists(os.path.join(pcgmix_dir, category)) else 0
            pcgmix_plus_count = len(os.listdir(os.path.join(pcgmix_plus_dir, category))) if os.path.exists(os.path.join(pcgmix_plus_dir, category)) else 0
            
            print(f"\n{category.upper()}:")
            print(f"  Original files: {stats['files']}")
            print(f"  Target augmentations: {stats['target']}")
            print(f"  Attempts made: {stats['attempts']}")
            print(f"  PCGmix generated: {pcgmix_count}")
            print(f"  PCGmix+ generated: {pcgmix_plus_count}")
            print(f"  Total created: {pcgmix_count + pcgmix_plus_count}")
            print(f"  Target achieved: {((pcgmix_count + pcgmix_plus_count)/stats['target']*100):.1f}%")
            print(f"  Cardiac-aware: {stats['cardiac_aware']}")
            
            total_files += stats['files']
            total_targets += stats['target']
            total_attempts += stats['attempts']
            total_augmentations += pcgmix_count + pcgmix_plus_count
            total_cardiac_aware += stats['cardiac_aware']
    
    print(f"\n" + "-"*60)
    print("OVERALL TOTALS:")
    print(f"Total original files processed: {total_files}")
    print(f"Total target augmentations: {total_targets}")
    print(f"Total attempts made: {total_attempts}")
    print(f"Total augmentations created: {total_augmentations}")
    print(f"Total cardiac-aware augmentations: {total_cardiac_aware}")
    if total_attempts > 0:
        print(f"Overall success rate: {(total_augmentations/total_attempts*100):.1f}%")
    if total_targets > 0:
        print(f"Target achievement rate: {(total_augmentations/total_targets*100):.1f}%")
    
    # Show new dataset distribution
    print(f"\n" + "-"*60)
    print("NEW TRAINING DATASET DISTRIBUTION (Original + Augmented):")
    current_counts = {
        'normal': 245,
        'murmur': 89,
        'extra_systole': 33,
        'extra_heart_audio': 14,
        'artifact': 29
    }
    
    for category in ['normal', 'murmur', 'extra_systole', 'extra_heart_audio', 'artifact']:
        original = current_counts.get(category, 0)
        augmented = 0
        if category in total_stats:
            pcgmix_count = len(os.listdir(os.path.join(pcgmix_dir, category))) if os.path.exists(os.path.join(pcgmix_dir, category)) else 0
            pcgmix_plus_count = len(os.listdir(os.path.join(pcgmix_plus_dir, category))) if os.path.exists(os.path.join(pcgmix_plus_dir, category)) else 0
            augmented = pcgmix_count + pcgmix_plus_count
        
        new_total = original + augmented
        print(f"  {category}: {original} → {new_total} (+{augmented})")

def create_comparison_augmentations():
    """
    Create both cardiac-aware and simple augmentations for comparison
    """
    print("Creating cardiac-aware augmentations...")
    create_cardiac_aware_augmentations()
    
    print("\n" + "="*50)
    print("For comparison, you can also run simple augmentations without cardiac awareness")
    print("="*50)

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Recommended augmentation targets based on your dataset analysis
    # Current distribution: normal(244), murmur(88), extra_systole(32), extra_heart_audio(13), artifact(28)
    recommended_targets = {
        'normal': 756,           
        'murmur': 162,           
        'extra_systole': 218,    
        'extra_heart_audio': 237, 
        'artifact': 222          
    }
    
    create_cardiac_aware_augmentations(category_targets=recommended_targets)