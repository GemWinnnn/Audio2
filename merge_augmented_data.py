import os
import shutil
from pathlib import Path

def merge_augmented_data():
    """
    Merge augmented data with train data for each class.
    Creates train and test folders in merged_data directory.
    """
    
    # Base paths
    base_path = Path("/Users/gemwincanete/Audio2/datasets")
    merged_data_path = base_path / "merged_data"
    denoised_path = base_path / "Denoised"
    augmented_path = base_path / "Cardiac_Aware_Augmented_data"
    
    # Classes to process
    classes = ["normal", "murmur", "extra_systole", "extra_heart_audio", "artifact"]
    
    # Augmented data types
    augmented_types = ["PCGmix_cardiac_aware", "PCGmix_plus_cardiac_aware"]
    
    print("Starting data merging process...")
    
    for class_name in classes:
        print(f"\nProcessing class: {class_name}")
        
        # Create train and test directories for this class
        class_train_path = merged_data_path / class_name / "train"
        class_test_path = merged_data_path / class_name / "test"
        
        class_train_path.mkdir(parents=True, exist_ok=True)
        class_test_path.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Copy train data from Denoised directory
        denoised_train_path = denoised_path / class_name / "Train"
        if denoised_train_path.exists():
            print(f"  Copying train data from {denoised_train_path}")
            for file_path in denoised_train_path.glob("*.wav"):
                if file_path.is_file():
                    dest_path = class_train_path / file_path.name
                    shutil.copy2(file_path, dest_path)
            print(f"  Copied {len(list(denoised_train_path.glob('*.wav')))} train files")
        
        # Step 2: Copy test data from Denoised directory
        denoised_test_path = denoised_path / class_name / "Test"
        if denoised_test_path.exists():
            print(f"  Copying test data from {denoised_test_path}")
            for file_path in denoised_test_path.glob("*.wav"):
                if file_path.is_file():
                    dest_path = class_test_path / file_path.name
                    shutil.copy2(file_path, dest_path)
            print(f"  Copied {len(list(denoised_test_path.glob('*.wav')))} test files")
        
        # Step 3: Copy augmented data to train folder
        for aug_type in augmented_types:
            aug_class_path = augmented_path / aug_type / class_name
            if aug_class_path.exists():
                print(f"  Copying augmented data from {aug_class_path}")
                for file_path in aug_class_path.glob("*.wav"):
                    if file_path.is_file():
                        dest_path = class_train_path / file_path.name
                        shutil.copy2(file_path, dest_path)
                print(f"  Copied {len(list(aug_class_path.glob('*.wav')))} augmented files from {aug_type}")
        
        # Count final files
        train_files = len(list(class_train_path.glob("*.wav")))
        test_files = len(list(class_test_path.glob("*.wav")))
        
        print(f"  Final counts for {class_name}:")
        print(f"    Train: {train_files} files")
        print(f"    Test: {test_files} files")
    
    print("\nData merging completed successfully!")
    
    # Print summary
    print("\n" + "="*50)
    print("FINAL SUMMARY")
    print("="*50)
    
    for class_name in classes:
        class_train_path = merged_data_path / class_name / "train"
        class_test_path = merged_data_path / class_name / "test"
        
        train_count = len(list(class_train_path.glob("*.wav"))) if class_train_path.exists() else 0
        test_count = len(list(class_test_path.glob("*.wav"))) if class_test_path.exists() else 0
        
        print(f"{class_name:20} | Train: {train_count:4d} | Test: {test_count:4d} | Total: {train_count + test_count:4d}")

if __name__ == "__main__":
    merge_augmented_data() 