import pandas as pd
import glob
import os
from pathlib import Path

def merge_csv_files_from_multiple_paths(paths, output_file='merged_data.csv'):
    """
    Merge all CSV files from multiple directory paths into a single CSV file.
    
    Args:
        paths (list): List of directory paths containing CSV files
        output_file (str): Name of the output merged CSV file
    
    Returns:
        pd.DataFrame: The merged dataframe
    """
    
    all_dataframes = []
    file_info = []
    
    # Process each directory path
    for path in paths:
        print(f"Processing directory: {path}")
        
        # Check if directory exists
        if not os.path.exists(path):
            print(f"Warning: Directory {path} does not exist. Skipping...")
            continue
        
        # Find all CSV files in the current directory
        csv_pattern = os.path.join(path, "*.csv")
        csv_files = glob.glob(csv_pattern)
        
        if not csv_files:
            print(f"No CSV files found in {path}")
            continue
        
        print(f"Found {len(csv_files)} CSV files in {path}")
        
        # Read each CSV file and add source information
        for csv_file in csv_files:
            try:
                # Read the CSV file
                df = pd.read_csv(csv_file)
                
                # Add source information columns
                df['source_directory'] = path
                df['source_filename'] = os.path.basename(csv_file)
                df['source_full_path'] = csv_file
                
                # Extract category from path (e.g., 'artifact', 'normal', etc.)
                path_parts = Path(path).parts
                if 'artifact' in path_parts:
                    df['category'] = 'artifact'
                elif 'extra_heart_audio' in path_parts:
                    df['category'] = 'extra_heart_audio'
                elif 'extra_systole' in path_parts:
                    df['category'] = 'extra_systole'
                elif 'murmur' in path_parts:
                    df['category'] = 'murmur'  
                elif 'normal' in path_parts:
                    df['category'] = 'normal'
                else:
                    df['category'] = 'unknown'
                
                all_dataframes.append(df)
                file_info.append({
                    'file': csv_file,
                    'rows': len(df),
                    'columns': len(df.columns)
                })
                
                print(f"  ✓ Loaded {csv_file}: {len(df)} rows, {len(df.columns)} columns")
                
            except Exception as e:
                print(f"  ✗ Error reading {csv_file}: {str(e)}")
    
    if not all_dataframes:
        print("No CSV files were successfully loaded!")
        return None
    
    # Merge all dataframes
    print(f"\nMerging {len(all_dataframes)} dataframes...")
    merged_df = pd.concat(all_dataframes, ignore_index=True)
    
    # Save to output file
    merged_df.to_csv(output_file, index=False)
    
    # Print summary
    print(f"\n=== MERGE SUMMARY ===")
    print(f"Total files processed: {len(file_info)}")
    print(f"Total rows in merged file: {len(merged_df)}")
    print(f"Total columns in merged file: {len(merged_df.columns)}")
    print(f"Output saved to: {output_file}")
    
    # Show category distribution
    if 'category' in merged_df.columns:
        print(f"\nCategory distribution:")
        category_counts = merged_df['category'].value_counts()
        for category, count in category_counts.items():
            print(f"  {category}: {count} rows")
    
    # Show file details
    print(f"\nFile details:")
    for info in file_info:
        print(f"  {os.path.basename(info['file'])}: {info['rows']} rows, {info['columns']} columns")
    
    return merged_df

# Define your directory paths
paths = [
    "/Users/gemwincanete/Audio/thesis/datasets/Denoised/artifact/Train/springer_segmentation_output",
    "/Users/gemwincanete/Audio/thesis/datasets/Denoised/extra_heart_audio/Train/springer_segmentation_output",
    "/Users/gemwincanete/Audio/thesis/datasets/Denoised/extra_systole/Train/springer_segmentation_output",
    "/Users/gemwincanete/Audio/thesis/datasets/Denoised/murmur/Train/springer_segmentation_output",
    "/Users/gemwincanete/Audio/thesis/datasets/Denoised/normal/Train/springer_segmentation_output"
]

# Run the merge
if __name__ == "__main__":
    # You can customize the output filename
    output_filename = "merged_springer_segmentation_data.csv"
    
    merged_data = merge_csv_files_from_multiple_paths(paths, output_filename)
    
    if merged_data is not None:
        # Optional: Display first few rows
        print(f"\nFirst 5 rows of merged data:")
        print(merged_data.head())
        
        # Optional: Show data types
        print(f"\nData types:")
        print(merged_data.dtypes)