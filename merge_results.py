#!/usr/bin/env python3

import pandas as pd
import os
import glob

def merge_qwen3_results():
    """Merge all Qwen3 CSV results into one file"""
    
    # Define paths
    data_dir = "Outputs/carg/gemini"
    output_dir = "data/Cleaned_results"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all CSV files in the qwen3 directory
    csv_pattern = os.path.join(data_dir, "experiment_diverse_gemini-2.5-flash_*.csv")
    csv_files = glob.glob(csv_pattern)
    
    if not csv_files:
        print(f"No CSV files found in {data_dir}")
        return
    
    print(f"Found {len(csv_files)} CSV files to merge:")
    
    # Read and merge all CSV files
    dfs = []
    for file_path in sorted(csv_files):  # Sort to ensure consistent order
        try:
            df = pd.read_csv(file_path)
            dfs.append(df)
            print(f"  - Loaded {file_path}: {len(df)} rows")
        except Exception as e:
            print(f"  - Error loading {file_path}: {e}")
    
    if not dfs:
        print("No valid CSV files found!")
        return
    
    # Merge all dataframes
    merged_df = pd.concat(dfs, ignore_index=True)
    
    # Save merged file
    output_file = os.path.join(output_dir, "gemini-2.5.csv")
    merged_df.to_csv(output_file, index=False)
    
    print(f"\nMerge completed!")
    print(f"Total rows merged: {len(merged_df)}")
    print(f"Merged file saved to: {output_file}")
    
    # Show a sample of the merged data
    print(f"\nFirst few rows of merged data:")
    print(merged_df.head())
    
    return output_file

if __name__ == "__main__":
    merge_qwen3_results()