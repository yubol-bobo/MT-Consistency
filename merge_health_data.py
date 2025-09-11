#!/usr/bin/env python3
import pandas as pd
import glob
import os
import re
from pathlib import Path

def clean_tuple_format(value):
    """Clean the tuple format to remove np.float64 references"""
    if pd.isna(value) or value == '':
        return ''
    
    value_str = str(value)
    # Remove np.float64() wrapper
    value_str = re.sub(r'np\.float64\(([\d.]+)\)', r'\1', value_str)
    return value_str

def merge_health_experiment_files(input_dir: str, output_file: str):
    """
    Merge experiment CSV files from health experiments into a single file
    similar to the format in data/Cleaned_results/
    """
    # Get all CSV files matching the pattern
    csv_files = glob.glob(os.path.join(input_dir, "experiment_diverse_gpt-4o_*.csv"))
    csv_files.sort()  # Ensure consistent ordering
    
    print(f"Found {len(csv_files)} CSV files to merge:")
    for file in csv_files:
        print(f"  - {os.path.basename(file)}")
    
    # List to store all dataframes
    all_dfs = []
    
    for csv_file in csv_files:
        # Extract batch number from filename
        batch_num = int(os.path.basename(csv_file).split('_')[-1].replace('.csv', ''))
        
        # Read the CSV
        df = pd.read_csv(csv_file)
        
        # Add batch number column
        df['batch_number'] = batch_num - 1  # Make it 0-indexed to match existing format
        
        # Clean up the tuple format in round columns
        round_cols = [col for col in df.columns if col.startswith('round_') and col != 'round_0']
        for col in round_cols:
            df[col] = df[col].apply(clean_tuple_format)
        
        all_dfs.append(df)
        print(f"Processed batch {batch_num}: {len(df)} rows")
    
    # Concatenate all dataframes
    merged_df = pd.concat(all_dfs, ignore_index=True)
    
    # Ensure columns are in the right order
    col_order = ['round_0', 'round_1', 'round_2', 'round_3', 'round_4', 
                 'round_5', 'round_6', 'round_7', 'round_8', 'batch_number']
    merged_df = merged_df[col_order]
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save merged data
    merged_df.to_csv(output_file, index=False)
    
    print(f"\nSuccessfully merged {len(csv_files)} files.")
    print(f"Total rows in merged dataset: {len(merged_df)}")
    print(f"Saved to: {output_file}")
    
    return merged_df

def create_health_evaluation_data():
    """Create merged data files for health experiments"""
    
    # Create directory for health cleaned results
    health_cleaned_dir = "data/health_cleaned_results"
    os.makedirs(health_cleaned_dir, exist_ok=True)
    
    # Merge GPT health experiment data
    input_dir = "Outputs/health/diverse/gpt"
    output_file = os.path.join(health_cleaned_dir, "gpt.csv")
    
    if os.path.exists(input_dir):
        merged_df = merge_health_experiment_files(input_dir, output_file)
        return True
    else:
        print(f"Input directory not found: {input_dir}")
        return False

if __name__ == "__main__":
    success = create_health_evaluation_data()
    if success:
        print("Health experiment data merged successfully!")
    else:
        print("Failed to merge health experiment data.")
