#!/usr/bin/env python3

import sys
import os
import pandas as pd
import shutil

def parse_health_data(value):
    """Parse health data format with np.float64"""
    if pd.isna(value) or str(value).strip() == '':
        return None, None
    
    s = str(value).strip()
    
    # If it's just a number (like round_0)
    try:
        return int(float(s)), None
    except:
        pass
    
    # If it's in tuple format with np.float64
    if s.startswith('(') and s.endswith(')'):
        try:
            # Remove np.float64( and extract values
            clean_s = s.replace('np.float64(', '').replace(')', '')
            # Split by comma and extract correctness and confidence
            parts = clean_s.strip('()').split(',')
            correctness = int(float(parts[0].strip()))
            confidence = float(parts[1].strip()) if len(parts) > 1 else None
            return correctness, confidence
        except Exception as e:
            print(f"Error parsing {s}: {e}")
            return None, None
    
    return None, None

def transform_health_dataframe(df):
    """Transform health data format to standard format"""
    df = df.copy()
    
    # Transform round columns 1-8
    for i in range(1, 9):
        col = f'round_{i}'
        if col in df.columns:
            parsed_data = df[col].apply(parse_health_data)
            df[f'{col}_ans'] = parsed_data.apply(lambda x: x[0] if x[0] is not None else float('nan'))
            df[f'{col}_conf'] = parsed_data.apply(lambda x: x[1] if x[1] is not None else float('nan'))
    
    return df

def merge_csv_files_in_directory(input_dir, output_dir):
    """Merge all CSV files in a directory into one file and transform data"""
    if not os.path.exists(input_dir):
        print(f"Directory {input_dir} does not exist!")
        return None
    
    # Find all CSV files
    csv_files = []
    for file in os.listdir(input_dir):
        if file.endswith('.csv'):
            csv_files.append(os.path.join(input_dir, file))
    
    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        return None
    
    print(f"Found {len(csv_files)} CSV files in {input_dir}")
    
    # Read and merge all CSV files
    dfs = []
    for file_path in csv_files:
        try:
            df = pd.read_csv(file_path)
            # Transform the dataframe to parse health data format
            df_transformed = transform_health_dataframe(df)
            dfs.append(df_transformed)
            print(f"  - Loaded and transformed {file_path}: {len(df)} rows")
        except Exception as e:
            print(f"  - Error loading {file_path}: {e}")
    
    if dfs:
        # Merge all dataframes
        merged_df = pd.concat(dfs, ignore_index=True)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save merged file
        # Extract model name from directory structure
        if 'carg' in input_dir.lower():
            model_name = 'carg'
        elif 'gpt' in input_dir.lower():
            model_name = 'gpt'
        else:
            model_name = 'unknown'
        
        output_file = os.path.join(output_dir, f"{model_name}.csv")
        merged_df.to_csv(output_file, index=False)
        print(f"  Merged and transformed file saved: {output_file} ({len(merged_df)} rows)")
        
        return output_file
    
    return None

def prepare_health_data():
    """Merge CSV files from each directory and prepare for evaluation"""
    print("="*60)
    print("PREPARING HEALTH DATA FOR EVALUATION")
    print("="*60)
    
    # Define directories
    carg_input_dir = "Outputs/health/diverse/carg"
    gpt_input_dir = "Outputs/health/diverse/gpt"
    merged_output_dir = "data/health_cleaned_results"
    
    # Clean up existing merged results
    if os.path.exists(merged_output_dir):
        shutil.rmtree(merged_output_dir)
    
    os.makedirs(merged_output_dir, exist_ok=True)
    
    # Merge CARG files
    print("\n1. Merging CARG results...")
    carg_merged = merge_csv_files_in_directory(carg_input_dir, merged_output_dir)
    
    # Merge GPT files  
    print("\n2. Merging GPT results...")
    gpt_merged = merge_csv_files_in_directory(gpt_input_dir, merged_output_dir)
    
    if carg_merged or gpt_merged:
        print(f"\nData preparation complete!")
        print(f"Merged results saved in: {merged_output_dir}")
        return True
    else:
        print("No data was merged!")
        return False

def main():
    # Add src directory to path
    src_dir = os.path.join(os.path.dirname(__file__), 'src')
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    
    # Step 1: Prepare the data by merging CSV files
    if not prepare_health_data():
        print("Data preparation failed!")
        return
    
    # Step 2: Import and run evaluation using eval_visualize
    print("\n" + "="*60)
    print("RUNNING HEALTH DATA EVALUATION")
    print("="*60)
    
    from eval_visualize import (
        evaluate_all_models, plot_accuracy_trends, calculate_model_metrics,
        plot_model_metrics_comparison, plot_model_round_accuracies
    )
    
    # Set paths for health data evaluation
    results_dir = "data/health_cleaned_results"
    plot_dir = "Outputs/health/plots"
    csv_dir = os.path.join(plot_dir, "csv")
    
    # Create output directories
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)
    
    print(f"Loading data from: {results_dir}")
    print(f"Saving plots to: {plot_dir}")
    
    try:
        # 1. Accuracy table and round trends
        print("\n1. Computing accuracy table and trends...")
        accuracy_table, all_data = evaluate_all_models(results_dir)
        accuracy_table.to_csv(os.path.join(csv_dir, "accuracy_table.csv"))
        plot_accuracy_trends(accuracy_table, save_path=os.path.join(plot_dir, "accuracy_trends.png"))
        
        # 2. Model metrics comparison
        print("2. Computing model metrics...")
        metrics = {'model': [], 'initial_accuracy': [], 'average_pwc': [], 'average_first_sway': [], 'average_SR_pair': []}
        for model in accuracy_table.index:
            df = all_data[model]
            metrics['model'].append(model)
            metrics['initial_accuracy'].append(accuracy_table.loc[model, 'round_0'])
            avg_pwc, avg_first_sway, avg_sway_recovery = calculate_model_metrics(df)
            metrics['average_pwc'].append(avg_pwc)
            metrics['average_first_sway'].append(avg_first_sway)
            metrics['average_SR_pair'].append(avg_sway_recovery)
        
        results_df = pd.DataFrame(metrics)
        results_df.to_csv(os.path.join(csv_dir, "model_metrics.csv"), index=False)
        plot_model_metrics_comparison(results_df, plot_dir)
        
        # 3. Round-by-round accuracy for all models
        print("3. Plotting round-by-round accuracies...")
        plot_model_round_accuracies(all_data, plot_dir)
        
        print(f"\nHealth experiment evaluation complete!")
        print(f"Results saved to: {plot_dir}")
        print(f"CSV files saved to: {csv_dir}")
        
        # Print summary
        print(f"\nSummary:")
        print(f"- Models evaluated: {list(all_data.keys())}")
        print(f"- Total data points: {sum(len(df) for df in all_data.values())}")
        for model, df in all_data.items():
            print(f"  - {model}: {len(df)} samples")
            
        # Print key metrics
        print(f"\nKey Metrics:")
        for _, row in results_df.iterrows():
            print(f"- {row['model']}:")
            print(f"  Initial Accuracy: {row['initial_accuracy']:.1f}%")
            print(f"  PWC Score: {row['average_pwc']:.1f}%")
            print(f"  First Sway: {row['average_first_sway']:.1f}%")
            print(f"  Sway Recovery: {row['average_SR_pair']:.1f}%")
            
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()