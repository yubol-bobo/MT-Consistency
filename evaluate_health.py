#!/usr/bin/env python3

import sys
import os

def main():
    # Add src directory to path
    src_dir = os.path.join(os.path.dirname(__file__), 'src')
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    
    # Import evaluation functions
    from eval_visualize import (
        evaluate_all_models, plot_accuracy_trends, calculate_model_metrics,
        plot_model_metrics_comparison, plot_model_round_accuracies
    )
    import pandas as pd
    
    # Set paths for health data evaluation
    results_dir = "data/health_cleaned_results"
    plot_dir = "Outputs/health/plots"
    csv_dir = os.path.join(plot_dir, "csv")
    
    # Create output directories
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)
    
    print("Running health experiment evaluation...")
    print(f"Loading data from: {results_dir}")
    print(f"Saving plots to: {plot_dir}")
    
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

if __name__ == '__main__':
    main()
