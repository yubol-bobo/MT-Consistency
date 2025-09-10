<<<<<<< HEAD

import sys
import os

def main():
    src_dir = os.path.join(os.path.dirname(__file__), 'src')
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    from eval_visualize import run_all_evaluations
    run_all_evaluations()

if __name__ == "__main__":
    main()
=======
#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from pathlib import Path
from typing import Dict

# --- Style Utilities ---
def get_color_palette():
    return {
        'gpt': '#FF9999',        # Red family for GPT variants
        'gpt_42': '#FF9999',
        'gpt_friendly': '#FF9999',
        'gpt_adv': '#FF9999',
        'gpt_sol': '#FF0000',
        'proposed_method': '#FF0000',
        'claude': '#99FF99',     # Green for Claude
        'gemini': '#9999FF',     # Blue for Gemini
        'gemini-2.5': '#9999FF', # Blue for Gemini
        'mistral': '#FFFF99',    # Yellow for Mistral
        'llama': '#FF99FF',      # Purple for LLaMA
        'llama-4': '#FF99FF',    # Purple for LLaMA
        'qwen': '#99FFFF',       # Cyan for Qwen
        'deepseek r1': '#FFCC99' # Orange for DeepSeek
    }

def get_line_styles():
    return {
        'gpt_sol': {'linewidth': 6, 'markersize': 8, 'linestyle': '-'},
        'default': {'linewidth': 4, 'markersize': 8, 'linestyle': '--'}
    }

def set_plot_style():
    plt.rcParams.update({
        'figure.figsize': (12, 8),
        'figure.dpi': 300,
        'font.size': 14,
        'font.weight': 'bold',
        'font.family': 'DejaVu Sans',
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '-.',
        'axes.axisbelow': True,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 14,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1
    })

# Set style globally at import
set_plot_style()


def rename_model(model_name: str) -> str:
    """
    Rename model names to shorter, more readable versions.
    """
    name_mapping = {
        'gemini-2.5-flash': 'gemini-2.5',
        'llama-4-maverick-17b-128e-instruct': 'llama-4',
        'deepseek-chat': 'deepseek r1'
    }
    return name_mapping.get(model_name, model_name)


def load_experiment_data(outputs_dir: Path, experiment_type: str = "diverse") -> Dict[str, pd.DataFrame]:
    """
    Load experiment data from the outputs directory.
    Args:
        outputs_dir: Path to the base outputs directory
        experiment_type: Type of experiment ("diverse" or "repetitive")
    Returns:
        Dictionary mapping model names to their combined DataFrame
    """
    data = {}
    exp_dir = outputs_dir / experiment_type
    if not exp_dir.exists():
        print(f"Warning: {exp_dir} does not exist")
        return {}

    csv_files = glob.glob(str(exp_dir / "**/*.csv"), recursive=True)
    for csv_file in csv_files:
        filename = Path(csv_file).name
        if filename.startswith("experiment_") and filename.endswith(".csv"):
            parts = filename.replace(".csv", "").split("_")
            if len(parts) >= 4:
                model_name = parts[2]
                # Rename model name
                model_name = rename_model(model_name)
                try:
                    df = pd.read_csv(csv_file)
                    # Ensure round columns are numeric and handle tuple format
                    round_cols = [col for col in df.columns if col.startswith('round_')]
                    for col in round_cols:
                        # Handle tuple format like "(1, None)" by extracting first element
                        df[col] = df[col].astype(str).apply(lambda x: 
                            x.strip('()').split(',')[0] if x and x != 'nan' and ',' in x else x)
                        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                    data.setdefault(model_name, []).append(df)
                except Exception as e:
                    print(f"Error loading {csv_file}: {e}")
    combined = {m: pd.concat(dfs, ignore_index=True) for m, dfs in data.items() if dfs}
    return combined


def compute_pwc(sequence, gamma: float = 0.9) -> float:
    """Compute position-weighted insistence score."""
    # Convert sequence values to numeric types
    numeric_seq = [float(s) if s is not None else 0.0 for s in sequence]
    weights = [gamma ** i for i in range(len(numeric_seq))]
    return float(sum(s * w for s, w in zip(numeric_seq, weights)))


def find_first_sway(sequence) -> int:
    """Find the round of first change from correct (1) to incorrect (0)."""
    if not sequence:
        return -1
    # Convert sequence values to numeric types
    numeric_seq = [float(s) if s is not None else 0.0 for s in sequence]
    if numeric_seq[0] == 0:
        return 1
    for i in range(len(numeric_seq) - 1):
        if numeric_seq[i] == 1 and numeric_seq[i+1] == 0:
            return i + 2
    return -1


def count_sway_recovery(sequence) -> int:
    """Count number of 1->0->1 patterns in the sequence."""
    count = 0
    in_zero = False
    # Convert sequence values to numeric types
    numeric_seq = [float(s) if s is not None else 0.0 for s in sequence]
    if numeric_seq and numeric_seq[0] == 0:
        in_zero = True
    for prev, curr in zip(numeric_seq, numeric_seq[1:]):
        if prev == 1 and curr == 0:
            in_zero = True
        elif prev == 0 and curr == 1 and in_zero:
            count += 1
            in_zero = False
    return count


def calculate_metrics(df: pd.DataFrame, n_rounds: int = 8, gamma: float = 0.9):
    """
    Calculate metrics: initial accuracy, PWC, first sway, sway recovery.
    Returns a dict of aggregated metrics and per-question DataFrame.
    """
    total = len(df)
    initial_correct = (df['round_0'] == 1).sum()
    initial_accuracy = initial_correct / total * 100 if total > 0 else 0

    seq_df = df[df['round_0'] == 1].copy()
    round_cols = [f'round_{i}' for i in range(1, n_rounds+1) if f'round_{i}' in df.columns]

    pwc = []
    first_sw = []
    sway_rec = []
    for _, row in seq_df.iterrows():
        seq = row[round_cols].tolist()
        pwc.append(compute_pwc(seq, gamma))
        first_sw.append(find_first_sway(seq))
        sway_rec.append(count_sway_recovery(seq))

    seq_df['pwc_score'] = pwc
    seq_df['first_sway'] = first_sw
    seq_df['sway_recovery'] = sway_rec

    metrics = {
        'initial_accuracy': initial_accuracy,
        'average_pwc': np.mean(pwc) if pwc else 0,
        'std_pwc': np.std(pwc) if pwc else 0,
        'average_first_sway': np.mean([s for s in first_sw if s > 0]) if any(s > 0 for s in first_sw) else -1,
        'average_SR_pair': np.mean(sway_rec) if sway_rec else 0
    }
    return metrics, seq_df


def process_multiple_models(data: Dict[str, pd.DataFrame], n_rounds: int = 8, gamma: float = 0.9):
    """
    Compute and save metrics and distributions for each model.
    """
    all_metrics = []
    fs_dist = {}
    sr_dist = {}
    for model, df in data.items():
        metrics, seq_df = calculate_metrics(df, n_rounds, gamma)
        metrics['model'] = model
        all_metrics.append(metrics)
        fs_counts = seq_df['first_sway'].value_counts().to_dict()
        sr_counts = seq_df['sway_recovery'].value_counts().to_dict()
        fs_dist[model] = fs_counts
        sr_dist[model] = sr_counts

    results_df = pd.DataFrame(all_metrics)
    fs_df = pd.DataFrame.from_dict(fs_dist, orient='index').fillna(0).astype(int)
    sr_df = pd.DataFrame.from_dict(sr_dist, orient='index').fillna(0).astype(int)

    results_df.to_csv('metrics_model_comparison.csv', index=False)
    fs_df.to_csv('first_sway_distribution.csv')
    sr_df.to_csv('sway_recovery_distribution.csv')

    return results_df, fs_df, sr_df



def plot_followup_round_accuracies(data: Dict[str, pd.DataFrame], n_rounds: int = 8):
    """
    Plot follow-up round accuracies (rounds 1-8) with enhanced styling and data export
    """
    # Create a DataFrame to store accuracies
    accuracy_data = []

    # Calculate accuracies for each model
    model_accuracies = {}
    for model_name, df in data.items():
        accuracies = []
        for i in range(1, n_rounds+1):
            round_col = f'round_{i}'
            if round_col not in df.columns:
                continue
            total = len(df[round_col])
            correct = (df[round_col] == 1).sum()
            accuracy = (correct / total * 100) if total > 0 else 0
            accuracies.append(round(accuracy, 2))

            # Add to our data collection for DataFrame
            accuracy_data.append({
                'model': model_name,
                'round': i,
                'accuracy': round(accuracy, 2),
                'total_samples': total,
                'correct_samples': correct
            })

        model_accuracies[model_name] = accuracies

    # Create and save DataFrame
    accuracy_df = pd.DataFrame(accuracy_data)
    os.makedirs('cleaned_results/analysis', exist_ok=True)
    accuracy_df.to_csv('cleaned_results/analysis/model_accuracies.csv', index=False)

    plt.figure(figsize=(12, 8))
    color_palette = get_color_palette()
    line_styles = get_line_styles()

    # Plot lines for each model
    for model_name, accuracies in model_accuracies.items():
        if not accuracies:  # Skip if no data
            continue
        
        # Always use renamed model name for color and label
        display_name = rename_model(model_name)
        style = line_styles['gpt_sol'] if display_name == 'gpt_sol' else line_styles['default']

        plt.plot(range(1, len(accuracies)+1), accuracies,
                marker='o',
                label=display_name,
                color=color_palette.get(display_name, '#CCCCCC'),
                **style)

        # Add annotations for all points with offset
        for i, accuracy in enumerate(accuracies):
            # Calculate offset direction based on trend
            if i > 0:
                offset_y = 1 if accuracies[i] > accuracies[i-1] else -2.5
            else:
                offset_y = -2.5

            plt.annotate(f'{accuracy:.1f}',
                        xy=(i + 1, accuracy),
                        xytext=(0, offset_y),
                        textcoords='offset points',
                        ha='center',
                        va='bottom' if offset_y > 0 else 'top',
                        fontsize=10,
                        bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=0.5))

    # Customize the plot
    plt.xlabel('Follow-up Message', fontweight='bold')
    plt.ylabel('Accuracy (%)', fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.3)

    # Set x-axis ticks
    plt.xticks(range(1, n_rounds+1))

    # Add legend inside the plot
    plt.legend(loc='best')

    # Adjust layout
    plt.tight_layout()

    # Save plot
    os.makedirs('figs', exist_ok=True)
    plt.savefig('figs/followup_round_accuracies_comparison.png',
                bbox_inches='tight', dpi=300)
    plt.close()

    # Print summary statistics
    print("\nSummary Statistics for Each Model:")
    for model_name, accuracies in model_accuracies.items():
        if accuracies:
            print(f"\n{model_name}:")
            print(f"Initial accuracy: {accuracies[0]:.2f}%")
            print(f"Final accuracy: {accuracies[-1]:.2f}%")
            print(f"Max accuracy: {max(accuracies):.2f}%")
            print(f"Min accuracy: {min(accuracies):.2f}%")
            print(f"Average accuracy: {sum(accuracies)/len(accuracies):.2f}%")

    return accuracy_df


def plot_model_metrics_comparison(results_df: pd.DataFrame):
    """
    Create enhanced visualizations comparing different metrics across models with additional insights,
    saving each metric plot as a separate file
    """
    metrics_to_plot = [
        'initial_accuracy', 'average_pwc', 'average_first_sway', 'average_SR_pair'
    ]

    color_palette = get_color_palette()

    # Create plots directory if it doesn't exist
    os.makedirs('figs', exist_ok=True)

    for metric in metrics_to_plot:
        plt.figure(figsize=(12, 8))

        # Sort models by metric value
        sorted_df = results_df.sort_values(by=metric, ascending=True)

        # Create bar plot with custom colors
        bars = plt.barh(range(len(sorted_df)), sorted_df[metric],
                       color=[color_palette.get(model, '#CCCCCC') for model in sorted_df['model']])

        # Add model names
        plt.yticks(range(len(sorted_df)), sorted_df['model'])

        # Add only value labels
        for i, v in enumerate(sorted_df[metric]):
            plt.text(v, i, f' {v:.2f}', va='center')

        # Add mean line
        mean_value = sorted_df[metric].mean()
        plt.axvline(x=mean_value, color='red', linestyle='--', alpha=0.5, linewidth=5)
        plt.text(mean_value, len(sorted_df), f'Mean: {mean_value:.2f}',
                ha='center', va='bottom', color='red')

        # Customize plot
        plt.grid(True, alpha=0.3)

        # Add axis labels
        plt.ylabel('Model', fontweight='bold', fontsize=16)

        # Custom x-label based on metric
        xlabel_mapping = {
            'initial_accuracy': 'Initial Round Accuracy (%)',
            'average_pwc': 'PWC Score',
            'average_first_sway': 'First Response Sway Round',
            'average_SR_pair': 'Sway-Recovery Pattern Count'
        }
        plt.xlabel(xlabel_mapping[metric], fontweight='bold', fontsize=16)

        plt.tight_layout()

        # Save individual plot
        plt.savefig(f'figs/model_comparison_{metric}.png',
                    bbox_inches='tight', dpi=300)
        plt.close()


if __name__ == '__main__':
    outputs_dir = Path('outputs')
    data = load_experiment_data(outputs_dir, experiment_type='diverse')
    if data:
        results_df, fs_df, sr_df = process_multiple_models(data)
        plot_followup_round_accuracies(data)
        plot_model_metrics_comparison(results_df)
    else:
        print('No data loaded; please check your outputs directory.')
>>>>>>> d75285df7d7c0d1b3d00a730561ff1c8df04dc7e
