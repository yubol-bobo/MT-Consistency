# PWC Evaluation Framework

This directory contains the `evaluate.py` script that implements the Position-Weighted Consistency (PWC) evaluation framework from the "Firm or Fickle?" paper.

## Overview

The evaluation script calculates comprehensive metrics for LLM consistency analysis:

- **Initial Accuracy**: Base performance on zero-shot responses
- **Follow-up Round Accuracies**: Performance across multiple interaction rounds
- **PWC Score**: Position-Weighted Consistency score that captures early-stage stability and recovery patterns
- **Mean First Sway**: Average round when models first change from correct to incorrect
- **Sway-Recovery Analysis**: Count of sway-recovery patterns (correct → incorrect → correct)

## Usage

### Basic Usage

```bash
python evaluate.py
```

This will:
1. Automatically load data from the `Outputs/` directory
2. Calculate all metrics for all available models
3. Generate comprehensive visualizations
4. Create a detailed summary report

### Advanced Usage

```python
from evaluate import PWCEvaluator

# Initialize evaluator
evaluator = PWCEvaluator(outputs_dir="path/to/outputs")

# Evaluate specific experiment type
results = evaluator.evaluate_all_models("diverse")  # or "repetitive"

# Generate individual plots
evaluator.plot_initial_accuracy_comparison()
evaluator.plot_follow_up_accuracy_trends()
evaluator.plot_pwc_score_comparison()
evaluator.plot_first_sway_analysis()
evaluator.plot_comprehensive_comparison()

# Generate summary report
evaluator.generate_summary_report()
```

## Output Files

The script generates the following files:

### Visualizations (`figs/` directory)
- `initial_accuracy_comparison.png`: Bar chart comparing initial accuracy across models
- `follow_up_accuracy_trends.png`: Line plot showing accuracy trends across follow-up rounds
- `pwc_score_comparison.png`: Bar chart with error bars comparing PWC scores
- `first_sway_analysis.png`: Analysis of when models first sway from correct answers
- `comprehensive_comparison.png`: 2x2 grid showing all key metrics

### Reports
- `evaluation_summary.txt`: Comprehensive text report with all metrics and rankings

## Data Format

The script expects experiment data in CSV format with columns:
- `round_0`: Initial response (1 = correct, 0 = incorrect)
- `round_1` through `round_8`: Follow-up responses

The script handles both simple binary values and tuple formats like `"(1, None)"`.

## Key Metrics Explained

### Initial Accuracy
- Percentage of correct responses in the first round
- Measures base knowledge without follow-up interactions

### PWC Score
- Position-weighted consistency score (0-1 scale)
- Higher weights for early rounds using exponential decay (γ=0.9)
- Captures both stability and recovery patterns

### Mean First Sway
- Average round number when models first change from correct to incorrect
- Only calculated for sequences that start with correct answers
- Lower values indicate earlier swaying

### Sway-Recovery Analysis
- Count of patterns where models go from correct → incorrect → correct
- Measures recovery ability after making mistakes

## Example Results

Based on the current data, the script found:

- **Gemini-2.5-Flash**: Highest PWC score (0.957), most consistent
- **LLaMA-4-Maverick**: Middle PWC score (0.874), moderate consistency  
- **DeepSeek-Chat**: Lowest PWC score (0.828), least consistent

All models showed 100% initial accuracy, but differed significantly in their consistency across follow-up rounds.

## Requirements

- Python 3.7+
- pandas
- numpy
- matplotlib
- seaborn

## Troubleshooting

1. **No data found**: Ensure experiment results are in `Outputs/diverse/` or `Outputs/repetitive/` directories
2. **Import errors**: Install required packages with `pip install pandas numpy matplotlib seaborn`
3. **Data format issues**: The script automatically handles different data formats, but ensure CSV files follow the expected structure

## Citation

If you use this evaluation framework, please cite:

```bibtex
@article{li2025firm,
  title={Firm or Fickle? Evaluating Large Language Models Consistency in Sequential Interactions},
  author={Li, Yubo and Miao, Yidi and Ding, Xueying and Krishnan, Ramayya and Padman, Rema},
  journal={arXiv preprint arXiv:2503.22353},
  year={2025}
}
``` 