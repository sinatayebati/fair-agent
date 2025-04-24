# AI Framework Experiment Guide

This guide explains how to run experiments with the AI Framework to evaluate fairness, uncertainty, and robustness across different datasets and models.

## Quick Start

For a simple dataset comparison experiment with default settings:

```bash
python experiments.py
```

This will compare the performance of the default model (distilgpt2) across three datasets: stereoset, crows_pairs, and winobias.

## Available Experiment Types

The framework currently supports two main experiment types:

1. **Dataset Comparison**: Compare model performance across different datasets
2. **Model Comparison**: Compare different models on the same dataset

## Dataset Comparison Experiment

This experiment evaluates how a model performs across different datasets, providing insights into domain-specific biases and limitations.

### Basic Usage

```bash
python experiments.py --experiment dataset_comparison \
                     --model distilgpt2 \
                     --datasets stereoset crows_pairs winobias \
                     --output-dir ./dataset_comparison_results
```

### Output

The experiment generates:
- A comprehensive JSON report comparing metrics across datasets
- Visualizations showing key metrics for each dataset
- A summary plot highlighting performance differences

## Model Comparison Experiment

This experiment compares multiple models on the same dataset to identify which models perform better for specific metrics.

### Basic Usage

```bash
python experiments.py --experiment model_comparison \
                     --models distilgpt2 gpt2 gpt2-medium \
                     --datasets stereoset \
                     --output-dir ./model_comparison_results
```

### Output

The experiment generates:
- Individual evaluation reports for each model
- Comparative visualizations for key metrics
- A summary report ranking models on different dimensions

## Advanced Options

### Uncertainty Methods

Choose from different uncertainty quantification methods:
- `ensemble`: Uses multiple samples with temperature variation (default)
- `mc_dropout`: Uses Monte Carlo dropout for stochastic estimation
- `conformal`: Uses conformal prediction for coverage guarantees
- `softmax_entropy`: Uses entropy of token probabilities
- `placeholder`: Simple heuristic-based method

Example:
```bash
python experiments.py --uncertainty mc_dropout
```

### Visualization Control

To disable visualization generation:
```bash
python experiments.py --no-visualize
```

## Using the Results

The experiment results are saved in the specified output directory, which includes:

1. **JSON Reports**: Detailed metrics for all evaluations
2. **Visualizations**: Plots comparing metrics across datasets/models
3. **Metadata**: Experiment configuration and runtime information

### Example Analysis

To analyze the results programmatically:

```python
import json
import matplotlib.pyplot as plt

# Load experiment results
with open("./experiment_results/dataset_comparison.json") as f:
    results = json.load(f)
    
# Access metrics for a specific dataset
stereoset_metrics = results["metrics"]["stereoset"]
fairness_score = stereoset_metrics["bias_metrics"]["avg_overall_fairness"]

print(f"Fairness score on StereoSet: {fairness_score}")
```

## Comparing Custom Datasets

To compare custom datasets, you'll need to:

1. Implement a data loader in `src/datasets.py`
2. Run the experiment specifying your custom dataset:

```bash
python experiments.py --datasets my_custom_dataset standard_dataset
```

## Best Practices

1. **Start Small**: Begin with smaller models like distilgpt2 for faster iteration
2. **Validate Results**: Run experiments multiple times to ensure consistency
3. **Compare Incrementally**: Change one variable at a time (model OR dataset)
4. **Check Logs**: Monitor ai_interactions.log for warnings or anomalies

## Troubleshooting

### Common Issues

- **Resource Errors**: Reduce batch sizes or use smaller models
- **Dataset Loading Errors**: Ensure you have the necessary HuggingFace credentials
- **Visualization Errors**: Install matplotlib and ensure the output directory is writable

### Error Messages

- "Error loading dataset X": Make sure you have the required authentication and the dataset exists
- "CUDA out of memory": Use a smaller model or reduce batch sizes
- "No data available for metric X": Check if the evaluation successfully generated the expected metrics

## Advanced Experiment Integration

For even more advanced experimentation, you can integrate with the framework in Python:

```python
from experiments import run_dataset_comparison_experiment

# Custom experiment configuration
results = run_dataset_comparison_experiment(
    model_name="gpt2-medium",
    datasets=["stereoset", "crows_pairs", "my_custom_dataset"],
    uncertainty_method="ensemble",
    output_dir="./custom_experiment_results",
    visualize=True
)

# Access and process results
if results["status"] == "success":
    print(f"Experiment completed in {results['runtime_seconds']:.2f} seconds")
    print(f"Results saved to {results['comparison_path']}")
```
