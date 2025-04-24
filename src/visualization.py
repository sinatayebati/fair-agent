"""
Visualization utilities for comparing dataset evaluation results.
"""

import json
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Dict, List, Any, Optional

logger = logging.getLogger('AI_Framework')

def plot_dataset_comparison(comparison_data_path: str, output_dir: str = "./evaluation_results"):
    """
    Create visualizations for dataset comparison metrics.
    
    Args:
        comparison_data_path: Path to the JSON comparison data
        output_dir: Directory to save visualization outputs
        
    Returns:
        Dictionary of generated plot paths
    """
    logger.info(f"Generating dataset comparison visualizations from {comparison_data_path}")
    
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Load comparison data
        with open(comparison_data_path, 'r') as f:
            comparison_data = json.load(f)
        
        plots = {}
        
        # Get list of datasets
        datasets = list(comparison_data.get("metrics", {}).keys())
        if not datasets:
            logger.warning("No datasets found in comparison data")
            return {}
        
        # 1. Plot bias metrics comparison
        bias_plot_path = os.path.join(output_dir, "bias_comparison.png")
        plots["bias"] = plot_bias_metrics(comparison_data, bias_plot_path, datasets)
        
        # 2. Plot security metrics comparison
        security_plot_path = os.path.join(output_dir, "security_comparison.png")
        plots["security"] = plot_security_metrics(comparison_data, security_plot_path, datasets)
        
        # 3. Plot uncertainty metrics comparison
        uncertainty_plot_path = os.path.join(output_dir, "uncertainty_comparison.png")
        plots["uncertainty"] = plot_uncertainty_metrics(comparison_data, uncertainty_plot_path, datasets)
        
        # 4. Create a summary plot with key metrics from each category
        summary_plot_path = os.path.join(output_dir, "dataset_summary.png")
        plots["summary"] = plot_summary_metrics(comparison_data, summary_plot_path, datasets)
        
        logger.info(f"Generated {len(plots)} dataset comparison visualizations")
        return plots
        
    except Exception as e:
        logger.error(f"Error generating dataset comparison visualizations: {e}")
        return {}

def plot_bias_metrics(data: Dict, output_path: str, datasets: List[str]) -> str:
    """
    Plot bias metrics comparison across datasets.
    
    Args:
        data: Comparison data dictionary
        output_path: Path to save the plot
        datasets: List of dataset names
        
    Returns:
        Path to saved plot or None if failed
    """
    try:
        # Extract bias metrics for each dataset
        metrics = {}
        for dataset in datasets:
            if dataset in data.get("metrics", {}) and "bias_metrics" in data["metrics"][dataset]:
                bias_data = data["metrics"][dataset]["bias_metrics"]
                # Extract relevant metrics
                metrics[dataset] = {
                    "Offensiveness": bias_data.get("avg_offensiveness_score", 0),
                    "Stereotype": bias_data.get("avg_stereotype_score", 0),
                    "Bias Score": bias_data.get("avg_bias_score", 0),
                    "Fairness": bias_data.get("avg_overall_fairness", 0)
                }
        
        if not metrics:
            logger.warning("No bias metrics found for plotting")
            return None
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        # Set up bar positions
        metric_names = list(next(iter(metrics.values())).keys())
        x = np.arange(len(metric_names))
        width = 0.8 / len(metrics)
        
        # Plot bars for each dataset
        for i, (dataset, dataset_metrics) in enumerate(metrics.items()):
            values = list(dataset_metrics.values())
            offset = width * i - width * len(metrics) / 2 + width / 2
            plt.bar(x + offset, values, width, label=dataset)
        
        plt.xlabel('Metrics')
        plt.ylabel('Score')
        plt.title('Bias & Fairness Metrics Comparison')
        plt.xticks(x, metric_names)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Save plot
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
        logger.info(f"Bias comparison plot saved to {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error generating bias comparison plot: {e}")
        return None

def plot_security_metrics(data: Dict, output_path: str, datasets: List[str]) -> str:
    """
    Plot security metrics comparison across datasets.
    
    Args:
        data: Comparison data dictionary
        output_path: Path to save the plot
        datasets: List of dataset names
        
    Returns:
        Path to saved plot or None if failed
    """
    try:
        # Extract security metrics for each dataset
        metrics = {}
        for dataset in datasets:
            if dataset in data.get("metrics", {}) and "security_metrics" in data["metrics"][dataset]:
                security_data = data["metrics"][dataset]["security_metrics"]
                # Extract relevant metrics
                metrics[dataset] = {
                    "Defense Precision": security_data.get("defense_precision", 0),
                    "Defense Recall": security_data.get("defense_recall", 0),
                    "False Positive Rate": security_data.get("false_positive_rate", 0),
                    "Anomaly Detection": security_data.get("anomaly_detection_rate", 0)
                }
        
        if not metrics:
            logger.warning("No security metrics found for plotting")
            return None
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        # Set up bar positions
        metric_names = list(next(iter(metrics.values())).keys())
        x = np.arange(len(metric_names))
        width = 0.8 / len(metrics)
        
        # Plot bars for each dataset
        for i, (dataset, dataset_metrics) in enumerate(metrics.items()):
            values = list(dataset_metrics.values())
            offset = width * i - width * len(metrics) / 2 + width / 2
            plt.bar(x + offset, values, width, label=dataset)
        
        plt.xlabel('Metrics')
        plt.ylabel('Score')
        plt.title('Security & Robustness Metrics Comparison')
        plt.xticks(x, metric_names)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Save plot
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
        logger.info(f"Security comparison plot saved to {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error generating security comparison plot: {e}")
        return None

def plot_uncertainty_metrics(data: Dict, output_path: str, datasets: List[str]) -> str:
    """
    Plot uncertainty metrics comparison across datasets.
    
    Args:
        data: Comparison data dictionary
        output_path: Path to save the plot
        datasets: List of dataset names
        
    Returns:
        Path to saved plot or None if failed
    """
    try:
        # Extract uncertainty metrics for each dataset
        metrics = {}
        for dataset in datasets:
            if dataset in data.get("metrics", {}) and "uncertainty_metrics" in data["metrics"][dataset]:
                uncertainty_data = data["metrics"][dataset]["uncertainty_metrics"]
                # Extract relevant metrics
                metrics[dataset] = {
                    "ECE": uncertainty_data.get("ece", 0),
                    "AUROC": uncertainty_data.get("auroc", 0),
                    "AUPRC": uncertainty_data.get("auprc", 0),
                    "Brier Score": uncertainty_data.get("brier_score", 0)
                }
        
        if not metrics:
            logger.warning("No uncertainty metrics found for plotting")
            return None
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        # Set up bar positions
        metric_names = list(next(iter(metrics.values())).keys())
        x = np.arange(len(metric_names))
        width = 0.8 / len(metrics)
        
        # Plot bars for each dataset
        for i, (dataset, dataset_metrics) in enumerate(metrics.items()):
            values = list(dataset_metrics.values())
            offset = width * i - width * len(metrics) / 2 + width / 2
            plt.bar(x + offset, values, width, label=dataset)
        
        plt.xlabel('Metrics')
        plt.ylabel('Score')
        plt.title('Uncertainty Quantification Metrics Comparison')
        plt.xticks(x, metric_names)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Save plot
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
        logger.info(f"Uncertainty comparison plot saved to {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error generating uncertainty comparison plot: {e}")
        return None

def plot_summary_metrics(data: Dict, output_path: str, datasets: List[str]) -> str:
    """
    Plot a summary of key metrics across all categories.
    
    Args:
        data: Comparison data dictionary
        output_path: Path to save the plot
        datasets: List of dataset names
        
    Returns:
        Path to saved plot or None if failed
    """
    try:
        # Extract key metrics from each category
        summary_metrics = {}
        for dataset in datasets:
            if dataset in data.get("metrics", {}):
                dataset_data = data["metrics"][dataset]
                
                # Initialize metrics dictionary
                summary_metrics[dataset] = {}
                
                # Add bias key metrics
                if "bias_metrics" in dataset_data:
                    bias_data = dataset_data["bias_metrics"]
                    summary_metrics[dataset]["Fairness Score"] = bias_data.get("avg_overall_fairness", 0)
                    summary_metrics[dataset]["Bias Score"] = bias_data.get("avg_bias_score", 0)
                    
                # Add security key metrics
                if "security_metrics" in dataset_data:
                    security_data = dataset_data["security_metrics"]
                    summary_metrics[dataset]["Defense Recall"] = security_data.get("defense_recall", 0)
                    
                # Add uncertainty key metrics
                if "uncertainty_metrics" in dataset_data:
                    uncertainty_data = dataset_data["uncertainty_metrics"]
                    summary_metrics[dataset]["AUROC"] = uncertainty_data.get("auroc", 0)
        
        if not summary_metrics:
            logger.warning("No summary metrics found for plotting")
            return None
        
        # Create plot - radar chart for summary
        metrics = list(next(iter(summary_metrics.values())).keys())
        
        # Set up the radar chart
        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Close the loop
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        
        # Plot each dataset
        for dataset, values in summary_metrics.items():
            values_list = list(values.values())
            values_list += values_list[:1]  # Close the loop
            ax.plot(angles, values_list, linewidth=2, label=dataset)
            ax.fill(angles, values_list, alpha=0.1)
        
        # Set the labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        
        # Add legend and title
        plt.legend(loc='upper right')
        plt.title('Dataset Performance Summary', size=15)
        
        # Save plot
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
        logger.info(f"Summary comparison plot saved to {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error generating summary comparison plot: {e}")
        return None

# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Example usage with dummy data
    dummy_path = "dataset_comparison.json"
    plots = plot_dataset_comparison(dummy_path)
    print(f"Generated plots: {plots}")
