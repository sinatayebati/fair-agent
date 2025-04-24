"""
Experiment runner for AI Framework evaluations.

This script provides utilities for running comprehensive experiments
across multiple datasets and evaluation metrics.
"""

import os
import json
import logging
import argparse
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# Load environment variables for API keys
load_dotenv()

from src.framework import AIFramework
from src.evaluation import (
    evaluate_bias, evaluate_security, evaluate_uncertainty, evaluate_overall_performance,
    compare_datasets_evaluation, create_evaluation_report
)
from src.datasets import load_bias_data, load_adversarial_data, load_uncertainty_data
from src.visualization import plot_dataset_comparison

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('Experiments')


def run_dataset_comparison_experiment(
    model_name: str = "gpt-4.1-nano",
    datasets: List[str] = None,
    uncertainty_method: str = "ensemble",
    output_dir: str = "./experiment_results",
    visualize: bool = True,
    use_openai: bool = True
):
    """
    Run a comprehensive dataset comparison experiment.
    
    Args:
        model_name: Name of the model to use
        datasets: List of datasets to compare (defaults to stereoset, crows_pairs, winobias)
        uncertainty_method: Method for uncertainty quantification
        output_dir: Output directory for results
        visualize: Whether to generate visualizations
        use_openai: Whether to use OpenAI API (True) or Hugging Face (False)
        
    Returns:
        Dictionary with experiment results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Default datasets if not specified
    if not datasets:
        datasets = ["stereoset", "crows_pairs", "winobias"]
        
    logger.info(f"Running dataset comparison experiment across {len(datasets)} datasets")
    logger.info(f"Model: {model_name}, Uncertainty method: {uncertainty_method}")
    logger.info(f"Using {'OpenAI API' if use_openai else 'Hugging Face'}")
    
    start_time = time.time()
    
    # Initialize the framework
    try:
        framework = AIFramework(
            model_name=model_name,
            uncertainty_method=uncertainty_method,
            use_langchain=False,
            use_openai=use_openai
        )
    except Exception as e:
        logger.error(f"Failed to initialize framework: {e}")
        return {"status": "error", "error": str(e)}
        
    # Run the dataset comparison
    comparison_path = os.path.join(output_dir, "dataset_comparison.json")
    try:
        comparison_report = compare_datasets_evaluation(
            framework, 
            datasets,
            output_path=comparison_path
        )
        
        # Generate visualizations if requested
        if visualize:
            visualizations_dir = os.path.join(output_dir, "visualizations")
            os.makedirs(visualizations_dir, exist_ok=True)
            
            try:
                plot_paths = plot_dataset_comparison(comparison_path, visualizations_dir)
                logger.info(f"Generated {len(plot_paths) if plot_paths else 0} visualization plots")
            except Exception as viz_error:
                logger.error(f"Error generating visualizations: {viz_error}")
        
    except Exception as e:
        logger.error(f"Error in dataset comparison: {e}")
        return {"status": "error", "error": str(e)}
        
    # Calculate experiment runtime
    runtime = time.time() - start_time
    
    # Save experiment metadata
    metadata = {
        "experiment_type": "dataset_comparison",
        "timestamp": datetime.now().isoformat(),
        "model": model_name,
        "uncertainty_method": uncertainty_method,
        "datasets": datasets,
        "runtime_seconds": runtime,
        "model_provider": "OpenAI API" if use_openai else "Hugging Face",
        "output_files": {
            "comparison_report": comparison_path,
            "visualizations_dir": os.path.join(output_dir, "visualizations") if visualize else None
        }
    }
    
    # Save metadata
    metadata_path = os.path.join(output_dir, "experiment_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
        
    logger.info(f"Experiment completed in {runtime:.2f} seconds")
    logger.info(f"Results saved to {output_dir}")
    
    return {
        "status": "success",
        "runtime_seconds": runtime,
        "comparison_path": comparison_path,
        "metadata_path": metadata_path
    }

def run_model_comparison_experiment(
    models: List[str] = None,
    dataset: str = "stereoset",
    uncertainty_method: str = "ensemble",
    output_dir: str = "./model_comparison_results",
    use_openai: bool = True
):
    """
    Compare multiple models on the same dataset.
    
    Args:
        models: List of model names to compare
        dataset: Dataset to use for comparison
        uncertainty_method: Uncertainty quantification method
        output_dir: Output directory for results
        use_openai: Whether to use OpenAI API (True) or Hugging Face (False)
        
    Returns:
        Dictionary with experiment results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Default models if not specified
    if not models:
        if use_openai:
            models = ["gpt-4.1-nano", "gpt-4.1-mini"]
        else:
            models = ["distilgpt2", "gpt2", "gpt2-medium"]
        
    logger.info(f"Running model comparison experiment with {len(models)} models")
    logger.info(f"Dataset: {dataset}, Uncertainty method: {uncertainty_method}")
    logger.info(f"Using {'OpenAI API' if use_openai else 'Hugging Face'}")
    
    start_time = time.time()
    
    results = {}
    for model_name in models:
        model_dir = os.path.join(output_dir, model_name.replace("/", "_"))
        os.makedirs(model_dir, exist_ok=True)
        
        logger.info(f"Testing model: {model_name}")
        
        try:
            # Initialize framework with this model
            framework = AIFramework(
                model_name=model_name,
                uncertainty_method=uncertainty_method,
                use_langchain=False,
                use_openai=use_openai
            )
            
            # Run bias evaluation
            bias_metrics = framework.evaluate(
                lambda: load_bias_data(dataset),
                evaluate_bias
            )
            
            # Run security evaluation
            security_metrics = framework.evaluate(
                lambda: load_adversarial_data(dataset),
                evaluate_security
            )
            
            # Run uncertainty evaluation
            uncertainty_data = load_uncertainty_data(dataset)
            uncertainty_labels = [item.get('label', 1) for item in uncertainty_data]
            uncertainty_metrics = framework.evaluate(
                lambda: uncertainty_data,
                evaluate_uncertainty,
                labels=uncertainty_labels
            )
            
            # Save model results
            model_results = {
                "bias_metrics": bias_metrics,
                "security_metrics": security_metrics,
                "uncertainty_metrics": uncertainty_metrics,
                "model": model_name,
                "model_provider": "OpenAI API" if use_openai else "Hugging Face"
            }
            
            model_results_path = os.path.join(model_dir, "evaluation_results.json")
            with open(model_results_path, "w") as f:
                json.dump(model_results, f, indent=2)
                
            results[model_name] = {
                "status": "success",
                "results_path": model_results_path
            }
            
        except Exception as e:
            logger.error(f"Error evaluating model {model_name}: {e}")
            results[model_name] = {
                "status": "error",
                "error": str(e)
            }
    
    # Calculate experiment runtime
    runtime = time.time() - start_time
    
    # Generate comparison visualizations
    try:
        create_model_comparison_visualizations(results, output_dir)
    except Exception as e:
        logger.error(f"Error generating model comparison visualizations: {e}")
    
    # Save experiment metadata
    metadata = {
        "experiment_type": "model_comparison",
        "timestamp": datetime.now().isoformat(),
        "models": models,
        "dataset": dataset,
        "uncertainty_method": uncertainty_method,
        "runtime_seconds": runtime,
        "model_provider": "OpenAI API" if use_openai else "Hugging Face",
        "models_evaluated": len([m for m, r in results.items() if r["status"] == "success"])
    }
    
    # Save metadata
    metadata_path = os.path.join(output_dir, "experiment_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
        
    logger.info(f"Model comparison experiment completed in {runtime:.2f} seconds")
    logger.info(f"Results saved to {output_dir}")
    
    return {
        "status": "success",
        "runtime_seconds": runtime,
        "results": results,
        "metadata_path": metadata_path
    }

def create_model_comparison_visualizations(results, output_dir):
    """
    Create visualizations comparing multiple models.
    """
    # Create visualizations directory
    viz_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    # Collect metrics from all models
    bias_metrics = {}
    security_metrics = {}
    uncertainty_metrics = {}
    
    for model_name, model_result in results.items():
        if model_result["status"] != "success":
            continue
            
        try:
            with open(model_result["results_path"], "r") as f:
                model_data = json.load(f)
                
            bias_metrics[model_name] = model_data.get("bias_metrics", {})
            security_metrics[model_name] = model_data.get("security_metrics", {})
            uncertainty_metrics[model_name] = model_data.get("uncertainty_metrics", {})
        except Exception as e:
            logger.error(f"Error reading results for model {model_name}: {e}")
    
    # Generate comparison charts
    if bias_metrics:
        create_comparison_chart(
            bias_metrics, 
            "avg_overall_fairness", 
            "Fairness Score Comparison", 
            os.path.join(viz_dir, "fairness_comparison.png"),
            higher_is_better=True
        )
        
        create_comparison_chart(
            bias_metrics, 
            "avg_stereotype_score", 
            "Stereotype Score Comparison", 
            os.path.join(viz_dir, "stereotype_comparison.png"),
            higher_is_better=False
        )
    
    if security_metrics:
        create_comparison_chart(
            security_metrics, 
            "defense_recall", 
            "Defense Recall Comparison", 
            os.path.join(viz_dir, "defense_recall_comparison.png"),
            higher_is_better=True
        )
    
    if uncertainty_metrics:
        create_comparison_chart(
            uncertainty_metrics, 
            "auroc", 
            "AUROC Comparison", 
            os.path.join(viz_dir, "auroc_comparison.png"),
            higher_is_better=True
        )

def create_comparison_chart(metrics_dict, metric_name, title, output_path, higher_is_better=True):
    """
    Create a bar chart comparing models on a specific metric.
    """
    plt.figure(figsize=(10, 6))
    
    models = []
    values = []
    
    for model, metrics in metrics_dict.items():
        if metric_name in metrics:
            models.append(model)
            values.append(metrics[metric_name])
    
    if not models:
        logger.warning(f"No data available for metric {metric_name}")
        return
    
    # Sort by metric value
    sorted_data = sorted(zip(models, values), key=lambda x: x[1], reverse=higher_is_better)
    models, values = zip(*sorted_data)
    
    # Create bar chart
    bars = plt.bar(models, values)
    
    # Color bars based on performance
    if higher_is_better:
        colors = ['#4CAF50', '#8BC34A', '#CDDC39', '#FFC107', '#FF9800', '#FF5722']
    else:
        colors = ['#FF5722', '#FF9800', '#FFC107', '#CDDC39', '#8BC34A', '#4CAF50']
    
    # Assign colors based on relative performance
    for i, bar in enumerate(bars):
        color_idx = min(int(i / len(bars) * len(colors)), len(colors) - 1)
        bar.set_color(colors[color_idx])
    
    plt.title(title)
    plt.xlabel('Models')
    plt.ylabel(metric_name)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add values on top of bars
    for i, v in enumerate(values):
        plt.text(i, v, f'{v:.3f}', ha='center', va='bottom')
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved comparison chart to {output_path}")

def main():
    """Main entry point for running experiments."""
    parser = argparse.ArgumentParser(description="AI Framework Experiment Runner")
    
    parser.add_argument(
        "--experiment",
        choices=["dataset_comparison", "model_comparison"],
        default="dataset_comparison",
        help="Type of experiment to run"
    )
    
    parser.add_argument(
        "--model",
        default="gpt-4.1-nano",
        help="Model to use (default: gpt-4.1-nano)"
    )
    
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="List of models to compare (for model_comparison experiment)"
    )
    
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["stereoset", "crows_pairs", "winobias"],
        help="List of datasets to evaluate (for dataset_comparison experiment)"
    )
    
    parser.add_argument(
        "--uncertainty",
        default="ensemble",
        choices=["ensemble", "mc_dropout", "conformal", "softmax_entropy", "placeholder"],
        help="Uncertainty quantification method"
    )
    
    parser.add_argument(
        "--output-dir",
        default="./experiment_results",
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--no-visualize",
        action="store_true",
        help="Disable visualization generation"
    )
    
    parser.add_argument(
        "--use-huggingface",
        action="store_true",
        help="Use Hugging Face models instead of OpenAI API"
    )
    
    args = parser.parse_args()
    
    # Determine if we should use OpenAI or Hugging Face
    use_openai = not args.use_huggingface
    
    # If using Hugging Face and no model specified, use a default HF model
    if args.use_huggingface and args.model == "gpt-4.1-nano":
        args.model = "distilgpt2"
    
    # Run the selected experiment
    if args.experiment == "dataset_comparison":
        result = run_dataset_comparison_experiment(
            model_name=args.model,
            datasets=args.datasets,
            uncertainty_method=args.uncertainty,
            output_dir=args.output_dir,
            visualize=not args.no_visualize,
            use_openai=use_openai
        )
    elif args.experiment == "model_comparison":
        result = run_model_comparison_experiment(
            models=args.models,
            dataset=args.datasets[0] if args.datasets else "stereoset",
            uncertainty_method=args.uncertainty,
            output_dir=args.output_dir,
            use_openai=use_openai
        )
    else:
        logger.error(f"Unknown experiment type: {args.experiment}")
        return 1
    
    # Print summary of results
    if result["status"] == "success":
        logger.info("Experiment completed successfully!")
        logger.info(f"Results saved to {args.output_dir}")
        return 0
    else:
        logger.error(f"Experiment failed: {result.get('error', 'Unknown error')}")
        return 1

if __name__ == "__main__":
    exit(main())
