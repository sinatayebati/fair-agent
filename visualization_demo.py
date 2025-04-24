"""
Visualization demo for dataset comparison results.
"""

import os
import json
import argparse
import logging
from src.visualization import plot_dataset_comparison

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('VisualizationDemo')

def main():
    parser = argparse.ArgumentParser(description="Visualize dataset comparison results")
    parser.add_argument(
        "--input", 
        default="./evaluation_results/dataset_comparison.json",
        help="Path to dataset comparison JSON file"
    )
    parser.add_argument(
        "--output-dir", 
        default="./evaluation_results/visualizations",
        help="Directory to save visualization outputs"
    )
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        return 1
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate visualizations
    logger.info(f"Generating visualizations from {args.input}")
    plot_paths = plot_dataset_comparison(args.input, args.output_dir)
    
    if plot_paths:
        logger.info(f"Visualizations saved to {args.output_dir}")
        for plot_type, path in plot_paths.items():
            if path:
                logger.info(f"- {plot_type}: {path}")
    else:
        logger.warning("No visualizations were generated")
    
    return 0

if __name__ == "__main__":
    exit(main())
