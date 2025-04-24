#!/bin/bash

# Simple shell script for running AI Framework experiments

# Default values
EXPERIMENT_TYPE="dataset_comparison"
MODEL="gpt-4.1-nano"
UNCERTAINTY="ensemble"
OUTPUT_DIR="./experiment_results_$(date +%Y%m%d_%H%M%S)"
DATASETS="stereoset crows_pairs winobias"
VISUALIZE="true"
USE_OPENAI="true"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    -e|--experiment)
      EXPERIMENT_TYPE="$2"
      shift
      shift
      ;;
    -m|--model)
      MODEL="$2"
      shift
      shift
      ;;
    --models)
      shift
      MODELS=()
      while [[ $# -gt 0 ]] && [[ "$1" != -* ]]; do
        MODELS+=("$1")
        shift
      done
      ;;
    -d|--datasets)
      shift
      DATASETS=""
      while [[ $# -gt 0 ]] && [[ "$1" != -* ]]; do
        DATASETS="$DATASETS $1"
        shift
      done
      ;;
    -u|--uncertainty)
      UNCERTAINTY="$2"
      shift
      shift
      ;;
    -o|--output-dir)
      OUTPUT_DIR="$2"
      shift
      shift
      ;;
    --no-visualize)
      VISUALIZE="false"
      shift
      ;;
    --use-huggingface)
      USE_OPENAI="false"
      shift
      ;;
    -h|--help)
      echo "Usage: ./run_experiment.sh [options]"
      echo "Options:"
      echo "  -e, --experiment TYPE      Experiment type: dataset_comparison or model_comparison"
      echo "  -m, --model MODEL          Model to use (default: gpt-4.1-nano)"
      echo "  --models MODEL1 MODEL2...  Models to compare (for model_comparison)"
      echo "  -d, --datasets D1 D2...    Datasets to evaluate"
      echo "  -u, --uncertainty METHOD   Uncertainty method (default: ensemble)"
      echo "  -o, --output-dir DIR       Output directory"
      echo "  --no-visualize             Disable visualization generation"
      echo "  --use-huggingface          Use Hugging Face models instead of OpenAI API"
      echo "  -h, --help                 Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $key"
      exit 1
      ;;
  esac
done

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Check for OpenAI API key if using OpenAI
if [ "$USE_OPENAI" = "true" ] && [ -z "$OPENAI_API_KEY" ]; then
  if [ -f ".env" ]; then
    source .env
  fi
  
  if [ -z "$OPENAI_API_KEY" ]; then
    echo "Error: OPENAI_API_KEY environment variable is not set."
    echo "Please set this variable in your environment or in a .env file."
    exit 1
  fi
fi

# Build command
CMD="python experiments.py --experiment $EXPERIMENT_TYPE --model $MODEL --uncertainty $UNCERTAINTY --output-dir $OUTPUT_DIR"

# Add model list if specified
if [ -n "${MODELS[*]}" ]; then
  CMD="$CMD --models ${MODELS[*]}"
fi

# Add datasets
if [ -n "$DATASETS" ]; then
  CMD="$CMD --datasets $DATASETS"
fi

# Add visualization flag
if [ "$VISUALIZE" = "false" ]; then
  CMD="$CMD --no-visualize"
fi

# Add huggingface flag
if [ "$USE_OPENAI" = "false" ]; then
  CMD="$CMD --use-huggingface"
fi

# Print command
echo "Running experiment with command:"
echo "$CMD"
echo "Results will be saved to: $OUTPUT_DIR"
echo "Using: $([ "$USE_OPENAI" = "true" ] && echo "OpenAI API" || echo "Hugging Face")"

# Run the experiment
eval "$CMD"

# Check result
if [ $? -eq 0 ]; then
  echo "Experiment completed successfully!"
  echo "Results saved to: $OUTPUT_DIR"
else
  echo "Experiment failed with error code $?."
  echo "Check the logs for details."
fi
