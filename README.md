# AI Framework

A comprehensive framework for enhancing large language models with bias/fairness analysis, uncertainty quantification, and adversarial defense.

## Overview

This framework provides a robust solution for enhancing AI visibility, fairness, and accountability. It integrates several important components:

- **Bias and Fairness Analysis**: Detects and quantifies bias, stereotyping, and fairness issues in LLM outputs
- **Uncertainty Quantification**: Provides confidence metrics and uncertainty estimation for model responses
- **Adversarial Defense**: Protects against prompt injection, jailbreaking, and other adversarial attacks
- **LangChain Integration**: Enables modular prompt workflows and chain-of-thought reasoning
- **Comprehensive Evaluation**: Includes metrics and visualizations for model performance

## Installation

### Setting Up the Conda Environment

1. Clone this repository:
   ```bash
   git clone https://your-repository-url/ai_framework.git
   cd ai_framework
   ```

2. Create and activate the conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate ai_framework
   ```

3. Install additional dependencies:
   ```bash
   # Download spaCy English model
   python -m spacy download en_core_web_sm
   ```

### Hugging Face Authentication

Some datasets used for evaluation require authentication with the Hugging Face Hub. To set up authentication:

1. Create an account on [Hugging Face](https://huggingface.co/) if you don't have one
2. Go to https://huggingface.co/settings/tokens to generate a token
3. Run the authentication script:
   ```bash
   python huggingface_auth.py
   ```
4. Follow the prompts to enter your token

Alternatively, you can set the token as an environment variable:
```bash
export HUGGINGFACE_TOKEN=your_token_here
```

If you don't authenticate, the framework will fall back to using local examples for evaluation, but some functionality will be limited.

### Troubleshooting Installation

If you encounter issues with the `conformal` package, you can modify `environment.yml` to use `nonconformist` as an alternative:

```yaml
- pip:
    # ...other packages...
    - nonconformist>=2.1.0  # Alternative to conformal
    # ...other packages...
```

If LangChain integration fails, you can run the framework without it using the `--no-langchain` flag.

## Project Structure

```
ai_framework/
├── src/
│   ├── __init__.py
│   ├── adversarial_defense.py     # Defenses against adversarial attacks
│   ├── bias_fairness.py           # Bias and fairness analysis
│   ├── datasets.py                # Dataset loading utilities
│   ├── evaluation.py              # Evaluation metrics and reporting
│   ├── framework.py               # Core framework implementation
│   ├── langchain_integration.py   # LangChain adapters and utilities
│   ├── llm_wrapper.py             # LLM interface wrapper
│   ├── monitoring_logging.py      # Logging and monitoring utilities
│   └── uncertainty_quantification.py  # Uncertainty estimation methods
├── main.py                        # Main CLI interface
├── simple_demo.py                 # Simplified demo without LangChain
├── environment.yml                # Conda environment specification
└── requirements.txt               # Pip requirements
```

## Usage

### Basic Usage

Run the main script with the default configuration:

```bash
python main.py
```

For a simplified demo that doesn't depend on LangChain:

```bash
python simple_demo.py
```

### Command Line Options

The main script supports several options:

```bash
python main.py --model distilgpt2 --uncertainty ensemble --mode basic
```

Available options:
- `--model`: Model name to use (default: distilgpt2)
- `--uncertainty`: Uncertainty quantification method (choices: ensemble, mc_dropout, conformal, softmax_entropy, placeholder)
- `--no-langchain`: Disable LangChain integration
- `--mode`: Demonstration mode (choices: basic, evaluation, langchain, batch, all)
- `--output-dir`: Directory for evaluation outputs

### Example Workflows

#### 1. Basic Prompt Processing

```python
from src.framework import AIFramework

framework = AIFramework(model_name="distilgpt2")
result = framework.process_prompt("Explain the concept of neural networks.")
print(result["response"])
```

#### 2. Batch Processing

```python
from src.framework import AIFramework

framework = AIFramework()
prompts = [
    "What is machine learning?",
    "Explain the concept of climate change.",
    "What are black holes?"
]
results = framework.batch_process(prompts, max_workers=2)
```

#### 3. Running Evaluations

```python
from src.framework import AIFramework
from src.datasets import load_bias_data
from src.evaluation import evaluate_bias

framework = AIFramework()
bias_metrics = framework.evaluate(load_bias_data, evaluate_bias)
print(bias_metrics)
```

## Components

### LLM Wrapper

The `LLMWrapper` class provides a simple interface to Hugging Face transformer models.

### Bias and Fairness Analysis

The `BiasFairnessAnalyzer` detects:
- Offensive content
- Stereotype reinforcement
- Counterfactual fairness

### Uncertainty Quantification

The `UncertaintyQuantifier` implements multiple methods:
- Ensemble-based uncertainty
- Monte Carlo dropout
- Conformal prediction
- Softmax entropy

### Adversarial Defense

The `AdversarialDefense` class provides:
- Prompt filtering and sanitization
- Response anomaly detection
- Jailbreak pattern recognition

## Logging and Monitoring

Interactions and events are logged to `ai_interactions.log` in a structured JSON format. The logs include:
- Prompts and responses
- Analysis summaries
- Processing times
- Anomaly and bias alerts

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Reduce batch sizes or model size
   - Use a smaller model like "distilgpt2" instead of larger models

2. **LangChain Integration Errors**:
   - Try running with `--no-langchain` flag
   - Check that langchain and langchain-core packages are installed

3. **Slow Performance**:
   - Some analysis components like counterfactual fairness can be computationally intensive
   - Use batch processing with smaller batches for better performance

4. **Package Not Found Errors**:
   - Make sure you're using the conda environment: `conda activate ai_framework`
   - Check environment.yml for missing packages

5. **Authentication Errors with Hugging Face**:
   - Run `python huggingface_auth.py` to set up authentication
   - Check that your token has at least 'read' permissions
   - You can verify authentication with: `huggingface-cli whoami`

### Getting Help

For additional assistance, please file an issue on the repository or contact the project maintainers.

## License

[Insert your license information here]
