"""
Utility functions for the AI Framework.
"""
import os
import json
import logging
import datetime
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger('AI_Framework')

def save_json(data: Any, filepath: str, indent: int = 2) -> bool:
    """
    Save data to a JSON file.
    
    Args:
        data: Data to save (must be JSON serializable)
        filepath: Output filepath
        indent: JSON indentation level
        
    Returns:
        True if saving was successful
    """
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=indent)
        return True
    except Exception as e:
        logger.error(f"Error saving JSON to {filepath}: {e}")
        return False

def load_json(filepath: str) -> Any:
    """
    Load data from a JSON file.
    
    Args:
        filepath: Input filepath
        
    Returns:
        Loaded data or None if error
    """
    try:
        if not os.path.exists(filepath):
            logger.warning(f"File not found: {filepath}")
            return None
            
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading JSON from {filepath}: {e}")
        return None

def parse_datetime(dt_str: str) -> Optional[datetime.datetime]:
    """
    Parse a datetime string into a datetime object.
    
    Args:
        dt_str: Datetime string
        
    Returns:
        Datetime object or None if parsing fails
    """
    formats = [
        "%Y-%m-%dT%H:%M:%S.%f",  # ISO format with microseconds
        "%Y-%m-%dT%H:%M:%S",      # ISO format without microseconds
        "%Y-%m-%d %H:%M:%S",      # Standard datetime format
        "%Y-%m-%d"                # Date only
    ]
    
    for fmt in formats:
        try:
            return datetime.datetime.strptime(dt_str, fmt)
        except ValueError:
            continue
    
    logger.warning(f"Could not parse datetime string: {dt_str}")
    return None

def plot_metrics_over_time(metrics: List[Dict[str, Any]], 
                          metric_key: str,
                          timestamps: List[str],
                          title: str = None,
                          output_path: str = None) -> bool:
    """
    Plot metrics over time.
    
    Args:
        metrics: List of dictionaries containing metrics
        metric_key: Key for the metric to plot
        timestamps: List of timestamp strings
        title: Plot title
        output_path: If provided, save plot to this path
        
    Returns:
        True if plotting was successful
    """
    try:
        # Extract metric values
        values = [m.get(metric_key, 0) for m in metrics]
        
        # Parse timestamps
        dates = [parse_datetime(ts) for ts in timestamps]
        valid_data = [(d, v) for d, v in zip(dates, values) if d is not None]
        
        if not valid_data:
            logger.warning("No valid data points for plotting")
            return False
            
        dates, values = zip(*valid_data)
        
        # Create plot
        plt.figure(figsize=(10, 6))
        plt.plot(dates, values, marker='o')
        
        # Add labels and title
        plt.xlabel('Timestamp')
        plt.ylabel(metric_key)
        
        if title:
            plt.title(title)
        else:
            plt.title(f'{metric_key} over time')
            
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save or show plot
        if output_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            plt.savefig(output_path)
            logger.info(f"Plot saved to {output_path}")
        else:
            plt.show()
            
        plt.close()
        return True
        
    except Exception as e:
        logger.error(f"Error plotting metrics: {e}")
        return False

def extract_log_metrics(log_filepath: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Extract metrics from log file.
    
    Args:
        log_filepath: Path to log file
        
    Returns:
        Tuple of (metrics list, timestamps list)
    """
    metrics = []
    timestamps = []
    
    try:
        if not os.path.exists(log_filepath):
            logger.warning(f"Log file not found: {log_filepath}")
            return [], []
            
        with open(log_filepath, 'r') as f:
            for line in f:
                try:
                    # Parse JSON from log line (assumes properly formatted JSON logs)
                    parts = line.strip().split('{', 1)
                    if len(parts) < 2:
                        continue
                        
                    # Extract timestamp from the log prefix
                    timestamp = parts[0].split(' - ')[0].strip()
                    timestamps.append(timestamp)
                    
                    # Parse the JSON part
                    json_str = '{' + parts[1]
                    log_entry = json.loads(json_str)
                    
                    # Extract metrics
                    if 'message' in log_entry and isinstance(log_entry['message'], dict):
                        if 'analysis_summary' in log_entry['message']:
                            metrics.append(log_entry['message']['analysis_summary'])
                    
                except Exception as e:
                    # Skip malformed lines
                    continue
                    
        return metrics, timestamps
        
    except Exception as e:
        logger.error(f"Error extracting metrics from log: {e}")
        return [], []

def calculate_token_usage(prompt: str, response: str) -> Dict[str, int]:
    """
    Estimate token usage for a prompt and response.
    
    Args:
        prompt: Input prompt
        response: Model response
        
    Returns:
        Dictionary with token counts
    """
    # Very rough estimation (4 chars ≈ 1 token)
    prompt_tokens = len(prompt) // 4
    response_tokens = len(response) // 4
    
    return {
        "prompt_tokens": prompt_tokens,
        "response_tokens": response_tokens,
        "total_tokens": prompt_tokens + response_tokens
    }

def format_duration(seconds: float) -> str:
    """
    Format a duration in seconds to a human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    if seconds < 0.001:
        return f"{seconds * 1000000:.2f} μs"
    elif seconds < 1:
        return f"{seconds * 1000:.2f} ms"
    elif seconds < 60:
        return f"{seconds:.2f} s"
    else:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.2f}s"

# Example usage
if __name__ == "__main__":
    # Set up logging for standalone testing
    logging.basicConfig(level=logging.INFO)
    
    # Test save and load JSON
    test_data = {"test": "data", "nested": {"value": 123}}
    save_json(test_data, "test_output.json")
    loaded_data = load_json("test_output.json")
    print("Loaded data:", loaded_data)
    
    # Test token usage calculation
    prompt = "Tell me about artificial intelligence."
    response = "Artificial intelligence (AI) is the simulation of human intelligence by machines."
    tokens = calculate_token_usage(prompt, response)
    print("Token usage:", tokens)
    
    # Test duration formatting
    print("0.0005 seconds =", format_duration(0.0005))
    print("0.5 seconds =", format_duration(0.5))
    print("5 seconds =", format_duration(5))
    print("125 seconds =", format_duration(125))

