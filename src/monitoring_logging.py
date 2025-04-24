import logging
import json
import time
from datetime import datetime
import numpy as np

# Custom JSON encoder to handle numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

# Configure logging
log_file = 'ai_interactions.log'
logger = logging.getLogger('AI_Framework')
logger.setLevel(logging.INFO)
# Prevent duplicate handlers if this module is reloaded
if not logger.handlers:
    # File handler for structured logging
    file_handler = logging.FileHandler(log_file)
    file_formatter = logging.Formatter('{"timestamp": "%(asctime)s", "level": "%(levelname)s", "module": "%(module)s", "message": %(message)s}')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console handler for basic visibility
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)


def log_interaction(prompt: str, response: str, metadata: dict = None):
    """Logs a single AI interaction with optional metadata."""
    log_entry = {
        "prompt": prompt,
        "response": response,
        "interaction_timestamp": datetime.utcnow().isoformat(),
        "metadata": metadata or {}
    }
    # Use json.dumps with the custom encoder to ensure the message part is valid JSON
    logger.info(json.dumps(log_entry, cls=NumpyEncoder))

def log_event(event_type: str, details: dict):
    """Logs a specific event (e.g., anomaly detected, bias alert)."""
    log_entry = {
        "event_type": event_type,
        "details": details,
        "event_timestamp": datetime.utcnow().isoformat()
    }
    logger.warning(json.dumps(log_entry, cls=NumpyEncoder)) # Use WARNING level for events like anomalies/alerts

# Example usage (optional)
if __name__ == '__main__':
    log_interaction("Hello AI", "Hello User!", {"user_id": "test_user", "session_id": "abc"})
    log_event("ANOMALY_DETECTED", {"type": "Adversarial Prompt", "score": 0.95})
    log_event("BIAS_ALERT", {"type": "Offensive Content", "score": 1.0})
    print(f"Check logs in {log_file}")
