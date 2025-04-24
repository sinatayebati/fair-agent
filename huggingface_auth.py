"""
Utility for authenticating with Hugging Face Hub.
This helps address the 401 Unauthorized errors when accessing datasets.
"""

import os
import logging
from getpass import getpass

logger = logging.getLogger('AI_Framework')

def setup_huggingface_authentication():
    """
    Setup authentication for Hugging Face Hub.
    
    This function checks if a token is already set and guides the user
    through the process of obtaining and setting up a token if needed.
    """
    try:
        from huggingface_hub import login
        
        # Check if token is already set in environment
        token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
        
        if token:
            logger.info("Hugging Face token found in environment variables")
            login(token=token)
            return True
        
        # Check if token is in .cache directory
        cache_file = os.path.expanduser("~/.cache/huggingface/token")
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                cached_token = f.read().strip()
                if cached_token:
                    logger.info("Hugging Face token found in cache")
                    login(token=cached_token)
                    return True
        
        # If no token is found, prompt the user
        print("\n=== Hugging Face Hub Authentication ===")
        print("Some datasets require authentication to access.")
        print("You can get a token at: https://huggingface.co/settings/tokens")
        print("(Select 'read' access when creating your token)")
        
        # Ask if the user wants to authenticate now
        choice = input("\nDo you want to authenticate now? (y/n): ").lower()
        if choice in ['y', 'yes']:
            token = getpass("\nEnter your Hugging Face token: ")
            if token:
                login(token=token)
                
                # Ask if the user wants to save the token as an environment variable
                save_choice = input("Save this token for future sessions? (y/n): ").lower()
                if save_choice in ['y', 'yes']:
                    with open(os.path.expanduser("~/.huggingface"), "w") as f:
                        f.write(f"export HUGGINGFACE_TOKEN={token}\n")
                    print("Token saved to ~/.huggingface")
                    print("You can add this to your shell profile by running:")
                    print("echo 'source ~/.huggingface' >> ~/.bashrc  # or ~/.zshrc")
                
                return True
        
        print("\nContinuing without authentication.")
        print("Some datasets might not be available.")
        return False
        
    except ImportError:
        logger.warning("huggingface_hub package not installed. Authentication not possible.")
        return False
    except Exception as e:
        logger.error(f"Error setting up Hugging Face authentication: {e}")
        return False

def is_authenticated():
    """
    Check if already authenticated with Hugging Face Hub.
    
    Returns:
        True if authenticated, False otherwise
    """
    try:
        from huggingface_hub import whoami
        try:
            user_info = whoami()
            return True
        except Exception:
            return False
    except ImportError:
        return False

if __name__ == "__main__":
    # Configure basic logging when run directly
    logging.basicConfig(level=logging.INFO)
    
    if is_authenticated():
        print("Already authenticated with Hugging Face Hub.")
    else:
        setup_huggingface_authentication()
