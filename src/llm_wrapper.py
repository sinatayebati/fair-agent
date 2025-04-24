import torch
import os
import time
from typing import Dict, Any, Optional, Union, List
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

logger = logging.getLogger('LLM_Wrapper')

class BaseLLMWrapper:
    """Base class for LLM wrappers."""
    def generate(self, prompt: str, max_length: int = 50, **kwargs) -> str:
        """Generate text based on the prompt."""
        raise NotImplementedError("Subclasses must implement this method")
    
    def get_probabilities(self, prompt: str):
        """Get token probabilities."""
        raise NotImplementedError("Subclasses must implement this method")

class HuggingFaceLLMWrapper(BaseLLMWrapper):
    """Wraps a Hugging Face Causal LM for generation."""
    def __init__(self, model_name="distilgpt2"):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            logger.info(f"Loading Hugging Face model: {model_name}")
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token # Set pad token for batching if needed
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
            logger.info(f"Hugging Face model loaded on {self.device}")
        except Exception as e:
            logger.error(f"Error loading Hugging Face model: {e}")
            raise

    def generate(self, prompt: str, max_length: int = 50, **kwargs) -> str:
        """Generates text based on the prompt."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Set default generation parameters
        generation_params = {
            "max_length": max_length + inputs.input_ids.shape[1], # Adjust max_length
            "pad_token_id": self.tokenizer.eos_token_id,
            "num_return_sequences": 1,
            "do_sample": True, # Default: enable sampling
            "top_k": 50,       # Default: top-k sampling
            "top_p": 0.95      # Default: nucleus sampling
        }
        
        # Update with user-provided kwargs (this will override defaults)
        generation_params.update(kwargs)
        
        # Generate text
        outputs = self.model.generate(
            **inputs,
            **generation_params
        )
        
        # Decode only the newly generated tokens
        generated_text = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return generated_text.strip()

    def get_probabilities(self, prompt: str):
        """
        Placeholder for getting token probabilities (needed for some uncertainty methods).
        Requires model configuration and potentially custom generation logic.
        """
        # This is complex and model-specific. For now, return None.
        # Implementation might involve running the model forward and accessing logits.
        logger.warning("get_probabilities is not fully implemented for Hugging Face models.")
        return None

class OpenAILLMWrapper(BaseLLMWrapper):
    """Wraps OpenAI API for text generation."""
    def __init__(self, model_name="gpt-4.1-nano"):
        try:
            import openai
            from openai import OpenAI
            
            # Get API key from environment
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable is not set")
            
            self.client = OpenAI(api_key=api_key)
            self.model_name = model_name
            logger.info(f"OpenAI wrapper initialized with model: {model_name}")
        except ImportError:
            logger.error("OpenAI package not installed. Install with 'pip install openai'")
            raise

    def generate(self, prompt: str, max_length: int = 50, temperature: float = 0.7, **kwargs) -> str:
        """Generates text using OpenAI API."""
        try:
            # Convert max_length to max_tokens for OpenAI
            max_tokens = kwargs.get("max_tokens", max_length)
            
            # Set default parameters for OpenAI
            params = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            
            # Add OpenAI specific parameters if provided
            for key in ["top_p", "frequency_penalty", "presence_penalty", "stop"]:
                if key in kwargs:
                    params[key] = kwargs[key]
            
            # Rate limiting to avoid hitting OpenAI API rate limits (3 req/min for free tier)
            time.sleep(0.2)  # Small delay to prevent rapid successive requests
            
            # Call OpenAI API
            response = self.client.chat.completions.create(**params)
            
            # Extract and return the generated text
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            return f"Error generating text: {str(e)}"

    def get_probabilities(self, prompt: str):
        """
        OpenAI API doesn't provide direct access to token probabilities in the same way.
        For uncertainty estimation, we could use logprobs parameter, but it's not 
        fully compatible with the expected interface.
        """
        logger.warning("get_probabilities not fully implemented for OpenAI models.")
        return None

def LLMWrapper(model_name: str = "gpt-4.1-nano", use_openai: bool = True):
    """
    Factory function to create the appropriate LLM wrapper.
    
    Args:
        model_name: Name of the model to use
        use_openai: Whether to use OpenAI (True) or Hugging Face (False)
    
    Returns:
        An LLM wrapper instance
    """
    # Check if we should force Hugging Face based on model name
    huggingface_models = ["distilgpt2", "gpt2", "gpt2-medium", "gpt2-large"]
    
    # If model is explicitly a Hugging Face model, use Hugging Face
    if model_name in huggingface_models:
        return HuggingFaceLLMWrapper(model_name)
    
    # If OpenAI is specified and the model name looks like an OpenAI model
    if use_openai and (model_name.startswith("gpt-") or model_name in ["text-davinci-003", "davinci"]):
        return OpenAILLMWrapper(model_name)
    
    # Default: use OpenAI if specified, otherwise fall back to Hugging Face
    if use_openai:
        return OpenAILLMWrapper(model_name)
    else:
        return HuggingFaceLLMWrapper(model_name)

# Example usage (optional, for testing)
if __name__ == '__main__':
    # Test Hugging Face wrapper
    hf_llm = LLMWrapper("distilgpt2", use_openai=False)
    prompt = "Once upon a time"
    response = hf_llm.generate(prompt)
    print(f"Hugging Face - Prompt: {prompt}")
    print(f"Hugging Face - Response: {response}")
    
    # Test OpenAI wrapper if API key is available
    if os.environ.get("OPENAI_API_KEY"):
        openai_llm = LLMWrapper("gpt-4.1-nano", use_openai=True)
        response = openai_llm.generate(prompt)
        print(f"OpenAI - Prompt: {prompt}")
        print(f"OpenAI - Response: {response}")
