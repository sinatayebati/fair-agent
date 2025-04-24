import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class LLMWrapper:
    """Wraps a Hugging Face Causal LM for generation."""
    def __init__(self, model_name="distilgpt2"):
        print(f"Loading model: {model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token # Set pad token for batching if needed
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        print(f"Model loaded on {self.device}")

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
        print("Warning: get_probabilities is not fully implemented.")
        return None

# Example usage (optional, for testing)
if __name__ == '__main__':
    llm = LLMWrapper()
    prompt = "Once upon a time"
    response = llm.generate(prompt)
    print(f"Prompt: {prompt}")
    print(f"Response: {response}")
