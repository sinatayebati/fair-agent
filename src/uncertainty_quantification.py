from typing import Dict, Any, List, Tuple, Optional, Union
import numpy as np
import torch
import logging
from sklearn.model_selection import train_test_split
from scipy.stats import entropy

logger = logging.getLogger('AI_Framework')

class UncertaintyQuantifier:
    """
    Estimates the uncertainty of LLM outputs using various methods, including conformal prediction.
    
    Methods supported:
    - softmax_entropy: Uses token probability distribution entropy
    - mc_dropout: Monte Carlo dropout for stochastic sampling
    - conformal: Conformal prediction for confidence intervals with coverage guarantees
    - ensemble: Uses multiple samples to estimate variance
    """

    def __init__(self, method: str = "ensemble", alpha: float = 0.1, n_samples: int = 5):
        """
        Initialize uncertainty quantifier.
        
        Args:
            method: Uncertainty estimation method
            alpha: Significance level for conformal prediction (1-alpha = confidence)
            n_samples: Number of samples for MC methods
        """
        self.method = method
        self.alpha = alpha  # For conformal prediction
        self.n_samples = n_samples  # For MC dropout and ensemble methods
        self.calibrated = False  # Whether conformal model is calibrated
        self.conformity_scores = []  # For conformal prediction calibration
        
        logger.info(f"Initializing UncertaintyQuantifier with method: {self.method}")
        
        if method == "conformal":
            logger.info(f"Conformal prediction will use alpha={alpha} (confidence: {(1-alpha)*100:.0f}%)")
            
        elif method == "mc_dropout":
            logger.info(f"MC Dropout will use {n_samples} samples")
            
        elif method == "ensemble":
            logger.info(f"Ensemble method will use {n_samples} samples")

    def calibrate_conformal(self, calibration_data: List[Dict[str, Any]], llm_wrapper) -> bool:
        """
        Calibrates the conformal predictor with example data.
        
        Args:
            calibration_data: List of examples with prompts and ground truth
            llm_wrapper: LLM wrapper to generate responses
            
        Returns:
            True if calibration was successful
        """
        if not calibration_data or not llm_wrapper:
            logger.warning("Cannot calibrate conformal predictor: missing data or model")
            return False
            
        logger.info(f"Calibrating conformal predictor with {len(calibration_data)} examples")
        
        self.conformity_scores = []
        
        # Process each calibration example
        for item in calibration_data:
            prompt = item.get("prompt", "")
            if not prompt:
                continue
                
            try:
                # Get LLM response and token probabilities
                response = llm_wrapper.generate(prompt)
                probs = llm_wrapper.get_probabilities(prompt)
                
                if not probs:
                    logger.warning("Cannot calibrate: probability information not available")
                    continue
                    
                # Use maximum token probability as conformity score
                if isinstance(probs, list):
                    # Handle case where we get a list of token probabilities
                    max_prob = max(probs) if probs else 0.0
                elif isinstance(probs, dict):
                    # Handle case where we get a dict of token:probability pairs
                    max_prob = max(probs.values()) if probs else 0.0
                elif isinstance(probs, np.ndarray) or isinstance(probs, torch.Tensor):
                    # Handle case where we get a numpy array or tensor
                    max_prob = float(probs.max())
                else:
                    max_prob = 0.0
                    
                # Invert probabilities for error metric (higher means more uncertain)
                self.conformity_scores.append(1.0 - max_prob)
            
            except Exception as e:
                logger.error(f"Error during conformal calibration: {e}")
                
        # Need at least a few samples to calibrate
        if len(self.conformity_scores) >= 10:
            # Sort scores for quantile calculation
            self.conformity_scores.sort()
            self.calibrated = True
            logger.info(f"Conformal calibration complete with {len(self.conformity_scores)} scores")
            return True
        else:
            logger.warning(f"Insufficient calibration data: {len(self.conformity_scores)} examples")
            return False

    def _conformal_prediction(self, token_probs: Union[List[float], Dict[str, float], np.ndarray]) -> Dict[str, Any]:
        """
        Calculate conformal prediction interval.
        
        Args:
            token_probs: Token probabilities from LLM
            
        Returns:
            Dictionary with conformal prediction results
        """
        if not self.calibrated or not self.conformity_scores:
            return {
                "confidence_score": 0.5,  # Default 0.5 confidence when not calibrated
                "uncertainty_score": 0.5,
                "coverage_guarantee": "Not calibrated",
                "calibrated": False
            }
            
        # Process token probabilities to get a single score
        if isinstance(token_probs, list):
            max_prob = max(token_probs) if token_probs else 0.0
        elif isinstance(token_probs, dict):
            max_prob = max(token_probs.values()) if token_probs else 0.0
        elif isinstance(token_probs, np.ndarray) or isinstance(token_probs, torch.Tensor):
            max_prob = float(token_probs.max())
        else:
            max_prob = 0.0
        
        # Invert probability to get an error metric
        nonconformity_score = 1.0 - max_prob
        
        # Calculate the quantile based on desired confidence level
        n = len(self.conformity_scores)
        quantile_idx = int(np.ceil((n+1) * (1-self.alpha))) - 1
        quantile_idx = min(max(quantile_idx, 0), n-1)  # Ensure valid index
        
        # Get quantile threshold
        threshold = self.conformity_scores[quantile_idx]
        
        # Calculate coverage guarantee
        coverage_guarantee = 1.0 - self.alpha
        
        # Compare nonconformity score to threshold
        is_conformal = nonconformity_score <= threshold
        
        # Calculate confidence/uncertainty scores
        normalized_score = nonconformity_score / (threshold if threshold > 0 else 1.0)
        confidence_score = 1.0 - min(normalized_score, 1.0)
        
        return {
            "confidence_score": confidence_score,
            "uncertainty_score": min(normalized_score, 1.0),
            "coverage_guarantee": f"{coverage_guarantee:.2f}",
            "is_conformal": is_conformal,
            "calibrated": True
        }

    def _softmax_entropy(self, token_probs: Any) -> Dict[str, Any]:
        """
        Calculate uncertainty based on token probability entropy.
        
        Args:
            token_probs: Token probabilities from LLM
            
        Returns:
            Dictionary with entropy-based uncertainty results
        """
        if isinstance(token_probs, list):
            probs = np.array(token_probs)
        elif isinstance(token_probs, dict):
            probs = np.array(list(token_probs.values()))
        elif isinstance(token_probs, np.ndarray) or isinstance(token_probs, torch.Tensor):
            probs = np.array(token_probs)
        else:
            # If no valid probabilities, return high uncertainty
            return {
                "confidence_score": 0.0,
                "uncertainty_score": 1.0,
                "method": "softmax_entropy",
                "entropy": float('inf')
            }
        
        # Ensure probabilities sum to 1
        if probs.sum() > 0:
            probs = probs / probs.sum()
            
        # Calculate entropy (higher = more uncertain)
        ent = float(entropy(probs))
        
        # Normalize entropy to [0,1] using a reasonable upper bound
        # Max entropy is log(n) where n is vocabulary size, but we use a reasonable limit
        max_entropy = 5.0  # Reasonable upper bound for most cases
        normalized_entropy = min(ent / max_entropy, 1.0)
        
        return {
            "confidence_score": 1.0 - normalized_entropy,
            "uncertainty_score": normalized_entropy,
            "method": "softmax_entropy",
            "entropy": ent
        }

    def _mc_dropout(self, prompt: str, llm_wrapper, n_samples: int = 5) -> Dict[str, Any]:
        """
        Estimate uncertainty using Monte Carlo dropout sampling.
        
        Args:
            prompt: Input prompt
            llm_wrapper: LLM wrapper for generation
            n_samples: Number of samples to generate
            
        Returns:
            Dictionary with MC dropout uncertainty results
        """
        if not llm_wrapper:
            return {
                "confidence_score": 0.0,
                "uncertainty_score": 1.0,
                "method": "mc_dropout",
                "samples": []
            }
            
        samples = []
        try:
            # Generate multiple samples with dropout enabled
            # Note: This assumes the model supports dropout during inference
            for i in range(n_samples):
                # Use temperature but avoid passing redundant do_sample parameter
                response = llm_wrapper.generate(prompt, temperature=1.0)
                samples.append(response)
                
            # Calculate variability across samples
            if len(samples) >= 2:
                # Calculate embeddings if embedding model is available
                try:
                    from sentence_transformers import SentenceTransformer
                    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
                    embeddings = model.encode(samples)
                    
                    # Calculate pairwise cosine similarities
                    from sklearn.metrics.pairwise import cosine_similarity
                    sim_matrix = cosine_similarity(embeddings)
                    
                    # Average similarity (lower = more uncertain)
                    np.fill_diagonal(sim_matrix, 0)  # Exclude self-similarity
                    mean_sim = sim_matrix.sum() / (sim_matrix.size - n_samples)  # Exclude diagonal
                    
                    # Convert to uncertainty score (higher = more uncertain)
                    uncertainty_score = 1.0 - mean_sim
                    confidence_score = mean_sim
                    
                    return {
                        "confidence_score": confidence_score,
                        "uncertainty_score": uncertainty_score,
                        "method": "mc_dropout",
                        "n_samples": n_samples,
                        "mean_similarity": mean_sim
                    }
                    
                except Exception as e:
                    # If embedding calculation fails, fall back to simpler method
                    logger.warning(f"Embedding calculation failed: {e}")
                    
                # Fallback: Use length variance as a simple heuristic
                lengths = [len(s) for s in samples]
                length_var = np.var(lengths)
                length_mean = np.mean(lengths)
                
                # Normalize variance
                normalized_var = min(length_var / (length_mean if length_mean > 0 else 1.0), 1.0)
                
                return {
                    "confidence_score": 1.0 - normalized_var,
                    "uncertainty_score": normalized_var,
                    "method": "mc_dropout",
                    "n_samples": n_samples,
                    "length_variance": float(length_var)
                }
            else:
                return {
                    "confidence_score": 0.0,
                    "uncertainty_score": 1.0,
                    "method": "mc_dropout",
                    "error": "Insufficient samples"
                }
                
        except Exception as e:
            logger.error(f"Error in MC dropout sampling: {e}")
            return {
                "confidence_score": 0.0,
                "uncertainty_score": 1.0,
                "method": "mc_dropout",
                "error": str(e)
            }

    def _ensemble_method(self, prompt: str, llm_wrapper, n_samples: int = 5) -> Dict[str, Any]:
        """
        Estimate uncertainty using ensemble of multiple generations with different parameters.
        
        Args:
            prompt: Input prompt
            llm_wrapper: LLM wrapper for generation
            n_samples: Number of samples to generate
            
        Returns:
            Dictionary with ensemble uncertainty results
        """
        if not llm_wrapper:
            return {
                "confidence_score": 0.0,
                "uncertainty_score": 1.0,
                "method": "ensemble"
            }
            
        samples = []
        try:
            # Generate multiple samples with different temperatures
            for i in range(n_samples):
                # Vary temperature for diversity
                temp = 0.6 + (0.8 * i / max(1, n_samples-1))
                # Avoid passing do_sample explicitly as it's included in default params
                response = llm_wrapper.generate(prompt, temperature=temp)
                samples.append(response)
                
            # Calculate diversity metrics using the same approach as MC dropout
            if len(samples) >= 2:
                try:
                    from sentence_transformers import SentenceTransformer
                    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
                    embeddings = model.encode(samples)
                    
                    # Calculate pairwise cosine similarities
                    from sklearn.metrics.pairwise import cosine_similarity
                    sim_matrix = cosine_similarity(embeddings)
                    
                    # Average similarity (lower = more uncertain)
                    np.fill_diagonal(sim_matrix, 0)  # Exclude self-similarity
                    mean_sim = sim_matrix.sum() / (sim_matrix.size - n_samples)  # Exclude diagonal
                    
                    # Convert to uncertainty score (higher = more uncertain)
                    uncertainty_score = 1.0 - mean_sim
                    confidence_score = mean_sim
                    
                    return {
                        "confidence_score": confidence_score,
                        "uncertainty_score": uncertainty_score,
                        "method": "ensemble",
                        "n_samples": n_samples,
                        "mean_similarity": mean_sim
                    }
                    
                except Exception as e:
                    logger.warning(f"Embedding calculation failed: {e}")
                    
                # Fallback method
                lengths = [len(s) for s in samples]
                length_var = np.var(lengths)
                length_mean = np.mean(lengths)
                
                # Normalize variance
                normalized_var = min(length_var / (length_mean if length_mean > 0 else 1.0), 1.0)
                
                return {
                    "confidence_score": 1.0 - normalized_var,
                    "uncertainty_score": normalized_var,
                    "method": "ensemble",
                    "n_samples": n_samples,
                    "length_variance": float(length_var)
                }
            else:
                return {
                    "confidence_score": 0.0,
                    "uncertainty_score": 1.0,
                    "method": "ensemble",
                    "error": "Insufficient samples"
                }
                
        except Exception as e:
            logger.error(f"Error in ensemble method: {e}")
            return {
                "confidence_score": 0.0,
                "uncertainty_score": 1.0,
                "method": "ensemble",
                "error": str(e)
            }

    def calculate_uncertainty(self, prompt: str, response: str, llm_wrapper=None) -> Dict[str, Any]:
        """
        Calculate uncertainty score for a model response.
        
        Args:
            prompt: Input prompt
            response: Model response
            llm_wrapper: LLM wrapper for additional methods
            
        Returns:
            Dictionary with uncertainty analysis results
        """
        if self.method == "softmax_entropy" and llm_wrapper:
            # Try to get token probabilities from the model
            probs = llm_wrapper.get_probabilities(prompt)
            if probs:
                return self._softmax_entropy(probs)
            else:
                logger.warning("Softmax entropy calculation failed: probabilities not available")
                # Fall back to ensemble method if probabilities aren't available
                return self._ensemble_method(prompt, llm_wrapper, self.n_samples)

        elif self.method == "mc_dropout" and llm_wrapper:
            # Use MC dropout method if model supports it
            return self._mc_dropout(prompt, llm_wrapper, self.n_samples)

        elif self.method == "conformal" and llm_wrapper:
            # Use conformal prediction if calibrated
            probs = llm_wrapper.get_probabilities(prompt)
            if probs:
                return self._conformal_prediction(probs)
            else:
                logger.warning("Conformal prediction failed: probabilities not available")
                return {
                    "confidence_score": 0.5,
                    "uncertainty_score": 0.5,
                    "coverage_guarantee": "Not available",
                    "calibrated": self.calibrated,
                    "error": "Probabilities not available"
                }

        elif self.method == "ensemble" and llm_wrapper:
            # Use ensemble method
            return self._ensemble_method(prompt, llm_wrapper, self.n_samples)

        else:
            # Fallback placeholder implementation
            logger.warning(f"Using placeholder uncertainty estimation for method {self.method}")
            
            # Use response length as a very basic uncertainty heuristic
            baseline_length = 100  # Expected reasonable response length
            actual_length = len(response)
            
            # Very short responses might indicate uncertainty
            length_ratio = actual_length / baseline_length if baseline_length > 0 else 0
            length_factor = min(length_ratio, 2.0) / 2.0  # Normalize to [0, 1]
            
            # Check for uncertainty indicators in text
            uncertainty_indicators = [
                "I'm not sure", "I don't know", "uncertain", "might be", 
                "possibly", "perhaps", "could be", "may be"
            ]
            
            has_uncertainty_indicators = any(indicator.lower() in response.lower() 
                                          for indicator in uncertainty_indicators)
            
            # Combine factors (length and indicators)
            uncertainty_score = 0.7 if has_uncertainty_indicators else 0.3
            uncertainty_score = (uncertainty_score + (1.0 - length_factor)) / 2.0
            
            return {
                "method": "placeholder",
                "uncertainty_score": uncertainty_score,
                "confidence_score": 1.0 - uncertainty_score,
                "coverage_guarantee": "N/A (Placeholder method)",
                "factors": {
                    "length_factor": length_factor,
                    "has_uncertainty_indicators": has_uncertainty_indicators
                }
            }

# Example usage (optional)
if __name__ == '__main__':
    quantifier = UncertaintyQuantifier(method="placeholder")
    uncertainty = quantifier.calculate_uncertainty("Why is the sky blue?", "The sky appears blue due to Rayleigh scattering.")
    print(uncertainty)

    quantifier_conf = UncertaintyQuantifier(method="conformal")
    uncertainty_conf = quantifier_conf.calculate_uncertainty("...", "...")
    print(uncertainty_conf)

