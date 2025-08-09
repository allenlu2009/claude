"""
Perplexity evaluation system for language models.

This package provides tools for evaluating language model perplexity using
sliding window strategy, optimized for memory-constrained GPUs.
"""

from .evaluator import PerplexityEvaluator
from .model_loader import load_model_and_tokenizer, unload_model
from .models import ModelConfig, ChunkParams, EvaluationResult, PerplexityConfig

__version__ = "0.1.0"
__all__ = [
    "PerplexityEvaluator",
    "load_model_and_tokenizer",
    "unload_model",
    "ModelConfig",
    "ChunkParams",
    "EvaluationResult",
    "PerplexityConfig",
]
