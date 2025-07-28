"""
Model loading utilities with memory optimization and flash attention support.

This module provides functions for loading language models and tokenizers with
automatic memory optimization and flash attention fallback strategies.
"""

import logging
import torch
from typing import Tuple, Optional, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM  # type: ignore
from ..config.model_configs import get_model_config, ModelConfig

# Set up logging
logger = logging.getLogger(__name__)


class ModelLoadError(Exception):
    """Custom exception for model loading errors."""

    pass


class MemoryConstraintError(Exception):
    """Custom exception for memory constraint violations."""

    pass


def check_memory_constraints(model_config: ModelConfig, memory_limit_gb: float) -> bool:
    """
    Check if model fits within memory constraints.

    Args:
        model_config: Model configuration
        memory_limit_gb: Memory limit in GB

    Returns:
        True if model fits within constraints

    Raises:
        MemoryConstraintError: If model exceeds memory limit
    """
    if model_config.memory_gb > memory_limit_gb:
        raise MemoryConstraintError(
            f"Model '{model_config.name}' requires {model_config.memory_gb}GB "
            f"but memory limit is {memory_limit_gb}GB"
        )
    return True


def get_device_info() -> Dict[str, Any]:
    """Get information about available compute devices."""
    device_info = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }

    if torch.cuda.is_available():
        device_props = torch.cuda.get_device_properties(0)
        device_info.update(
            {
                "gpu_name": device_props.name,
                "total_memory_gb": device_props.total_memory / (1024**3),
                "compute_capability": f"{device_props.major}.{device_props.minor}",
            }
        )

        # Check current memory usage
        allocated_mb = torch.cuda.memory_allocated() / (1024**2)
        cached_mb = torch.cuda.memory_reserved() / (1024**2)
        device_info.update(
            {
                "allocated_memory_mb": allocated_mb,
                "cached_memory_mb": cached_mb,
            }
        )

    return device_info


def clear_gpu_memory():
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("GPU memory cache cleared")


def load_model_and_tokenizer(
    model_name: str,
    device: str = "cuda",
    use_flash_attention: bool = True,
    memory_limit_gb: Optional[float] = None,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load model and tokenizer with memory optimization and flash attention support.

    Args:
        model_name: Name of the model to load
        device: Device to use ('cuda', 'cpu', or 'auto')
        use_flash_attention: Whether to attempt flash attention
        memory_limit_gb: Optional memory limit in GB

    Returns:
        Tuple of (model, tokenizer)

    Raises:
        ModelLoadError: If model loading fails
        MemoryConstraintError: If model exceeds memory limits
    """
    logger.info(f"Loading model: {model_name}")

    # Get model configuration
    try:
        model_config = get_model_config(model_name)
    except ValueError as e:
        raise ModelLoadError(f"Unknown model: {model_name}") from e

    # Check memory constraints
    if memory_limit_gb is not None:
        check_memory_constraints(model_config, memory_limit_gb)

    # Log device information
    device_info = get_device_info()
    logger.info(f"Device info: {device_info}")

    # Determine actual device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Using device: {device}")

    try:
        # Load tokenizer first
        tokenizer = _load_tokenizer(model_config.hf_name)

        # Load model with appropriate attention implementation
        model = _load_model_with_attention_fallback(
            model_config, device, use_flash_attention
        )

        # Configure model for evaluation
        model.eval()  # type: ignore

        # Add padding token if needed
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token  # type: ignore
            logger.info("Set pad_token to eos_token")

        # Log final memory usage
        if device == "cuda":
            final_memory = torch.cuda.memory_allocated() / (1024**2)
            logger.info(f"Model loaded. GPU memory usage: {final_memory:.1f} MB")

        return model, tokenizer

    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {str(e)}")
        raise ModelLoadError(f"Failed to load model {model_name}") from e


def _load_tokenizer(hf_name: str) -> AutoTokenizer:
    """Load tokenizer with proper configuration."""
    logger.info(f"Loading tokenizer: {hf_name}")

    tokenizer = AutoTokenizer.from_pretrained(hf_name, trust_remote_code=True)  # type: ignore

    logger.info("Tokenizer loaded successfully")
    return tokenizer


def _load_model_with_attention_fallback(
    model_config: ModelConfig, device: str, use_flash_attention: bool
) -> AutoModelForCausalLM:
    """
    Load model with flash attention, falling back to eager if needed.

    Args:
        model_config: Model configuration
        device: Target device
        use_flash_attention: Whether to attempt flash attention

    Returns:
        Loaded model
    """
    hf_name = model_config.hf_name

    # Base configuration
    base_config = {
        "trust_remote_code": True,
        "torch_dtype": "auto",
    }

    # Add device mapping for GPU
    if device == "cuda":
        base_config["device_map"] = "cuda"

    # Try flash attention first if supported and requested
    if use_flash_attention and model_config.supports_flash_attention:
        try:
            logger.info("Attempting to load with flash attention")
            flash_config = {**base_config, "attn_implementation": "flash_attention_2"}

            model = AutoModelForCausalLM.from_pretrained(hf_name, **flash_config)  # type: ignore

            # Enable gradient checkpointing for memory efficiency
            if hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable()
                logger.info("Gradient checkpointing enabled")

            logger.info("Model loaded successfully with flash attention")
            return model

        except Exception as e:
            logger.warning(f"Flash attention failed: {str(e)}")
            logger.info("Falling back to eager attention")

    # Fallback to eager attention
    try:
        logger.info("Loading with eager attention")
        eager_config = {**base_config, "attn_implementation": "eager"}

        model = AutoModelForCausalLM.from_pretrained(hf_name, **eager_config)  # type: ignore

        # Enable gradient checkpointing for memory efficiency
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")

        logger.info("Model loaded successfully with eager attention")
        return model

    except Exception as e:
        logger.error(f"Model loading failed with eager attention: {str(e)}")
        raise


def unload_model(model: AutoModelForCausalLM) -> None:
    """
    Properly unload model and clear memory.

    Args:
        model: Model to unload
    """
    logger.info("Unloading model and clearing memory")

    # Delete model
    del model

    # Clear GPU cache
    clear_gpu_memory()

    logger.info("Model unloaded successfully")


def get_memory_usage() -> Dict[str, float]:
    """
    Get current memory usage statistics.

    Returns:
        Dictionary with memory usage in MB
    """
    if not torch.cuda.is_available():
        return {"cuda_available": False}

    allocated_mb = torch.cuda.memory_allocated() / (1024**2)
    cached_mb = torch.cuda.memory_reserved() / (1024**2)
    max_allocated_mb = torch.cuda.max_memory_allocated() / (1024**2)

    return {
        "cuda_available": True,
        "allocated_mb": allocated_mb,
        "cached_mb": cached_mb,
        "max_allocated_mb": max_allocated_mb,
    }


def estimate_model_memory(
    model_config: ModelConfig, context_length: int = 2048
) -> float:
    """
    Estimate memory usage for a model with given context length.

    Args:
        model_config: Model configuration
        context_length: Context length to estimate for

    Returns:
        Estimated memory usage in GB
    """
    # Base model memory
    base_memory = model_config.memory_gb

    # Additional memory for longer contexts (rough estimate)
    # Attention memory scales quadratically with sequence length
    base_context = 2048
    if context_length > base_context:
        context_multiplier = (
            context_length / base_context
        ) ** 1.5  # Between linear and quadratic
        additional_memory = base_memory * 0.2 * (context_multiplier - 1)
        return base_memory + additional_memory

    return base_memory
