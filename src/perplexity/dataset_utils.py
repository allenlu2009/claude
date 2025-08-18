"""
Dataset utilities for perplexity evaluation.

This module provides functions for loading and preprocessing datasets used in
perplexity evaluation, with error handling and validation.
"""

import logging
import os
import urllib.request
from typing import Optional, Any
import requests
from datasets import load_dataset
from ..config.model_configs import get_dataset_config, DatasetConfig

# Set up logging
logger = logging.getLogger(__name__)


class DatasetLoadError(Exception):
    """Custom exception for dataset loading errors."""

    pass


def load_dataset_text(
    dataset_name: str, 
    max_samples: Optional[int] = None,
    max_tokens: Optional[int] = None,
    use_all_tokens: bool = False
) -> str:
    """
    Load text from a dataset and concatenate into a single string.

    Args:
        dataset_name: Name of the dataset to load
        max_samples: Optional limit on number of samples to process
        max_tokens: Optional limit on number of tokens to extract
        use_all_tokens: If True, use all available tokens (ignores max_samples)

    Returns:
        Concatenated text string from the dataset

    Raises:
        DatasetLoadError: If dataset loading fails
    """
    try:
        dataset_config = get_dataset_config(dataset_name)
        return _load_text_from_config(dataset_config, max_samples, max_tokens, use_all_tokens)
    except Exception as e:
        raise DatasetLoadError(
            f"Failed to load dataset '{dataset_name}': {str(e)}"
        ) from e


def _load_text_from_config(
    config: DatasetConfig, 
    max_samples: Optional[int] = None,
    max_tokens: Optional[int] = None,
    use_all_tokens: bool = False
) -> str:
    """
    Load text from dataset using configuration.

    Args:
        config: Dataset configuration
        max_samples: Optional limit on samples
        max_tokens: Optional limit on tokens to extract
        use_all_tokens: If True, use all available tokens

    Returns:
        Concatenated text string
    """
    logger.info(f"Loading dataset: {config.name}")

    try:
        # Use custom loaders for PTB and Shakespeare to avoid HuggingFace issues
        if config.name == "PTB":
            text = _load_ptb_text()
        elif config.name == "Shakespeare":
            text = _load_shakespeare_text()
        else:
            # Handle standard HuggingFace datasets
            if isinstance(config.hf_dataset, tuple):
                dataset_name, dataset_config = config.hf_dataset
                dataset = load_dataset(dataset_name, dataset_config, split=config.split)
            else:
                dataset = load_dataset(config.hf_dataset, split=config.split)

            logger.info(f"Dataset loaded successfully. Total samples: {len(dataset)}")

            # Handle different sampling strategies
            if use_all_tokens:
                logger.info("Using all available tokens from dataset")
            elif max_samples is not None:
                dataset = dataset.select(range(min(max_samples, len(dataset))))
                logger.info(f"Limited to {len(dataset)} samples")
            elif config.max_samples is not None:
                dataset = dataset.select(range(min(config.max_samples, len(dataset))))
                logger.info(f"Limited to {len(dataset)} samples (from config)")

            # Extract text based on dataset-specific handling
            text = _extract_text_from_dataset(dataset, config)
        
        # Apply token-based truncation if requested
        if max_tokens is not None:
            text = _truncate_text_by_tokens(text, max_tokens)
            logger.info(f"Text truncated to approximately {max_tokens} tokens")

        logger.info(f"Text extraction complete. Total length: {len(text)} characters")
        return text

    except Exception as e:
        logger.error(f"Error loading dataset {config.name}: {str(e)}")
        raise


def _load_ptb_text() -> str:
    """
    Load PTB test data from GitHub repository.
    
    Returns:
        PTB test text as string
    """
    url = "https://raw.githubusercontent.com/tomsercu/lstm/master/data/ptb.test.txt"
    
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    ptb_file = "data/ptb_test.txt"
    
    # Download if not already present
    if not os.path.exists(ptb_file):
        logger.info("Downloading PTB test data...")
        try:
            urllib.request.urlretrieve(url, ptb_file)
            logger.info("PTB test data downloaded successfully!")
        except Exception as e:
            logger.error(f"PTB download failed: {e}")
            logger.info("Falling back to WikiText-2...")
            # Fallback to WikiText-2
            from datasets import load_dataset
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
            text = "\n\n".join([item for item in dataset["text"] if item.strip()])
            return text
    
    # Read the file
    try:
        with open(ptb_file, "r", encoding='utf-8') as f:
            text = f.read()
        logger.info(f"PTB test data loaded from local file: {len(text)} characters")
        return text
    except Exception as e:
        logger.error(f"Failed to read PTB file: {e}")
        raise


def _load_shakespeare_text() -> str:
    """
    Load Shakespeare text directly from Karpathy's repository.
    
    Returns:
        Shakespeare text as string
    """
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    
    try:
        logger.info("Downloading Shakespeare text...")
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes
        text = response.text
        logger.info(f"Shakespeare text downloaded successfully: {len(text)} characters")
        return text
    except Exception as e:
        logger.error(f"Failed to download Shakespeare text: {e}")
        raise


def _extract_text_from_dataset(dataset: Any, config: DatasetConfig) -> str:
    """
    Extract and concatenate text from dataset based on configuration.

    Args:
        dataset: Loaded HuggingFace dataset
        config: Dataset configuration

    Returns:
        Concatenated text string
    """
    text_field = config.text_field

    if config.name == "PTB":
        # Special handling for PTB dataset - join sentences with newlines
        texts = dataset[text_field]
        return " \n ".join(texts)
    elif config.name in ["Wikitext2", "Wikitext103"]:
        # Join with double newlines to preserve document structure
        texts = dataset[text_field]
        # Filter out empty strings
        texts = [text for text in texts if text.strip()]
        return "\n\n".join(texts)
    elif config.name == "Shakespeare":
        # Shakespeare dataset might be structured differently
        texts = dataset[text_field]
        return "\n\n".join(texts) if isinstance(texts, list) else texts
    elif config.name == "C4":
        # C4 has clean text, join with double newlines
        texts = dataset[text_field]
        return "\n\n".join(texts)
    else:
        # Default handling - join with double newlines
        texts = dataset[text_field]
        if isinstance(texts, list):
            return "\n\n".join(str(text) for text in texts if str(text).strip())
        else:
            return str(texts)


def validate_text_length(text: str, min_length: int = 1000) -> bool:
    """
    Validate that text meets minimum length requirements.

    Args:
        text: Text to validate
        min_length: Minimum required length in characters

    Returns:
        True if text is valid, False otherwise
    """
    if len(text) < min_length:
        logger.warning(f"Text length ({len(text)}) is below minimum ({min_length})")
        return False
    return True


def get_text_stats(text: str) -> dict:
    """
    Get basic statistics about the text.

    Args:
        text: Text to analyze

    Returns:
        Dictionary with text statistics
    """
    lines = text.split("\n")
    words = text.split()

    return {
        "total_characters": len(text),
        "total_lines": len(lines),
        "total_words": len(words),
        "avg_line_length": (
            sum(len(line) for line in lines) / len(lines) if lines else 0
        ),
        "avg_word_length": (
            sum(len(word) for word in words) / len(words) if words else 0
        ),
    }


def preview_text(text: str, num_chars: int = 500) -> str:
    """
    Get a preview of the text for inspection.

    Args:
        text: Text to preview
        num_chars: Number of characters to include in preview

    Returns:
        Preview string
    """
    if len(text) <= num_chars:
        return text
    return text[:num_chars] + "..."


def _truncate_text_by_tokens(text: str, max_tokens: int) -> str:
    """
    Truncate text to approximately max_tokens using rough estimation.
    
    Args:
        text: Text to truncate
        max_tokens: Maximum number of tokens (approximate)
        
    Returns:
        Truncated text string
        
    Note:
        Uses rough estimation of ~4 characters per token for English text.
        This avoids expensive tokenization during data loading.
    """
    if max_tokens <= 0:
        return ""
    
    # Rough estimation: ~4 characters per token for English text
    # This is approximate but avoids expensive tokenization during loading
    estimated_chars = max_tokens * 4
    
    if len(text) <= estimated_chars:
        return text
    
    # Truncate at word boundary near the estimated position
    truncated = text[:estimated_chars]
    
    # Find last complete word to avoid cutting mid-word
    last_space = truncated.rfind(' ')
    if last_space > estimated_chars * 0.9:  # Only if not too far back
        truncated = truncated[:last_space]
    
    logger.info(f"Truncated text from {len(text)} to {len(truncated)} characters "
                f"(targeting ~{max_tokens} tokens)")
    
    return truncated


# Convenience functions for specific datasets
def load_wikitext2(
    max_samples: Optional[int] = None,
    max_tokens: Optional[int] = None,
    use_all_tokens: bool = False
) -> str:
    """Load WikiText-2 dataset."""
    return load_dataset_text("Wikitext2", max_samples, max_tokens, use_all_tokens)


def load_ptb(
    max_samples: Optional[int] = None,
    max_tokens: Optional[int] = None,
    use_all_tokens: bool = False
) -> str:
    """Load Penn Treebank dataset."""
    return load_dataset_text("PTB", max_samples, max_tokens, use_all_tokens)


def load_shakespeare(
    max_samples: Optional[int] = None,
    max_tokens: Optional[int] = None,
    use_all_tokens: bool = False
) -> str:
    """Load Shakespeare dataset."""
    return load_dataset_text("Shakespeare", max_samples, max_tokens, use_all_tokens)
