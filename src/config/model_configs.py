"""
Model and dataset configurations for perplexity evaluation.

This module contains predefined configurations for supported models and datasets,
including memory requirements and compatibility information.
"""

from typing import Dict
from ..perplexity.models import ModelConfig, DatasetConfig


# Model configurations with memory requirements based on research
MODEL_CONFIGS: Dict[str, ModelConfig] = {
    "gpt2": ModelConfig(
        name="gpt2",
        hf_name="gpt2",
        max_length=1024,
        memory_gb=1.5,
        supports_flash_attention=False,  # Older architecture
    ),
    "gpt2-large": ModelConfig(
        name="gpt2-large",
        hf_name="gpt2-large",
        max_length=1024,
        memory_gb=3.0,
        supports_flash_attention=False,
    ),
    "gpt2-xl": ModelConfig(
        name="gpt2-xl",
        hf_name="gpt2-xl",
        max_length=1024,
        memory_gb=6.0,
        supports_flash_attention=False,
    ),
    "Phi3-mini-4k": ModelConfig(
        name="Phi3-mini-4k",
        hf_name="microsoft/Phi-3-mini-4k-instruct",
        max_length=4096,
        memory_gb=7.5,  # 7.1-8.5 GB from research
        supports_flash_attention=True,
    ),
    "Phi4-mini": ModelConfig(
        name="Phi4-mini",
        hf_name="microsoft/Phi-4-mini-instruct",
        max_length=131072,
        memory_gb=7.7,  # From research
        supports_flash_attention=True,
    ),
    "Phi4-mini-flash": ModelConfig(
        name="Phi4-mini-flash",
        hf_name="microsoft/Phi-4-mini-flash-reasoning",
        max_length=262144,
        memory_gb=8.5,  # Estimated based on larger context
        supports_flash_attention=True,
    ),
    "Llama3.2-1B": ModelConfig(
        name="Llama3.2-1B",
        hf_name="meta-llama/Llama-3.2-1B-Instruct",
        max_length=8192,
        memory_gb=2.5,
        supports_flash_attention=True,
    ),
    "Llama3.2-3B": ModelConfig(
        name="Llama3.2-3B",
        hf_name="meta-llama/Llama-3.2-3B",
        max_length=8192,
        memory_gb=6.5,
        supports_flash_attention=True,
    ),
    "Qwen2.5-3B": ModelConfig(
        name="Qwen2.5-3B",
        hf_name="Qwen/Qwen2.5-3B-Instruct",
        max_length=32768,
        memory_gb=6.8,
        supports_flash_attention=True,
    ),
    "Qwen2-VL-7B": ModelConfig(
        name="Qwen2-VL-7B",
        hf_name="Qwen/Qwen2-VL-7B-Instruct",
        max_length=4096,
        memory_gb=14.0,  # Larger due to vision components
        supports_flash_attention=True,
    ),
    "gemma-7B": ModelConfig(
        name="gemma-7B",
        hf_name="google/gemma-7b-it",
        max_length=8192,
        memory_gb=14.5,
        supports_flash_attention=True,
    ),
    "gemma2-2B": ModelConfig(
        name="gemma2-2B",
        hf_name="google/gemma-2-2b-it",
        max_length=131072,
        memory_gb=4.5,
        supports_flash_attention=True,
    ),
    "gemma2-7B": ModelConfig(
        name="gemma2-7B",
        hf_name="google/gemma-2-7b-it",
        max_length=131072,
        memory_gb=15.0,  # May exceed RTX 3060 limits
        supports_flash_attention=True,
    ),
    "gemma3-1B": ModelConfig(
        name="gemma3-1B",
        hf_name="google/gemma-3-1b-it",
        max_length=131072,
        memory_gb=2.2,
        supports_flash_attention=True,
    ),
    "gemma3-4B": ModelConfig(
        name="gemma3-4B",
        hf_name="google/gemma-3-4b-it",
        max_length=131072,
        memory_gb=8.5,
        supports_flash_attention=True,
    ),
}


# Dataset configurations
DATASET_CONFIGS: Dict[str, DatasetConfig] = {
    "Wikitext2": DatasetConfig(
        name="Wikitext2",
        hf_dataset=("wikitext", "wikitext-2-raw-v1"),
        split="test",
        text_field="text",
    ),
    "Wikitext103": DatasetConfig(
        name="Wikitext103",
        hf_dataset=("wikitext", "wikitext-103-raw-v1"),
        split="test",
        text_field="text",
    ),
    "Shakespeare": DatasetConfig(
        name="Shakespeare",
        hf_dataset="karpathy/tiny_shakespeare",
        split="train",
        text_field="text",
    ),
    "PTB": DatasetConfig(
        name="PTB",
        hf_dataset=("ptb_text_only", "penn_treebank"),
        split="test",
        text_field="sentence",
    ),
    "C4": DatasetConfig(
        name="C4",
        hf_dataset=("c4", "en"),
        split="validation",  # Use validation split as test is very large
        text_field="text",
        max_samples=1000,  # Limit samples due to size
    ),
}


def get_model_config(model_name: str) -> ModelConfig:
    """Get configuration for a specific model."""
    if model_name not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model: {model_name}. Available models: {list(MODEL_CONFIGS.keys())}"
        )
    return MODEL_CONFIGS[model_name]


def get_dataset_config(dataset_name: str) -> DatasetConfig:
    """Get configuration for a specific dataset."""
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. Available datasets: {list(DATASET_CONFIGS.keys())}"
        )
    return DATASET_CONFIGS[dataset_name]


def get_models_within_memory_limit(memory_limit_gb: float) -> Dict[str, ModelConfig]:
    """Get models that fit within the specified memory limit."""
    return {
        name: config
        for name, config in MODEL_CONFIGS.items()
        if config.memory_gb <= memory_limit_gb
    }


def get_rtx_3060_compatible_models() -> Dict[str, ModelConfig]:
    """Get models compatible with RTX 3060 (12GB VRAM limit)."""
    return get_models_within_memory_limit(12.0)


def get_default_chunk_params() -> list:
    """Get default chunk parameters for evaluation."""
    from ..perplexity.models import ChunkParams

    return [
        ChunkParams(block_size=2048, stride_ratio=0.25, batch_size=1),
        ChunkParams(block_size=2048, stride_ratio=0.5, batch_size=1),
        ChunkParams(block_size=2048, stride_ratio=0.75, batch_size=1),
        ChunkParams(block_size=2048, stride_ratio=1.0, batch_size=1),
        ChunkParams(block_size=4096, stride_ratio=0.25, batch_size=1),
        ChunkParams(block_size=4096, stride_ratio=0.5, batch_size=1),
        ChunkParams(block_size=4096, stride_ratio=0.75, batch_size=1),
        ChunkParams(block_size=4096, stride_ratio=1.0, batch_size=1),
    ]


# Predefined model sets for common use cases
PRESET_MODEL_SETS = {
    "phi_models": ["Phi3-mini-4k", "Phi4-mini", "Phi4-mini-flash"],
    "gpt2_models": ["gpt2", "gpt2-large", "gpt2-xl"],
    "llama_models": ["Llama3.2-1B", "Llama3.2-3B"],
    "gemma_models": ["gemma3-1B", "gemma2-2B", "gemma3-4B"],
    "rtx_3060_safe": [
        "gpt2",
        "gpt2-large",
        "gpt2-xl",
        "Phi3-mini-4k",
        "Phi4-mini",
        "Llama3.2-1B",
        "Llama3.2-3B",
        "gemma3-1B",
        "gemma2-2B",
        "gemma3-4B",
    ],
    "small_models": ["gpt2", "Llama3.2-1B", "gemma3-1B"],
    "flash_attention_models": [
        name
        for name, config in MODEL_CONFIGS.items()
        if config.supports_flash_attention
    ],
}


def get_preset_models(preset_name: str) -> list:
    """Get a preset list of model names."""
    if preset_name not in PRESET_MODEL_SETS:
        raise ValueError(
            f"Unknown preset: {preset_name}. Available presets: {list(PRESET_MODEL_SETS.keys())}"
        )
    return PRESET_MODEL_SETS[preset_name]
