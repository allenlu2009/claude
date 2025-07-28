"""
Pydantic models for perplexity evaluation system.

This module defines the core data models used throughout the perplexity evaluation system,
ensuring type safety and data validation.
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Union
from datetime import datetime


class ModelConfig(BaseModel):
    """Configuration for a language model."""

    name: str = Field(description="Short name identifier for the model")
    hf_name: str = Field(description="HuggingFace model identifier")
    max_length: int = Field(
        gt=0, description="Maximum sequence length supported by the model"
    )
    memory_gb: float = Field(
        gt=0, description="Expected VRAM usage in GB for inference"
    )
    supports_flash_attention: bool = Field(
        default=True, description="Whether model supports flash attention"
    )

    @field_validator("memory_gb")
    @classmethod
    def validate_memory_gb(cls, v: float) -> float:
        if v > 24:  # Reasonable upper bound for consumer GPUs
            raise ValueError("Memory requirement too high for consumer GPUs")
        return v


class ChunkParams(BaseModel):
    """Parameters for text chunking during evaluation."""

    block_size: int = Field(gt=0, description="Size of each chunk in tokens")
    stride_ratio: float = Field(
        ge=0.1, le=2.0, description="Stride as ratio of block_size (0.1-2.0, >1.0 for testing)"
    )
    batch_size: int = Field(default=1, gt=0, description="Batch size for evaluation")

    @property
    def stride(self) -> int:
        """Calculate stride from stride_ratio."""
        return int(self.block_size * self.stride_ratio)

    @field_validator("stride_ratio")
    @classmethod
    def validate_stride_ratio(cls, v: float) -> float:
        if v < 0.1:
            raise ValueError("Stride ratio must be at least 0.1 for meaningful overlap")
        return v


class EvaluationResult(BaseModel):
    """Results from a perplexity evaluation run."""

    model_name: str
    dataset_name: str
    chunk_params: ChunkParams
    avg_nll: float = Field(description="Average negative log-likelihood")
    perplexity: float = Field(gt=0, description="Computed perplexity value")
    num_tokens: int = Field(ge=0, description="Number of tokens evaluated")
    memory_used_mb: float = Field(ge=0, description="Peak memory usage in MB")
    evaluation_time_seconds: float = Field(ge=0, description="Total evaluation time")
    timestamp: datetime = Field(default_factory=datetime.now)

    @field_validator("perplexity")
    @classmethod
    def validate_perplexity(cls, v: float) -> float:
        if v <= 0 and v != float('inf'):
            raise ValueError("Perplexity must be positive")
        # Allow inf for failed evaluations
        if v != float('inf') and v > 10000:  # Sanity check for reasonable perplexity values
            raise ValueError("Perplexity value seems unreasonably high")
        return v


class DatasetConfig(BaseModel):
    """Configuration for dataset loading."""

    name: str = Field(description="Dataset identifier")
    hf_dataset: Union[str, tuple] = Field(
        description="HuggingFace dataset path or (dataset, config) tuple"
    )
    split: str = Field(default="test", description="Dataset split to use")
    text_field: str = Field(default="text", description="Field containing text data")
    max_samples: Optional[int] = Field(
        default=None, description="Maximum number of samples to process"
    )


class PerplexityConfig(BaseModel):
    """Complete configuration for perplexity evaluation."""

    models: List[str] = Field(description="List of model names to evaluate")
    datasets: List[str] = Field(description="List of dataset names to use")
    chunk_params: List[ChunkParams] = Field(
        description="List of chunking parameter sets"
    )
    device: str = Field(default="cuda", description="Device to use for evaluation")
    max_samples: Optional[int] = Field(
        default=None, description="Global limit on samples per dataset"
    )
    output_file: Optional[str] = Field(default=None, description="File to save results")
    use_flash_attention: bool = Field(
        default=True, description="Attempt to use flash attention"
    )
    memory_limit_gb: float = Field(default=12.0, description="VRAM limit in GB")

    @field_validator("device")
    @classmethod
    def validate_device(cls, v: str) -> str:
        if v not in ["cuda", "cpu", "auto"]:
            raise ValueError("Device must be one of: cuda, cpu, auto")
        return v

    @field_validator("models", "datasets")
    @classmethod
    def validate_non_empty_lists(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("List cannot be empty")
        return v


class EvaluationSummary(BaseModel):
    """Summary of multiple evaluation results."""

    results: List[EvaluationResult]
    total_time_seconds: float
    total_models: int
    total_datasets: int
    successful_evaluations: int
    failed_evaluations: int
    average_perplexity: float

    @classmethod
    def from_results(
        cls, results: List[EvaluationResult], total_time: float
    ) -> "EvaluationSummary":
        """Create summary from evaluation results."""
        successful = len(results)
        avg_ppl = (
            sum(r.perplexity for r in results) / successful if successful > 0 else 0.0
        )

        # Extract unique counts
        unique_models = len(set(r.model_name for r in results))
        unique_datasets = len(set(r.dataset_name for r in results))

        return cls(
            results=results,
            total_time_seconds=total_time,
            total_models=unique_models,
            total_datasets=unique_datasets,
            successful_evaluations=successful,
            failed_evaluations=0,  # Will be updated by caller if needed
            average_perplexity=avg_ppl,
        )
