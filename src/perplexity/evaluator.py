"""
Core perplexity evaluation logic with sliding window strategy.

This module implements the main perplexity evaluation functionality using
sliding window approach for better context utilization.
"""

import logging
import math
import time
import warnings
import torch
from typing import List, Tuple
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from .models import ChunkParams, EvaluationResult
from .model_loader import get_memory_usage
from .monitoring import HardwareMonitor

# Set up logging
logger = logging.getLogger(__name__)


class PerplexityEvaluator:
    """Main class for perplexity evaluation using sliding window strategy."""

    def __init__(self, device: str = "cuda"):
        """
        Initialize the evaluator.

        Args:
            device: Device to use for evaluation
        """
        self.device = device

    def tokenize_and_chunk(
        self,
        text: str,
        tokenizer: AutoTokenizer,
        chunk_params: ChunkParams,
        max_length: int,
        handle_residue: bool = True,
    ) -> Tuple[List[torch.Tensor], List[Tuple[int, int, int]]]:
        """
        Tokenize text and create overlapping chunks using sliding window strategy.

        Args:
            text: Input text to tokenize and chunk
            tokenizer: Tokenizer to use
            chunk_params: Chunking parameters
            max_length: Maximum sequence length supported by model
            handle_residue: Whether to include residue tokens as final chunk

        Returns:
            Tuple of (samples, begin_locs) where:
            - samples: List of tokenized chunks
            - begin_locs: List of (begin_loc, end_loc, prev_end_loc) tuples
        """
        logger.info(f"Tokenizing text of length {len(text)} characters")

        # Tokenize the entire text
        tokens = tokenizer(text, return_tensors="pt", truncation=False)["input_ids"]  # type: ignore
        seq_len = tokens.size(1)

        logger.info(f"Tokenized to {seq_len} tokens")

        # Validate token IDs are within vocabulary range to prevent CUDA assertion errors
        vocab_size = len(tokenizer)
        max_token_id = tokens.max().item()
        if max_token_id >= vocab_size:
            num_oob = (tokens >= vocab_size).sum().item()
            logger.warning(
                f"Found {num_oob} out-of-range token IDs (max={max_token_id}, "
                f"vocab_size={vocab_size}). Clamping to vocab range."
            )
            tokens = tokens.clamp(max=vocab_size - 1)

        # Use block size from chunk params (already validated at CLI level)
        block_size = chunk_params.block_size
        stride = int(block_size * chunk_params.stride_ratio)

        if stride > block_size:
            logger.warning(
                f"Stride ({stride}) is larger than block_size ({block_size}). Adjusting stride."
            )
            stride = block_size // 2

        logger.info(
            f"Using block_size={block_size}, stride={stride}, "
            f"stride_ratio={chunk_params.stride_ratio:.2f}"
        )

        samples = []
        begin_locs = []
        prev_end_loc = 0

        # Create sliding window chunks
        for begin_loc in range(0, seq_len, stride):
            end_loc = min(begin_loc + block_size, seq_len)
            chunk = tokens[:, begin_loc:end_loc]

            # Skip chunks that are too short (less than 2 tokens)
            if chunk.size(1) < 2:
                logger.debug(
                    f"Skipping chunk at {begin_loc}-{end_loc}: too short ({chunk.size(1)} tokens)"
                )
                continue

            samples.append(chunk)
            begin_locs.append((begin_loc, end_loc, prev_end_loc))
            prev_end_loc = end_loc

            if end_loc == seq_len:
                break

        # Handle residue tokens (remaining tokens that don't fill a complete chunk)
        if handle_residue and begin_locs:
            last_end = begin_locs[-1][1]  # Get the end position of the last chunk
            if last_end < seq_len:
                # Create a final chunk with residue tokens
                residue_chunk = tokens[:, last_end:seq_len]
                if residue_chunk.size(1) >= 2:  # Only if meaningful residue
                    samples.append(residue_chunk)
                    begin_locs.append((last_end, seq_len, last_end))
                    logger.info(f"Added residue chunk: {residue_chunk.size(1)} tokens")

        logger.info(f"Created {len(samples)} chunks for evaluation (including residue)")
        return samples, begin_locs

    def evaluate_model_on_chunks(
        self,
        samples: List[torch.Tensor],
        begin_locs: List[Tuple[int, int, int]],
        model: AutoModelForCausalLM,
        model_name: str,
        dataset_name: str,
        chunk_params: ChunkParams,
    ) -> EvaluationResult:
        """
        Evaluate model on tokenized chunks and compute perplexity.

        Args:
            samples: List of tokenized chunks
            begin_locs: List of location tuples for each chunk
            model: Model to evaluate
            model_name: Name of the model
            dataset_name: Name of the dataset
            chunk_params: Chunking parameters used

        Returns:
            EvaluationResult with computed metrics
        """
        logger.info(f"Evaluating model on {len(samples)} chunks")

        start_time = time.time()
        initial_memory = get_memory_usage()

        # Initialize and start hardware monitor
        hw_monitor = HardwareMonitor()
        hw_monitor.start()

        nll_sum = 0.0
        n_tokens = 0
        losses = []

        # Evaluate each chunk with progress bar
        for i, (input_ids, (begin_loc, end_loc, prev_end_loc)) in enumerate(
            tqdm(zip(samples, begin_locs), total=len(samples), desc="Evaluating chunks")
        ):
            try:
                # Move to device and ensure proper dtype
                input_ids = input_ids.to(device=self.device, dtype=torch.long)

                # Skip if sequence is too short
                if input_ids.size(1) < 2:
                    logger.debug(f"Chunk {i+1}: Skipped (too short)")
                    continue

                # Calculate target length for sliding window
                trg_len = end_loc - prev_end_loc
                target_ids = input_ids.clone()

                # CRITICAL: Mask previous context tokens for sliding window
                # Only compute loss on new tokens, not the overlapping context
                if trg_len < input_ids.size(1):
                    target_ids[:, :-trg_len] = -100

                # Forward pass with no gradient computation
                with torch.no_grad():
                    # Suppress transformer warnings about unrecognized loss types
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", message=".*loss_type.*was set in the config.*")
                        outputs = model(input_ids, labels=target_ids)  # type: ignore
                    neg_log_likelihood = outputs.loss

                # Count valid tokens (not masked)
                num_valid_tokens = (target_ids != -100).sum().item()
                batch_size = target_ids.size(0)
                num_loss_tokens = (
                    num_valid_tokens - batch_size
                )  # Subtract special tokens

                if num_loss_tokens > 0:
                    nll_sum += neg_log_likelihood.item() * num_loss_tokens
                    n_tokens += num_loss_tokens

                losses.append(neg_log_likelihood.item())

                logger.debug(
                    f"Chunk {i+1}: Loss = {neg_log_likelihood.item():.4f}, "
                    f"Tokens = {num_loss_tokens}"
                )
                
                # Explicitly delete GPU tensors at end of iteration
                del input_ids
                del target_ids
                del outputs
                del neg_log_likelihood

            except torch.cuda.OutOfMemoryError as e:
                logger.error(
                    f"CUDA out of memory with block_size={chunk_params.block_size}. "
                    f"Try reducing block size or use --device cpu"
                )
                logger.error(f"Memory error details: {str(e)}")
                raise RuntimeError(
                    f"Insufficient GPU memory for block_size {chunk_params.block_size}. "
                    f"Reduce block_size or use --device cpu"
                )
            except Exception as e:
                logger.error(f"Chunk {i+1}: Error - {str(e)}")
                logger.error(f"Input shape: {input_ids.shape}")
                continue

        # Stop hardware monitor and collect stats
        hw_stats = hw_monitor.stop()

        # Calculate final metrics
        avg_nll = nll_sum / n_tokens if n_tokens > 0 else float("inf")

        # Compute perplexity from average negative log-likelihood
        if n_tokens > 0 and not math.isinf(avg_nll):
            perplexity = math.exp(avg_nll)
        else:
            perplexity = float("inf")

        # Calculate evaluation time and memory usage
        evaluation_time = time.time() - start_time
        final_memory = get_memory_usage()

        memory_used_mb = 0.0
        if initial_memory.get("cuda_available") and final_memory.get("cuda_available"):
            memory_used_mb = final_memory["max_allocated_mb"] - initial_memory.get(
                "max_allocated_mb", 0
            )

        logger.info(
            f"Evaluation complete: avg_nll={avg_nll:.4f}, "
            f"perplexity={perplexity:.4f}, tokens={n_tokens}"
        )

        return EvaluationResult(
            model_name=model_name,
            dataset_name=dataset_name,
            chunk_params=chunk_params,
            avg_nll=round(float(avg_nll), 4),
            perplexity=round(perplexity, 4) if not math.isinf(perplexity) else float("inf"),
            num_tokens=n_tokens,
            memory_used_mb=float(round(memory_used_mb)),
            evaluation_time_seconds=float(round(evaluation_time)),
            avg_power_draw_mw=float(round(hw_stats.get("avg_power_draw_mw"))) if hw_stats.get("avg_power_draw_mw") is not None else None,
            max_power_draw_mw=float(round(hw_stats.get("max_power_draw_mw"))) if hw_stats.get("max_power_draw_mw") is not None else None,
            avg_gpu_utilization=float(round(hw_stats.get("avg_gpu_utilization"))) if hw_stats.get("avg_gpu_utilization") is not None else None,
            max_gpu_utilization=float(round(hw_stats.get("max_gpu_utilization"))) if hw_stats.get("max_gpu_utilization") is not None else None,
            avg_memory_allocated_mb=float(round(hw_stats.get("avg_memory_allocated_mb"))) if hw_stats.get("avg_memory_allocated_mb") is not None else None,
            max_memory_allocated_mb=float(round(hw_stats.get("max_memory_allocated_mb"))) if hw_stats.get("max_memory_allocated_mb") is not None else None,
            avg_memory_reserved_mb=float(round(hw_stats.get("avg_memory_reserved_mb"))) if hw_stats.get("avg_memory_reserved_mb") is not None else None,
            max_memory_reserved_mb=float(round(hw_stats.get("max_memory_reserved_mb"))) if hw_stats.get("max_memory_reserved_mb") is not None else None,
            avg_memory_used_system_mb=float(round(hw_stats.get("avg_memory_used_system_mb"))) if hw_stats.get("avg_memory_used_system_mb") is not None else None,
            max_memory_used_system_mb=float(round(hw_stats.get("max_memory_used_system_mb"))) if hw_stats.get("max_memory_used_system_mb") is not None else None,
            num_hw_samples=hw_stats.get("num_hw_samples", 0),
        )

    def evaluate_text(
        self,
        text: str,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        model_name: str,
        dataset_name: str,
        chunk_params: ChunkParams,
        max_length: int,
        handle_residue: bool = True,
    ) -> EvaluationResult:
        """
        Complete evaluation pipeline for a text string.

        Args:
            text: Text to evaluate
            model: Model to use for evaluation
            tokenizer: Tokenizer to use
            model_name: Name of the model
            dataset_name: Name of the dataset
            chunk_params: Chunking parameters
            max_length: Maximum sequence length
            handle_residue: Whether to include residue tokens

        Returns:
            EvaluationResult with all metrics
        """
        logger.info(f"Starting evaluation: {model_name} on {dataset_name}")

        # Tokenize and chunk the text
        samples, begin_locs = self.tokenize_and_chunk(
            text, tokenizer, chunk_params, max_length, handle_residue
        )

        if not samples:
            logger.error("No valid samples generated from text")
            return EvaluationResult(
                model_name=model_name,
                dataset_name=dataset_name,
                chunk_params=chunk_params,
                avg_nll=float("inf"),
                perplexity=float("inf"),
                num_tokens=0,
                memory_used_mb=0.0,
                evaluation_time_seconds=0.0,
            )

        # Evaluate the model on chunks
        result = self.evaluate_model_on_chunks(
            samples, begin_locs, model, model_name, dataset_name, chunk_params
        )

        return result

    def compare_stride_ratios(
        self,
        text: str,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        model_name: str,
        dataset_name: str,
        block_size: int,
        stride_ratios: List[float],
        max_length: int,
    ) -> List[EvaluationResult]:
        """
        Compare different stride ratios for the same model and dataset.

        Args:
            text: Text to evaluate
            model: Model to use
            tokenizer: Tokenizer to use
            model_name: Model name
            dataset_name: Dataset name
            block_size: Block size to use
            stride_ratios: List of stride ratios to compare
            max_length: Maximum sequence length

        Returns:
            List of EvaluationResult for each stride ratio
        """
        results = []

        for stride_ratio in stride_ratios:
            chunk_params = ChunkParams(
                block_size=block_size, stride_ratio=stride_ratio, batch_size=1
            )

            logger.info(f"Evaluating with stride_ratio={stride_ratio}")

            result = self.evaluate_text(
                text,
                model,
                tokenizer,
                model_name,
                dataset_name,
                chunk_params,
                max_length,
            )

            results.append(result)

        return results


def validate_sliding_window_benefit(results: List[EvaluationResult]) -> bool:
    """
    Validate that sliding window (lower stride ratios) gives better perplexity.

    Args:
        results: List of evaluation results with different stride ratios

    Returns:
        True if sliding window shows expected benefit
    """
    if len(results) < 2:
        return True  # Can't compare

    # Sort by stride ratio
    sorted_results = sorted(results, key=lambda r: r.chunk_params.stride_ratio)

    # Check if perplexity generally decreases with smaller stride ratios
    improvements = 0
    comparisons = 0

    for i in range(len(sorted_results) - 1):
        if sorted_results[i].perplexity < sorted_results[i + 1].perplexity:
            improvements += 1
        comparisons += 1

    # At least 50% of comparisons should show improvement
    improvement_rate = improvements / comparisons if comparisons > 0 else 0

    logger.info(f"Sliding window improvement rate: {improvement_rate:.2%}")
    return improvement_rate >= 0.5
