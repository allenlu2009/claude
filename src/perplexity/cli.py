"""
Command-line interface for perplexity evaluation system.

This module provides a comprehensive CLI for running perplexity evaluations
with support for multiple models, datasets, and configuration options.
"""

import argparse
import json
import logging
import sys
import time
from typing import List

from .models import ChunkParams, PerplexityConfig, EvaluationSummary
from .evaluator import PerplexityEvaluator
from .model_loader import load_model_and_tokenizer, unload_model, get_device_info
from .dataset_utils import load_dataset_text
from ..config.model_configs import (
    get_model_config,
    get_rtx_3060_compatible_models,
    get_preset_models,
    MODEL_CONFIGS,
    DATASET_CONFIGS,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.getLogger().setLevel(level)

    # Set specific loggers
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.WARNING)


def create_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Perplexity evaluation system for language models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate Phi3 on WikiText-2
  python -m src.perplexity.cli evaluate --models Phi3-mini-4k --datasets Wikitext2
  
  # Evaluate multiple models with custom chunk parameters
  python -m src.perplexity.cli evaluate --models Phi3-mini-4k Phi4-mini --datasets Wikitext2 PTB --block-size 4096 --stride-ratios 0.25 0.5
  
  # Use a preset model set
  python -m src.perplexity.cli evaluate --preset phi_models --datasets Wikitext2
  
  # List available models and datasets
  python -m src.perplexity.cli list
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Run perplexity evaluation")
    eval_parser.add_argument(
        "--models",
        nargs="+",
        help="Model names to evaluate (use --list to see available models)",
    )
    eval_parser.add_argument(
        "--preset",
        help="Use a preset model set (phi_models, gpt2_models, rtx_3060_safe, etc.)",
    )
    eval_parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help="Dataset names to use for evaluation",
    )
    eval_parser.add_argument(
        "--block-size",
        type=int,
        default=4096,
        help="Block size for chunking (default: 4096)",
    )
    eval_parser.add_argument(
        "--stride-ratios",
        nargs="+",
        type=float,
        default=[0.25, 0.5, 0.75, 1.0],
        help="Stride ratios to test, range 0.1-1.0 (default: 0.25 0.5 0.75 1.0)",
    )
    eval_parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use (default: auto)",
    )
    eval_parser.add_argument(
        "--memory-limit",
        type=float,
        default=12.0,
        help="Memory limit in GB (default: 12.0 for RTX 3060)",
    )
    eval_parser.add_argument(
        "--max-samples", type=int, help="Maximum number of samples per dataset"
    )
    eval_parser.add_argument(
        "--max-tokens", type=int, help="Maximum number of tokens per dataset"
    )
    eval_parser.add_argument(
        "--use-all-tokens", action="store_true", 
        help="Use all available tokens from dataset (ignores max-samples)"
    )
    eval_parser.add_argument(
        "--no-residue", action="store_true",
        help="Skip residue tokens that don't fill complete chunks"
    )
    eval_parser.add_argument(
        "--output", type=str, help="Output file for results (JSON format)"
    )
    eval_parser.add_argument(
        "--config", type=str, help="Load configuration from JSON file"
    )
    eval_parser.add_argument(
        "--no-flash-attention",
        action="store_true",
        help="Disable flash attention (use eager implementation)",
    )

    # List command
    list_parser = subparsers.add_parser(
        "list", help="List available models and datasets"
    )
    list_parser.add_argument(
        "--models", action="store_true", help="List available models"
    )
    list_parser.add_argument(
        "--datasets", action="store_true", help="List available datasets"
    )
    list_parser.add_argument(
        "--presets", action="store_true", help="List available model presets"
    )
    list_parser.add_argument(
        "--rtx-3060", action="store_true", help="List RTX 3060 compatible models"
    )

    # Info command
    subparsers.add_parser("info", help="Show system information")

    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare stride ratios")
    compare_parser.add_argument("--model", required=True, help="Model to evaluate")
    compare_parser.add_argument("--dataset", required=True, help="Dataset to use")
    compare_parser.add_argument(
        "--block-size", type=int, default=2048, help="Block size"
    )
    compare_parser.add_argument(
        "--stride-ratios",
        nargs="+",
        type=float,
        default=[0.1, 0.25, 0.5, 0.75, 1.0],
        help="Stride ratios to compare",
    )

    # Global options
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--quiet", "-q", action="store_true", help="Quiet output")

    return parser


def load_config_file(config_path: str) -> PerplexityConfig:
    """Load configuration from JSON file."""
    try:
        with open(config_path, "r") as f:
            config_data = json.load(f)
        return PerplexityConfig(**config_data)
    except Exception as e:
        logger.error(f"Failed to load config file {config_path}: {e}")
        sys.exit(1)


def create_chunk_params(
    block_size: int, stride_ratios: List[float]
) -> List[ChunkParams]:
    """Create chunk parameters from block size and stride ratios."""
    return [
        ChunkParams(block_size=block_size, stride_ratio=ratio, batch_size=1)
        for ratio in stride_ratios
    ]


def print_device_info() -> None:
    """Print device information."""
    info = get_device_info()
    print("\n=== Device Information ===")
    print(f"CUDA Available: {info['cuda_available']}")
    if info["cuda_available"]:
        print(f"GPU: {info['gpu_name']}")
        print(f"Total Memory: {info['total_memory_gb']:.1f} GB")
        print(f"Compute Capability: {info['compute_capability']}")
        print(
            f"Current Usage: {info['allocated_memory_mb']:.1f} MB allocated, "
            f"{info['cached_memory_mb']:.1f} MB cached"
        )


def list_models() -> None:
    """List available models."""
    print("\n=== Available Models ===")
    rtx_compatible = get_rtx_3060_compatible_models()

    for name, config in MODEL_CONFIGS.items():
        status = "✓ RTX 3060" if name in rtx_compatible else "  High Memory"
        flash = "Flash" if config.supports_flash_attention else "Eager"
        print(
            f"{status} {name:20} {config.memory_gb:5.1f}GB  {flash:5}  {config.hf_name}"
        )


def list_datasets() -> None:
    """List available datasets."""
    print("\n=== Available Datasets ===")
    for name, config in DATASET_CONFIGS.items():
        print(f"{name:15} {config.split:10} {config.hf_dataset}")


def list_presets() -> None:
    """List available model presets."""
    from ..config.model_configs import PRESET_MODEL_SETS

    print("\n=== Available Model Presets ===")
    for preset, models in PRESET_MODEL_SETS.items():
        print(f"{preset:20} {', '.join(models[:3])}{'...' if len(models) > 3 else ''}")


def run_evaluation(args: argparse.Namespace) -> EvaluationSummary:
    """Run the main evaluation process."""
    # Load configuration
    if args.config:
        config = load_config_file(args.config)
    else:
        # Build configuration from arguments
        if args.models:
            models = args.models
        elif args.preset:
            models = get_preset_models(args.preset)
        else:
            models = []
            
        if not models:
            logger.error("No models specified. Use --models or --preset")
            sys.exit(1)

        chunk_params = create_chunk_params(args.block_size, args.stride_ratios)

        config = PerplexityConfig(
            models=models,
            datasets=args.datasets,
            chunk_params=chunk_params,
            device=args.device,
            max_samples=args.max_samples,
            max_tokens=args.max_tokens,
            use_all_tokens=args.use_all_tokens,
            handle_residue=not args.no_residue,
            output_file=args.output,
            use_flash_attention=not args.no_flash_attention,
            memory_limit_gb=args.memory_limit,
        )

    # Initialize evaluator
    evaluator = PerplexityEvaluator(device=config.device)
    all_results = []
    start_time = time.time()

    logger.info(
        f"Starting evaluation with {len(config.models)} models and {len(config.datasets)} datasets"
    )

    # Run evaluations
    for model_name in config.models:
        try:
            logger.info(f"\n{'='*50}")
            logger.info(f"Loading model: {model_name}")

            # Get model configuration
            model_config = get_model_config(model_name)

            # Check memory constraints
            if model_config.memory_gb > config.memory_limit_gb:
                logger.warning(
                    f"Skipping {model_name}: requires {model_config.memory_gb}GB "
                    f"but limit is {config.memory_limit_gb}GB"
                )
                continue

            # Load model and tokenizer
            model, tokenizer = load_model_and_tokenizer(
                model_name=model_name,
                device=config.device,
                use_flash_attention=config.use_flash_attention,
                memory_limit_gb=config.memory_limit_gb,
            )

            # Evaluate on each dataset
            for dataset_name in config.datasets:
                logger.info(f"\nEvaluating on dataset: {dataset_name}")

                # Load dataset text with token-based options
                text = load_dataset_text(
                    dataset_name, 
                    config.max_samples,
                    config.max_tokens,
                    config.use_all_tokens
                )

                # Evaluate with each chunk parameter set
                for chunk_param in config.chunk_params:
                    logger.info(
                        f"Chunk params: block_size={chunk_param.block_size}, "
                        f"stride_ratio={chunk_param.stride_ratio}"
                    )

                    result = evaluator.evaluate_text(
                        text=text,
                        model=model,
                        tokenizer=tokenizer,
                        model_name=model_name,
                        dataset_name=dataset_name,
                        chunk_params=chunk_param,
                        max_length=model_config.max_length,
                        handle_residue=config.handle_residue,
                    )

                    all_results.append(result)

                    logger.info(
                        f"Result: perplexity={result.perplexity:.4f}, "
                        f"avg_nll={result.avg_nll:.4f}, "
                        f"tokens={result.num_tokens}"
                    )

            # Clean up model to free memory
            unload_model(model)

        except Exception as e:
            logger.error(f"Error evaluating model {model_name}: {e}")
            continue

    # Create summary
    total_time = time.time() - start_time
    summary = EvaluationSummary.from_results(all_results, total_time)

    # Save results if requested
    if config.output_file:
        save_results(summary, config.output_file)

    return summary


def save_results(summary: EvaluationSummary, output_file: str) -> None:
    """Save evaluation results to file."""
    try:
        with open(output_file, "w") as f:
            json.dump(summary.model_dump(), f, indent=2, default=str)
        logger.info(f"Results saved to {output_file}")
    except Exception as e:
        logger.error(f"Failed to save results: {e}")


def print_summary(summary: EvaluationSummary) -> None:
    """Print evaluation summary."""
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total Models:      {summary.total_models}")
    print(f"Total Datasets:    {summary.total_datasets}")
    print(f"Successful Evals:  {summary.successful_evaluations}")
    print(f"Failed Evals:      {summary.failed_evaluations}")
    print(f"Average Perplexity: {summary.average_perplexity:.4f}")
    print(f"Total Time:        {summary.total_time_seconds:.1f}s")

    if summary.results:
        print(
            f"\n{'Model':<20} {'Dataset':<15} {'Block':<8} {'Stride':<8} {'Perplexity':<12} {'Tokens':<10}"
        )
        print("-" * 80)
        for result in summary.results:
            print(
                f"{result.model_name:<20} "
                f"{result.dataset_name:<15} "
                f"{result.chunk_params.block_size:<8} "
                f"{result.chunk_params.stride_ratio:<8.2f} "
                f"{result.perplexity:<12.4f} "
                f"{result.num_tokens:<10}"
            )


def main() -> None:
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()

    # Setup logging
    if args.quiet:
        logging.getLogger().setLevel(logging.ERROR)
    else:
        setup_logging(args.verbose)

    if args.command == "list":
        if args.models or not any([args.datasets, args.presets, args.rtx_3060]):
            list_models()
        if args.datasets:
            list_datasets()
        if args.presets:
            list_presets()
        if args.rtx_3060:
            print("\n=== RTX 3060 Compatible Models ===")
            compatible = get_rtx_3060_compatible_models()
            for name, config in compatible.items():
                print(f"{name:20} {config.memory_gb:.1f}GB")

    elif args.command == "info":
        print_device_info()

    elif args.command == "evaluate":
        summary = run_evaluation(args)
        print_summary(summary)

    elif args.command == "compare":
        # TODO: Implement stride ratio comparison
        logger.error("Compare command not yet implemented")
        sys.exit(1)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
