name: "Model Perplexity Evaluation System"
description: |

## Purpose
A comprehensive perplexity evaluation system for comparing language models (Phi3, Phi4, etc.) using standard datasets like WikiText-2, optimized for memory-constrained GPUs.

## Core Principles
1. **Context is King**: Include ALL necessary documentation, examples, and caveats
2. **Validation Loops**: Provide executable tests/lints the AI can run and fix
3. **Information Dense**: Use keywords and patterns from the codebase
4. **Progressive Success**: Start simple, validate, then enhance
5. **Global rules**: Be sure to follow all rules in CLAUDE.md

---

## Goal
Create a production-ready perplexity evaluation system that computes perplexity for common public models (Phi3, Phi4, GPT-2, Llama, Gemma, Qwen) using standard datasets (WikiText-2, PTB, Shakespeare), optimized for RTX 3060 GPU memory constraints with flash attention support.

## Why
- **Research Value**: Provides standardized perplexity benchmarks for model comparison
- **Memory Optimization**: Demonstrates efficient GPU memory usage for consumer hardware
- **Integration with Existing**: Builds upon the existing perplexity_llm_claude.py example
- **Best Practices**: Implements current 2025 perplexity evaluation methodologies

## What
A modular Python system that:
- Evaluates perplexity using sliding window strategy (not disjoint chunks)
- Supports multiple models with automatic memory optimization
- Uses flash attention when available for memory efficiency
- Provides comprehensive logging and results storage
- Handles VRAM constraints gracefully with fallback strategies

### Success Criteria
- [ ] Successfully evaluates Phi3-mini-4k and Phi4-mini models on WikiText-2
- [ ] Memory usage stays within RTX 3060 limits (8-12GB VRAM)
- [ ] Implements sliding window strategy with configurable stride ratios
- [ ] Flash attention enabled automatically when supported
- [ ] Results comparable to reference implementations
- [ ] Comprehensive unit tests with >80% coverage
- [ ] CLI interface for easy execution

## All Needed Context

### Documentation & References (list all context needed to implement the feature)
```yaml
# MUST READ - Include these in your context window
- url: https://huggingface.co/docs/transformers/en/perplexity
  why: Official HuggingFace perplexity evaluation documentation with sliding window strategy
  critical: Explains why sliding window is better than disjoint chunks
  
- url: https://github.com/Dao-AILab/flash-attention
  why: Flash Attention implementation details and memory optimization
  section: README.md installation and usage patterns
  critical: RTX 3060 compatibility requirements and fallback strategies

- url: https://huggingface.co/microsoft/Phi-3-mini-4k-instruct
  why: Phi3 model specifications and memory requirements
  critical: 7.1-8.5 GB VRAM for inference, flash attention compatibility

- url: https://huggingface.co/microsoft/Phi-4-mini-instruct  
  why: Phi4 model specifications and flash attention requirements
  critical: 7.7GB VRAM requirement, requires flash attention or eager fallback

- file: examples/perplexity_llm_claude.py
  why: Existing implementation patterns for model loading, tokenization, and evaluation
  critical: ChunkParam dataclass, evaluate_model function, memory management patterns

- url: https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/wikitext/README.md
  why: Standard evaluation harness patterns for WikiText dataset
  critical: Proper tokenization and evaluation methodology
```

### Current Codebase tree (run `tree` in the root of the project) to get an overview of the codebase
```bash
.
├── CLAUDE.md
├── INITIAL.md
├── PRPs/
│   ├── templates/
│   │   └── prp_base.md
│   └── EXAMPLE_multi_agent_prp.md
├── examples/
│   └── perplexity_llm_claude.py
└── use-cases/
    ├── mcp-server/
    ├── pydantic-ai/
    └── template-generator/
```

### Desired Codebase tree with files to be added and responsibility of file
```bash
src/
├── perplexity/
│   ├── __init__.py           # Package initialization
│   ├── models.py             # Pydantic models for configuration and results
│   ├── evaluator.py          # Core perplexity evaluation logic
│   ├── model_loader.py       # Model and tokenizer loading with memory optimization
│   ├── dataset_utils.py      # Dataset loading and preprocessing utilities
│   └── cli.py               # Command-line interface
├── config/
│   └── model_configs.py     # Model-specific configurations and memory limits
tests/
├── __init__.py
├── test_evaluator.py        # Unit tests for core evaluation logic
├── test_model_loader.py     # Tests for model loading and memory management
├── test_dataset_utils.py    # Tests for dataset utilities
└── conftest.py             # Pytest configuration and fixtures
requirements.txt            # Python dependencies
README.md                   # Documentation update
```

### Known Gotchas of our codebase & Library Quirks
```python
# CRITICAL: Flash Attention 2 requires specific GPU architecture
# RTX 3060 supports flash attention but may need fallback to eager
# Example: model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="flash_attention_2")
# If fails: model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager")

# CRITICAL: Phi models require trust_remote_code=True
# Example: tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# CRITICAL: Memory management - models must be explicitly deleted and CUDA cache cleared
# Pattern from examples/perplexity_llm_claude.py:
# del model
# torch.cuda.empty_cache()

# CRITICAL: Use venv_linux virtual environment for all Python execution
# As specified in CLAUDE.md

# CRITICAL: Sliding window strategy requires careful stride calculation
# stride = int(block_size * stride_ratio) 
# Smaller stride = more context but slower computation

# CRITICAL: Target masking for proper perplexity calculation
# target_ids[:, :-trg_len] = -100  # Mask previous context for sliding window
```

## Implementation Blueprint

### Data models and structure

Create the core data models, we ensure type safety and consistency.
```python
from pydantic import BaseModel, Field
from typing import List, Optional, Union
from dataclasses import dataclass

class ModelConfig(BaseModel):
    name: str
    hf_name: str
    max_length: int
    memory_gb: float
    supports_flash_attention: bool = True

class ChunkParams(BaseModel):
    block_size: int = Field(gt=0, description="Size of each chunk")
    stride_ratio: float = Field(ge=0.1, le=1.0, description="Stride as ratio of block_size")
    batch_size: int = Field(default=1, gt=0)

class EvaluationResult(BaseModel):
    model_name: str
    dataset_name: str
    chunk_params: ChunkParams
    avg_nll: float
    perplexity: float
    num_tokens: int
    memory_used_mb: float
    evaluation_time_seconds: float

class PerplexityConfig(BaseModel):
    models: List[str]
    datasets: List[str] 
    chunk_params: List[ChunkParams]
    device: str = "cuda"
    max_samples: Optional[int] = None
    output_file: Optional[str] = None
```

### list of tasks to be completed to fullfill the PRP in the order they should be completed

```yaml
Task 1: Setup project structure and dependencies
CREATE src/perplexity/__init__.py:
  - EMPTY file for package initialization

CREATE requirements.txt:
  - MIRROR dependencies from examples/perplexity_llm_claude.py
  - ADD torch, transformers, datasets, flash-attn, pydantic, pytest
  - PRESERVE version compatibility for RTX 3060

Task 2: Create configuration system
CREATE src/perplexity/models.py:
  - MIRROR pattern from examples/perplexity_llm_claude.py model_args structure
  - IMPLEMENT Pydantic models for type safety
  - ADD memory optimization parameters

CREATE src/config/model_configs.py:
  - EXTRACT model configurations from examples/perplexity_llm_claude.py
  - ADD memory requirements based on research findings
  - INCLUDE flash attention compatibility flags

Task 3: Implement dataset utilities
CREATE src/perplexity/dataset_utils.py:
  - MIRROR dataset2text function from examples/perplexity_llm_claude.py
  - REFACTOR for modularity and error handling
  - ADD support for additional datasets

Task 4: Create model loading system with memory optimization
CREATE src/perplexity/model_loader.py:
  - MIRROR load_model_and_tokenizer from examples/perplexity_llm_claude.py
  - ADD memory constraint checking for RTX 3060
  - IMPLEMENT flash attention fallback strategy
  - ADD gradient checkpointing for memory efficiency

Task 5: Implement core evaluation logic
CREATE src/perplexity/evaluator.py:
  - MIRROR tokenization_and_chunk and evaluate_model functions
  - IMPLEMENT sliding window strategy with configurable stride
  - ADD comprehensive error handling and logging
  - OPTIMIZE for memory efficiency

Task 6: Create CLI interface
CREATE src/perplexity/cli.py:
  - IMPLEMENT argparse-based CLI
  - SUPPORT configuration file input
  - ADD progress reporting and result saving
  - MIRROR execution pattern from examples/perplexity_llm_claude.py

Task 7: Comprehensive testing
CREATE tests/test_evaluator.py:
  - TEST sliding window implementation
  - TEST memory constraint handling
  - TEST perplexity calculation accuracy

CREATE tests/test_model_loader.py:
  - TEST model loading with flash attention
  - TEST memory optimization
  - TEST error handling for unsupported models

CREATE tests/conftest.py:
  - SETUP pytest fixtures for models and datasets
  - MOCK GPU memory for testing
```

### Per task pseudocode as needed added to each task

```python
# Task 5: Core Evaluation Logic
class PerplexityEvaluator:
    def __init__(self, device: str = "cuda"):
        self.device = device
        
    def tokenize_and_chunk(self, text: str, tokenizer, chunk_params: ChunkParams, max_length: int):
        # PATTERN: Mirror examples/perplexity_llm_claude.py:182-206
        tokens = tokenizer(text, return_tensors="pt", truncation=False)["input_ids"]
        block_size = min(chunk_params.block_size, max_length)
        stride = int(chunk_params.stride_ratio * block_size)  # CRITICAL: stride calculation
        
        samples, begin_locs = [], []
        for begin_loc in range(0, tokens.size(1), stride):
            # SLIDING WINDOW: overlapping chunks for better context
            end_loc = min(begin_loc + block_size, tokens.size(1))
            chunk = tokens[:, begin_loc:end_loc]
            if chunk.size(1) >= 2:  # Skip too-short chunks
                samples.append(chunk)
                begin_locs.append((begin_loc, end_loc, prev_end_loc))
        return samples, begin_locs

    def evaluate_model(self, samples, begin_locs, model) -> EvaluationResult:
        # PATTERN: Mirror examples/perplexity_llm_claude.py:208-253
        nll_sum, n_tokens = 0.0, 0
        
        for input_ids, (begin_loc, end_loc, prev_end_loc) in zip(samples, begin_locs):
            input_ids = input_ids.to(device=self.device, dtype=torch.long)
            
            # CRITICAL: Target masking for sliding window
            trg_len = end_loc - prev_end_loc
            target_ids = input_ids.clone()
            if trg_len < input_ids.size(1):
                target_ids[:, :-trg_len] = -100  # Mask previous context
            
            # MEMORY: Use no_grad to save memory
            with torch.no_grad():
                outputs = model(input_ids, labels=target_ids)
                neg_log_likelihood = outputs.loss
                
            num_valid_tokens = (target_ids != -100).sum().item()
            num_loss_tokens = num_valid_tokens - target_ids.size(0)
            
            if num_loss_tokens > 0:
                nll_sum += neg_log_likelihood * num_loss_tokens
                n_tokens += num_loss_tokens
        
        avg_nll = nll_sum / n_tokens if n_tokens > 0 else float('inf')
        perplexity = torch.exp(avg_nll.clone().detach()).item()
        return EvaluationResult(avg_nll=avg_nll, perplexity=perplexity, num_tokens=n_tokens)
```

### Integration Points
```yaml
MEMORY MANAGEMENT:
  - pattern: "del model; torch.cuda.empty_cache()" after each model evaluation
  - monitoring: torch.cuda.memory_allocated() before/after operations
  
FLASH ATTENTION:
  - primary: attn_implementation="flash_attention_2"
  - fallback: attn_implementation="eager" if flash fails
  - check: RTX 3060 Ampere architecture support
  
DATASETS:
  - mirror: dataset_args from examples/perplexity_llm_claude.py
  - extend: add validation and error handling
  
CLI:
  - pattern: argparse with subcommands (evaluate, benchmark, compare)
  - config: support JSON/YAML configuration files
```

## Validation Loop

### Level 1: Syntax & Style
```bash
# Run these FIRST - fix any errors before proceeding
ruff check src/ --fix  # Auto-fix what's possible
mypy src/             # Type checking
black src/            # Code formatting

# Expected: No errors. If errors, READ the error and fix.
```

### Level 2: Unit Tests each new feature/file/function use existing test patterns
```python
# CREATE comprehensive test suite:
def test_sliding_window_implementation():
    """Test that sliding window gives better perplexity than disjoint chunks"""
    # Use small model and short text for testing
    evaluator = PerplexityEvaluator()
    
    # Test with stride_ratio=1.0 (disjoint) vs stride_ratio=0.5 (overlapping)
    disjoint_ppl = evaluator.evaluate_text(text, stride_ratio=1.0)
    sliding_ppl = evaluator.evaluate_text(text, stride_ratio=0.5)
    
    # Sliding window should give lower (better) perplexity
    assert sliding_ppl < disjoint_ppl

def test_memory_constraint_handling():
    """Test that evaluation respects memory limits"""
    # Mock GPU memory to simulate RTX 3060 constraints
    with mock.patch('torch.cuda.get_device_properties') as mock_gpu:
        mock_gpu.return_value.total_memory = 12 * 1024**3  # 12GB
        
        evaluator = PerplexityEvaluator()
        # Should not exceed memory limits
        result = evaluator.evaluate_large_model()
        assert result is not None  # Should complete without OOM

def test_flash_attention_fallback():
    """Test graceful fallback when flash attention fails"""
    with mock.patch('transformers.AutoModelForCausalLM.from_pretrained') as mock_load:
        # First call fails with flash attention
        mock_load.side_effect = [RuntimeError("Flash attention not supported"), MagicMock()]
        
        loader = ModelLoader()
        model, tokenizer = loader.load_model("test-model")
        
        # Should have called twice (flash, then eager)
        assert mock_load.call_count == 2
        assert mock_load.call_args_list[1][1]['attn_implementation'] == 'eager'
```

```bash
# Run and iterate until passing:
uv run pytest tests/ -v --cov=src/perplexity --cov-report=html
# Target: >80% coverage, all tests passing
```

### Level 3: Integration Test
```bash
# Test with actual model evaluation
uv run python -m src.perplexity.cli evaluate \
  --models Phi3-mini-4k \
  --datasets Wikitext2 \
  --config config/test_config.json

# Expected: Successfully evaluates without OOM, produces reasonable perplexity values
# For Phi3-mini on WikiText-2: expect perplexity in range 10-30 based on literature
```

## Final validation Checklist
- [ ] All tests pass: `uv run pytest tests/ -v`
- [ ] No linting errors: `uv run ruff check src/`
- [ ] No type errors: `uv run mypy src/`
- [ ] Manual test successful: CLI evaluation completes
- [ ] Memory constraints respected: VRAM usage < 12GB
- [ ] Flash attention works with fallback: No runtime errors
- [ ] Results reasonable: Perplexity values match expected ranges
- [ ] Documentation complete: README updated with usage examples

---

## Anti-Patterns to Avoid
- ❌ Don't use disjoint chunks - always use sliding window strategy
- ❌ Don't ignore memory constraints - check VRAM before loading models
- ❌ Don't hardcode model configurations - use configurable system
- ❌ Don't skip flash attention fallback - RTX 3060 may have compatibility issues
- ❌ Don't forget to clear CUDA cache between models
- ❌ Don't use synchronous operations where async would improve performance
- ❌ Don't ignore tokenization differences between models

## Quality Score: 9/10

**Confidence level for one-pass implementation success**: This PRP includes comprehensive context, follows existing patterns from the codebase, incorporates current best practices from research, and provides detailed validation gates. The score reflects high confidence due to:

✅ **Complete Context**: All necessary documentation and implementation details included
✅ **Existing Patterns**: Builds directly on working example in codebase
✅ **Research-Based**: Incorporates 2025 best practices for perplexity evaluation
✅ **Memory Optimized**: Specifically addresses RTX 3060 constraints
✅ **Validation Gates**: Comprehensive testing strategy with specific success criteria
✅ **Error Handling**: Includes flash attention fallback and memory management
✅ **Modular Design**: Clear separation of concerns following CLAUDE.md guidelines

**Minor Risk**: Flash attention compatibility may require testing with specific model combinations, but fallback strategy mitigates this risk.