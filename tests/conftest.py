"""
Pytest configuration and fixtures for perplexity evaluation tests.
"""

import pytest
import torch
from unittest.mock import Mock, MagicMock
from transformers import AutoTokenizer

from src.perplexity.models import ChunkParams, ModelConfig, DatasetConfig, EvaluationResult
from src.config.model_configs import get_model_config


@pytest.fixture
def sample_text():
    """Sample text for testing."""
    return (
        "The quick brown fox jumps over the lazy dog. "
        "This is a sample text for testing perplexity evaluation. "
        "It contains multiple sentences and should be long enough "
        "to create several chunks during tokenization. "
        "We need sufficient text to test the sliding window approach "
        "and ensure our evaluation methodology is working correctly."
    )


@pytest.fixture
def short_text():
    """Short text for edge case testing."""
    return "Short text."


@pytest.fixture
def chunk_params():
    """Standard chunk parameters for testing."""
    return ChunkParams(block_size=128, stride_ratio=0.5, batch_size=1)


@pytest.fixture
def chunk_params_disjoint():
    """Disjoint chunk parameters (stride_ratio=1.0)."""
    return ChunkParams(block_size=128, stride_ratio=1.0, batch_size=1)


@pytest.fixture
def sample_model_config():
    """Sample model configuration for testing."""
    return ModelConfig(
        name="test-model",
        hf_name="test/test-model",
        max_length=2048,
        memory_gb=4.0,
        supports_flash_attention=True
    )


@pytest.fixture
def sample_dataset_config():
    """Sample dataset configuration for testing."""
    return DatasetConfig(
        name="test-dataset",
        hf_dataset="test/dataset",
        split="test",
        text_field="text"
    )


@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer for testing."""
    tokenizer = Mock()
    tokenizer.pad_token = "[PAD]"
    tokenizer.eos_token = "[EOS]"
    tokenizer.__len__ = Mock(return_value=50000)  # Standard vocab size

    # Mock tokenization to return simple token sequences
    def mock_tokenize(text, return_tensors=None, truncation=False):
        # Simple tokenization: split by spaces and assign incremental IDs
        tokens = text.split()
        token_ids = list(range(len(tokens)))
        
        if return_tensors == "pt":
            return {"input_ids": torch.tensor([token_ids])}
        return {"input_ids": token_ids}
    
    tokenizer.side_effect = mock_tokenize
    tokenizer.__call__ = mock_tokenize
    return tokenizer


@pytest.fixture
def mock_model():
    """Mock model for testing."""
    model = Mock()
    model.eval = Mock()
    model.gradient_checkpointing_enable = Mock()
    
    # Mock forward pass
    def mock_forward(input_ids, labels=None):
        # Return mock loss based on input
        batch_size, seq_len = input_ids.shape
        loss = torch.tensor(2.0)  # Fixed loss for testing
        
        output = Mock()
        output.loss = loss
        return output
    
    model.side_effect = mock_forward
    model.__call__ = mock_forward
    return model


@pytest.fixture
def sample_evaluation_result(chunk_params):
    """Sample evaluation result for testing."""
    return EvaluationResult(
        model_name="test-model",
        dataset_name="test-dataset",
        chunk_params=chunk_params,
        avg_nll=2.0,
        perplexity=7.389,
        num_tokens=100,
        memory_used_mb=512.0,
        evaluation_time_seconds=30.0
    )


@pytest.fixture
def mock_cuda_available(monkeypatch):
    """Mock CUDA availability."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    
    # Mock device properties
    mock_props = Mock()
    mock_props.name = "RTX 3060"
    mock_props.total_memory = 12 * 1024**3  # 12GB
    mock_props.major = 8
    mock_props.minor = 6
    
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda x: mock_props)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda: 1024**2 * 100)  # 100MB
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda: 1024**2 * 200)   # 200MB
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 1024**2 * 150)  # 150MB
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)


@pytest.fixture
def mock_cuda_unavailable(monkeypatch):
    """Mock CUDA unavailability."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)


@pytest.fixture
def mock_dataset_load(monkeypatch):
    """Mock dataset loading."""
    mock_dataset = Mock()
    mock_dataset.__len__ = Mock(return_value=100)
    mock_dataset.__getitem__ = Mock(side_effect=lambda i: {"text": f"Sample text {i}"})
    mock_dataset.select = Mock(return_value=mock_dataset)
    
    # Mock different dataset structures
    def mock_load_dataset(dataset_name, config_name=None, split="test"):
        if dataset_name == "ptb_text_only":
            mock_dataset["sentence"] = [f"Sentence {i}." for i in range(50)]
        else:
            mock_dataset["text"] = [f"Document {i} text content." for i in range(100)]
        return mock_dataset
    
    monkeypatch.setattr("src.perplexity.dataset_utils.load_dataset", mock_load_dataset)


@pytest.fixture
def mock_transformers_load(monkeypatch):
    """Mock transformers model loading."""
    def mock_tokenizer_load(model_name, **kwargs):
        tokenizer = Mock()
        tokenizer.pad_token = None
        tokenizer.eos_token = "[EOS]"
        return tokenizer
    
    def mock_model_load(model_name, **kwargs):
        model = Mock()
        model.eval = Mock()
        model.gradient_checkpointing_enable = Mock()
        return model
    
    monkeypatch.setattr("src.perplexity.model_loader.AutoTokenizer.from_pretrained", mock_tokenizer_load)
    monkeypatch.setattr("src.perplexity.model_loader.AutoModelForCausalLM.from_pretrained", mock_model_load)


@pytest.fixture
def memory_constraint_model():
    """Model that exceeds memory constraints."""
    return ModelConfig(
        name="large-model",
        hf_name="test/large-model",
        max_length=8192,
        memory_gb=16.0,  # Exceeds RTX 3060 limit
        supports_flash_attention=True
    )


@pytest.fixture
def phi3_model_config():
    """Real Phi3 model configuration for integration tests."""
    return get_model_config("Phi3-mini-4k")


# Test data generators
@pytest.fixture
def generate_evaluation_results():
    """Factory for generating multiple evaluation results."""
    def _generate(count=5, model_names=None, dataset_names=None):
        if model_names is None:
            model_names = [f"model-{i}" for i in range(count)]
        if dataset_names is None:
            dataset_names = [f"dataset-{i}" for i in range(count)]
        
        results = []
        for i in range(count):
            result = EvaluationResult(
                model_name=model_names[i % len(model_names)],
                dataset_name=dataset_names[i % len(dataset_names)],
                chunk_params=ChunkParams(block_size=2048, stride_ratio=0.5, batch_size=1),
                avg_nll=2.0 + (i * 0.1),
                perplexity=7.389 + (i * 0.5),
                num_tokens=1000 + (i * 100),
                memory_used_mb=500.0 + (i * 50),
                evaluation_time_seconds=30.0 + (i * 5)
            )
            results.append(result)
        return results
    
    return _generate


# Pytest configuration
def pytest_configure(config):
    """Pytest configuration."""
    # Add custom markers
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "gpu: marks tests that require GPU")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers."""
    for item in items:
        # Mark GPU tests
        if "gpu" in item.nodeid.lower() or "cuda" in item.nodeid.lower():
            item.add_marker(pytest.mark.gpu)
        
        # Mark integration tests
        if "integration" in item.nodeid.lower() or "test_integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        
        # Mark slow tests
        if "slow" in item.nodeid.lower() or "model" in item.nodeid.lower():
            item.add_marker(pytest.mark.slow)