"""
Tests for model loading and memory optimization.
"""

import pytest
import torch
from unittest.mock import Mock, patch, MagicMock

from src.perplexity.model_loader import (
    load_model_and_tokenizer, check_memory_constraints, get_device_info,
    clear_gpu_memory, unload_model, get_memory_usage, estimate_model_memory,
    ModelLoadError, MemoryConstraintError
)
from src.perplexity.models import ModelConfig


class TestMemoryConstraints:
    """Test memory constraint checking."""
    
    def test_check_memory_constraints_pass(self, sample_model_config):
        """Test memory constraint check that passes."""
        sample_model_config.memory_gb = 8.0
        
        result = check_memory_constraints(sample_model_config, memory_limit_gb=12.0)
        assert result is True
    
    def test_check_memory_constraints_fail(self, sample_model_config):
        """Test memory constraint check that fails."""
        sample_model_config.memory_gb = 16.0
        
        with pytest.raises(MemoryConstraintError):
            check_memory_constraints(sample_model_config, memory_limit_gb=12.0)
    
    def test_check_memory_constraints_exact(self, sample_model_config):
        """Test memory constraint check at exact limit."""
        sample_model_config.memory_gb = 12.0
        
        result = check_memory_constraints(sample_model_config, memory_limit_gb=12.0)
        assert result is True


class TestDeviceInfo:
    """Test device information functions."""
    
    def test_get_device_info_cuda_available(self, mock_cuda_available):
        """Test device info when CUDA is available."""
        info = get_device_info()
        
        assert info['cuda_available'] is True
        assert info['device_count'] == 1
        assert 'gpu_name' in info
        assert 'total_memory_gb' in info
        assert 'compute_capability' in info
        assert 'allocated_memory_mb' in info
        assert 'cached_memory_mb' in info
    
    def test_get_device_info_cuda_unavailable(self, mock_cuda_unavailable):
        """Test device info when CUDA is unavailable."""
        info = get_device_info()
        
        assert info['cuda_available'] is False
        assert info['device_count'] == 0
        assert 'gpu_name' not in info
    
    def test_clear_gpu_memory_cuda_available(self, mock_cuda_available):
        """Test GPU memory clearing when CUDA is available."""
        # Should not raise any errors
        clear_gpu_memory()
    
    def test_clear_gpu_memory_cuda_unavailable(self, mock_cuda_unavailable):
        """Test GPU memory clearing when CUDA is unavailable."""
        # Should not raise any errors
        clear_gpu_memory()
    
    def test_get_memory_usage_cuda_available(self, mock_cuda_available):
        """Test memory usage when CUDA is available."""
        usage = get_memory_usage()
        
        assert usage['cuda_available'] is True
        assert 'allocated_mb' in usage
        assert 'cached_mb' in usage
        assert 'max_allocated_mb' in usage
    
    def test_get_memory_usage_cuda_unavailable(self, mock_cuda_unavailable):
        """Test memory usage when CUDA is unavailable."""
        usage = get_memory_usage()
        
        assert usage['cuda_available'] is False
        assert len(usage) == 1  # Only cuda_available key


class TestModelLoading:
    """Test model and tokenizer loading."""
    
    @patch('src.perplexity.model_loader.get_model_config')
    @patch('src.perplexity.model_loader._load_tokenizer')
    @patch('src.perplexity.model_loader._load_model_with_attention_fallback')
    def test_load_model_and_tokenizer_success(self, mock_load_model, mock_load_tokenizer, mock_get_config):
        """Test successful model and tokenizer loading."""
        # Setup mocks
        mock_config = ModelConfig(
            name="test-model",
            hf_name="test/model",
            max_length=2048,
            memory_gb=4.0,
            supports_flash_attention=True
        )
        mock_get_config.return_value = mock_config
        
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "[EOS]"
        mock_load_tokenizer.return_value = mock_tokenizer
        
        mock_model = Mock()
        mock_model.eval = Mock()
        mock_load_model.return_value = mock_model
        
        # Test loading
        model, tokenizer = load_model_and_tokenizer("test-model")
        
        assert model is mock_model
        assert tokenizer is mock_tokenizer
        assert tokenizer.pad_token == "[EOS]"  # Should set pad_token to eos_token
        mock_model.eval.assert_called_once()
    
    @patch('src.perplexity.model_loader.get_model_config')
    def test_load_model_and_tokenizer_unknown_model(self, mock_get_config):
        """Test loading unknown model."""
        mock_get_config.side_effect = ValueError("Unknown model")
        
        with pytest.raises(ModelLoadError):
            load_model_and_tokenizer("unknown-model")
    
    @patch('src.perplexity.model_loader.get_model_config')
    @patch('src.perplexity.model_loader.check_memory_constraints')
    def test_load_model_and_tokenizer_memory_constraint(self, mock_check_memory, mock_get_config):
        """Test loading with memory constraint violation."""
        mock_config = ModelConfig(
            name="large-model",
            hf_name="test/large-model",
            max_length=2048,
            memory_gb=16.0,
            supports_flash_attention=True
        )
        mock_get_config.return_value = mock_config
        mock_check_memory.side_effect = MemoryConstraintError("Memory limit exceeded")
        
        with pytest.raises(MemoryConstraintError):
            load_model_and_tokenizer("large-model", memory_limit_gb=12.0)
    
    @patch('src.perplexity.model_loader.get_model_config')
    @patch('src.perplexity.model_loader._load_tokenizer')
    @patch('src.perplexity.model_loader._load_model_with_attention_fallback')
    def test_load_model_and_tokenizer_device_auto(self, mock_load_model, mock_load_tokenizer, mock_get_config, mock_cuda_available):
        """Test loading with device='auto'."""
        mock_config = ModelConfig(
            name="test-model",
            hf_name="test/model",
            max_length=2048,
            memory_gb=4.0,
            supports_flash_attention=True
        )
        mock_get_config.return_value = mock_config
        
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = "[PAD]"
        mock_load_tokenizer.return_value = mock_tokenizer
        
        mock_model = Mock()
        mock_model.eval = Mock()
        mock_load_model.return_value = mock_model
        
        model, tokenizer = load_model_and_tokenizer("test-model", device="auto")
        
        # Should use cuda since it's available
        mock_load_model.assert_called_once()
        args, kwargs = mock_load_model.call_args
        assert args[1] == "cuda"  # device argument
    
    @patch('src.perplexity.model_loader.AutoTokenizer.from_pretrained')
    def test_load_tokenizer_success(self, mock_tokenizer_load):
        """Test successful tokenizer loading."""
        from src.perplexity.model_loader import _load_tokenizer
        
        mock_tokenizer = Mock()
        mock_tokenizer_load.return_value = mock_tokenizer
        
        result = _load_tokenizer("test/model")
        
        assert result is mock_tokenizer
        mock_tokenizer_load.assert_called_once_with(
            "test/model",
            trust_remote_code=True
        )
    
    @patch('src.perplexity.model_loader.AutoModelForCausalLM.from_pretrained')
    def test_load_model_with_flash_attention_success(self, mock_model_load):
        """Test successful model loading with flash attention."""
        from src.perplexity.model_loader import _load_model_with_attention_fallback
        
        mock_model = Mock()
        mock_model.gradient_checkpointing_enable = Mock()
        mock_model_load.return_value = mock_model
        
        model_config = ModelConfig(
            name="test-model",
            hf_name="test/model",
            max_length=2048,
            memory_gb=4.0,
            supports_flash_attention=True
        )
        
        result = _load_model_with_attention_fallback(model_config, "cuda", True)
        
        assert result is mock_model
        mock_model.gradient_checkpointing_enable.assert_called_once()
        
        # Should have tried flash attention first
        call_args = mock_model_load.call_args[1]
        assert call_args['attn_implementation'] == 'flash_attention_2'
    
    @patch('src.perplexity.model_loader.AutoModelForCausalLM.from_pretrained')
    def test_load_model_with_flash_attention_fallback(self, mock_model_load):
        """Test model loading with flash attention fallback to eager."""
        from src.perplexity.model_loader import _load_model_with_attention_fallback
        
        mock_model = Mock()
        mock_model.gradient_checkpointing_enable = Mock()
        
        # First call (flash attention) fails, second call (eager) succeeds
        mock_model_load.side_effect = [
            Exception("Flash attention not supported"),
            mock_model
        ]
        
        model_config = ModelConfig(
            name="test-model",
            hf_name="test/model",
            max_length=2048,
            memory_gb=4.0,
            supports_flash_attention=True
        )
        
        result = _load_model_with_attention_fallback(model_config, "cuda", True)
        
        assert result is mock_model
        assert mock_model_load.call_count == 2
        
        # Check that second call used eager attention
        second_call_args = mock_model_load.call_args[1]
        assert second_call_args['attn_implementation'] == 'eager'
    
    @patch('src.perplexity.model_loader.AutoModelForCausalLM.from_pretrained')
    def test_load_model_no_flash_attention_support(self, mock_model_load):
        """Test model loading when flash attention is not supported."""
        from src.perplexity.model_loader import _load_model_with_attention_fallback
        
        mock_model = Mock()
        mock_model.gradient_checkpointing_enable = Mock()
        mock_model_load.return_value = mock_model
        
        model_config = ModelConfig(
            name="test-model",
            hf_name="test/model",
            max_length=2048,
            memory_gb=4.0,
            supports_flash_attention=False
        )
        
        result = _load_model_with_attention_fallback(model_config, "cuda", True)
        
        assert result is mock_model
        # Should have skipped flash attention and used eager directly
        call_args = mock_model_load.call_args[1]
        assert call_args['attn_implementation'] == 'eager'
    
    @patch('src.perplexity.model_loader.AutoModelForCausalLM.from_pretrained')
    def test_load_model_flash_attention_disabled(self, mock_model_load):
        """Test model loading when flash attention is disabled."""
        from src.perplexity.model_loader import _load_model_with_attention_fallback
        
        mock_model = Mock()
        mock_model.gradient_checkpointing_enable = Mock()
        mock_model_load.return_value = mock_model
        
        model_config = ModelConfig(
            name="test-model",
            hf_name="test/model",
            max_length=2048,
            memory_gb=4.0,
            supports_flash_attention=True
        )
        
        result = _load_model_with_attention_fallback(model_config, "cuda", False)
        
        assert result is mock_model
        # Should have used eager attention directly
        call_args = mock_model_load.call_args[1]
        assert call_args['attn_implementation'] == 'eager'
    
    @patch('src.perplexity.model_loader.clear_gpu_memory')
    def test_unload_model(self, mock_clear_memory):
        """Test model unloading."""
        mock_model = Mock()
        
        unload_model(mock_model)
        
        mock_clear_memory.assert_called_once()


class TestMemoryEstimation:
    """Test memory usage estimation."""
    
    def test_estimate_model_memory_base_context(self, sample_model_config):
        """Test memory estimation with base context length."""
        sample_model_config.memory_gb = 8.0
        
        estimated = estimate_model_memory(sample_model_config, context_length=2048)
        
        # Should return base memory for base context
        assert estimated == 8.0
    
    def test_estimate_model_memory_longer_context(self, sample_model_config):
        """Test memory estimation with longer context."""
        sample_model_config.memory_gb = 8.0
        
        estimated = estimate_model_memory(sample_model_config, context_length=8192)
        
        # Should be higher than base memory
        assert estimated > 8.0
    
    def test_estimate_model_memory_shorter_context(self, sample_model_config):
        """Test memory estimation with shorter context."""
        sample_model_config.memory_gb = 8.0
        
        estimated = estimate_model_memory(sample_model_config, context_length=1024)
        
        # Should return base memory (no reduction for shorter context)
        assert estimated == 8.0


class TestErrorHandling:
    """Test error handling in model loading."""
    
    @patch('src.perplexity.model_loader.get_model_config')
    @patch('src.perplexity.model_loader._load_tokenizer')
    def test_load_model_tokenizer_error(self, mock_load_tokenizer, mock_get_config):
        """Test handling of tokenizer loading error."""
        mock_config = ModelConfig(
            name="test-model",
            hf_name="test/model",
            max_length=2048,
            memory_gb=4.0,
            supports_flash_attention=True
        )
        mock_get_config.return_value = mock_config
        mock_load_tokenizer.side_effect = Exception("Tokenizer loading failed")
        
        with pytest.raises(ModelLoadError):
            load_model_and_tokenizer("test-model")
    
    @patch('src.perplexity.model_loader.get_model_config')
    @patch('src.perplexity.model_loader._load_tokenizer')
    @patch('src.perplexity.model_loader._load_model_with_attention_fallback')
    def test_load_model_model_error(self, mock_load_model, mock_load_tokenizer, mock_get_config):
        """Test handling of model loading error."""
        mock_config = ModelConfig(
            name="test-model",
            hf_name="test/model",
            max_length=2048,
            memory_gb=4.0,
            supports_flash_attention=True
        )
        mock_get_config.return_value = mock_config
        
        mock_tokenizer = Mock()
        mock_load_tokenizer.return_value = mock_tokenizer
        mock_load_model.side_effect = Exception("Model loading failed")
        
        with pytest.raises(ModelLoadError):
            load_model_and_tokenizer("test-model")
    
    @patch('src.perplexity.model_loader.AutoModelForCausalLM.from_pretrained')
    def test_load_model_both_attention_types_fail(self, mock_model_load):
        """Test when both flash attention and eager attention fail."""
        from src.perplexity.model_loader import _load_model_with_attention_fallback
        
        # Both calls fail
        mock_model_load.side_effect = [
            Exception("Flash attention failed"),
            Exception("Eager attention failed")
        ]
        
        model_config = ModelConfig(
            name="test-model",
            hf_name="test/model",
            max_length=2048,
            memory_gb=4.0,
            supports_flash_attention=True
        )
        
        with pytest.raises(Exception):
            _load_model_with_attention_fallback(model_config, "cuda", True)


class TestIntegrationScenarios:
    """Test integration scenarios."""
    
    @patch('src.perplexity.model_loader.get_model_config')
    @patch('src.perplexity.model_loader._load_tokenizer')
    @patch('src.perplexity.model_loader._load_model_with_attention_fallback')
    def test_rtx_3060_compatible_model(self, mock_load_model, mock_load_tokenizer, mock_get_config, mock_cuda_available):
        """Test loading RTX 3060 compatible model."""
        # Phi3-mini-4k config (within RTX 3060 limits)
        mock_config = ModelConfig(
            name="Phi3-mini-4k",
            hf_name="microsoft/Phi-3-mini-4k-instruct",
            max_length=4096,
            memory_gb=7.5,
            supports_flash_attention=True
        )
        mock_get_config.return_value = mock_config
        
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "[EOS]"
        mock_load_tokenizer.return_value = mock_tokenizer
        
        mock_model = Mock()
        mock_model.eval = Mock()
        mock_load_model.return_value = mock_model
        
        model, tokenizer = load_model_and_tokenizer(
            "Phi3-mini-4k",
            memory_limit_gb=12.0,
            use_flash_attention=True
        )
        
        assert model is mock_model
        assert tokenizer is mock_tokenizer
        # Should have attempted flash attention
        mock_load_model.assert_called_once_with(mock_config, "cuda", True)
    
    @patch('src.perplexity.model_loader.get_model_config')
    def test_model_exceeds_rtx_3060_limits(self, mock_get_config):
        """Test model that exceeds RTX 3060 memory limits."""
        # Large model config (exceeds RTX 3060 limits)
        mock_config = ModelConfig(
            name="large-model",
            hf_name="test/large-model",
            max_length=8192,
            memory_gb=16.0,
            supports_flash_attention=True
        )
        mock_get_config.return_value = mock_config
        
        with pytest.raises(MemoryConstraintError):
            load_model_and_tokenizer("large-model", memory_limit_gb=12.0)