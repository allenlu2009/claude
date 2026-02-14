"""
Tests for core perplexity evaluation logic.
"""

import pytest
import torch
from unittest.mock import Mock, patch

from src.perplexity.evaluator import PerplexityEvaluator, validate_sliding_window_benefit
from src.perplexity.models import ChunkParams, EvaluationResult


class TestPerplexityEvaluator:
    """Test cases for PerplexityEvaluator."""
    
    def test_init(self):
        """Test evaluator initialization."""
        evaluator = PerplexityEvaluator()
        assert evaluator.device == "cuda"
        
        evaluator_cpu = PerplexityEvaluator(device="cpu")
        assert evaluator_cpu.device == "cpu"
    
    def test_tokenize_and_chunk_basic(self, sample_text, mock_tokenizer, chunk_params):
        """Test basic tokenization and chunking."""
        evaluator = PerplexityEvaluator()
        
        # Mock tokenizer to return predictable token sequence
        token_ids = list(range(100))  # 100 tokens
        mock_tokenizer.return_value = {"input_ids": torch.tensor([token_ids])}
        
        samples, begin_locs = evaluator.tokenize_and_chunk(
            sample_text, mock_tokenizer, chunk_params, max_length=2048
        )
        
        assert len(samples) > 0
        assert len(samples) == len(begin_locs)
        
        # Check that chunks have the right size
        for sample in samples:
            assert sample.size(1) <= chunk_params.block_size
            assert sample.size(1) >= 2  # Minimum chunk size
    
    def test_tokenize_and_chunk_sliding_window(self, sample_text, mock_tokenizer):
        """Test sliding window chunking with overlap."""
        evaluator = PerplexityEvaluator()
        
        # Create test data with known length
        token_ids = list(range(200))  # 200 tokens
        mock_tokenizer.return_value = {"input_ids": torch.tensor([token_ids])}
        mock_tokenizer.side_effect = lambda text, **kwargs: {"input_ids": torch.tensor([token_ids])}
        mock_tokenizer.__call__ = lambda text, **kwargs: {"input_ids": torch.tensor([token_ids])}
        
        chunk_params = ChunkParams(block_size=100, stride_ratio=0.5, batch_size=1)
        
        samples, begin_locs = evaluator.tokenize_and_chunk(
            sample_text, mock_tokenizer, chunk_params, max_length=2048
        )
        
        # Should have overlapping chunks
        assert len(samples) > 2  # Multiple chunks due to stride
        
        # Check stride calculation
        expected_stride = int(100 * 0.5)  # 50
        assert chunk_params.stride == expected_stride
        
        # Verify overlap exists
        if len(begin_locs) > 1:
            first_begin, first_end, _ = begin_locs[0]
            second_begin, second_end, _ = begin_locs[1]
            assert second_begin < first_end  # Overlap exists
    
    def test_tokenize_and_chunk_disjoint(self, sample_text, mock_tokenizer):
        """Test disjoint chunking (stride_ratio=1.0)."""
        evaluator = PerplexityEvaluator()
        
        token_ids = list(range(200))
        mock_tokenizer.return_value = {"input_ids": torch.tensor([token_ids])}
        
        chunk_params = ChunkParams(block_size=100, stride_ratio=1.0, batch_size=1)
        
        samples, begin_locs = evaluator.tokenize_and_chunk(
            sample_text, mock_tokenizer, chunk_params, max_length=2048
        )
        
        # Should have non-overlapping chunks
        if len(begin_locs) > 1:
            first_begin, first_end, _ = begin_locs[0]
            second_begin, second_end, _ = begin_locs[1]
            assert second_begin == first_end  # No overlap
    
    def test_tokenize_and_chunk_short_text(self, short_text, mock_tokenizer, chunk_params):
        """Test chunking with very short text."""
        evaluator = PerplexityEvaluator()
        
        # Very few tokens
        token_ids = [1, 2]  # Only 2 tokens
        mock_tokenizer.return_value = {"input_ids": torch.tensor([token_ids])}
        
        samples, begin_locs = evaluator.tokenize_and_chunk(
            short_text, mock_tokenizer, chunk_params, max_length=2048
        )
        
        # Should handle short text gracefully
        assert len(samples) <= 1
        if samples:
            assert samples[0].size(1) >= 2
    
    def test_tokenize_and_chunk_max_length_constraint(self, sample_text, mock_tokenizer):
        """Test that block size respects model max length."""
        evaluator = PerplexityEvaluator()
        
        token_ids = list(range(1000))
        mock_tokenizer.return_value = {"input_ids": torch.tensor([token_ids])}
        
        chunk_params = ChunkParams(block_size=2048, stride_ratio=0.5, batch_size=1)
        max_length = 1024  # Smaller than block_size
        
        samples, begin_locs = evaluator.tokenize_and_chunk(
            sample_text, mock_tokenizer, chunk_params, max_length
        )
        
        # All chunks should respect max_length
        for sample in samples:
            assert sample.size(1) <= max_length
    
    def test_evaluate_model_on_chunks_basic(self, mock_model, chunk_params):
        """Test basic model evaluation on chunks."""
        evaluator = PerplexityEvaluator(device="cpu")
        
        # Create test samples
        samples = [torch.tensor([[1, 2, 3, 4, 5]]), torch.tensor([[6, 7, 8, 9, 10]])]
        begin_locs = [(0, 5, 0), (5, 10, 5)]
        
        # Mock model output
        mock_output = Mock()
        mock_output.loss = torch.tensor(2.0)
        mock_model.return_value = mock_output
        
        result = evaluator.evaluate_model_on_chunks(
            samples, begin_locs, mock_model, "test-model", "test-dataset", chunk_params
        )
        
        assert isinstance(result, EvaluationResult)
        assert result.model_name == "test-model"
        assert result.dataset_name == "test-dataset"
        assert result.perplexity > 0
        assert result.num_tokens > 0
    
    def test_evaluate_model_on_chunks_empty_samples(self, mock_model, chunk_params):
        """Test evaluation with empty samples."""
        evaluator = PerplexityEvaluator()
        
        result = evaluator.evaluate_model_on_chunks(
            [], [], mock_model, "test-model", "test-dataset", chunk_params
        )
        
        assert result.perplexity == float('inf')
        assert result.num_tokens == 0
    
    @patch('src.perplexity.evaluator.get_memory_usage')
    def test_evaluate_model_on_chunks_memory_tracking(self, mock_memory, mock_model, chunk_params):
        """Test memory usage tracking during evaluation."""
        evaluator = PerplexityEvaluator()
        
        # Mock memory usage
        mock_memory.side_effect = [
            {'cuda_available': True, 'allocated_mb': 100, 'max_allocated_mb': 150},
            {'cuda_available': True, 'allocated_mb': 200, 'max_allocated_mb': 250}
        ]
        
        samples = [torch.tensor([[1, 2, 3, 4, 5]])]
        begin_locs = [(0, 5, 0)]
        
        mock_output = Mock()
        mock_output.loss = torch.tensor(1.5)
        mock_model.return_value = mock_output
        
        result = evaluator.evaluate_model_on_chunks(
            samples, begin_locs, mock_model, "test-model", "test-dataset", chunk_params
        )
        
        assert result.memory_used_mb == 100.0  # 250 - 150
    
    def test_evaluate_text_full_pipeline(self, sample_text, mock_model, mock_tokenizer, chunk_params):
        """Test complete evaluation pipeline."""
        evaluator = PerplexityEvaluator()
        
        # Setup mocks
        token_ids = list(range(50))
        mock_tokenizer.return_value = {"input_ids": torch.tensor([token_ids])}
        
        mock_output = Mock()
        mock_output.loss = torch.tensor(2.0)
        mock_model.return_value = mock_output
        
        result = evaluator.evaluate_text(
            text=sample_text,
            model=mock_model,
            tokenizer=mock_tokenizer,
            model_name="test-model",
            dataset_name="test-dataset",
            chunk_params=chunk_params,
            max_length=2048
        )
        
        assert isinstance(result, EvaluationResult)
        assert result.model_name == "test-model"
        assert result.dataset_name == "test-dataset"
        assert result.perplexity > 0
    
    def test_compare_stride_ratios(self, sample_text, mock_model, mock_tokenizer):
        """Test stride ratio comparison."""
        evaluator = PerplexityEvaluator()
        
        # Setup mocks
        token_ids = list(range(100))
        mock_tokenizer.return_value = {"input_ids": torch.tensor([token_ids])}
        
        # Mock model to return slightly different losses for different stride ratios
        def mock_model_call(input_ids, labels=None):
            output = Mock()
            # Return loss based on input size (simulating different performance)
            output.loss = torch.tensor(2.0 + input_ids.size(1) * 0.001)
            return output
        
        mock_model.side_effect = mock_model_call
        
        stride_ratios = [0.25, 0.5, 1.0]
        results = evaluator.compare_stride_ratios(
            text=sample_text,
            model=mock_model,
            tokenizer=mock_tokenizer,
            model_name="test-model",
            dataset_name="test-dataset",
            block_size=50,
            stride_ratios=stride_ratios,
            max_length=2048
        )
        
        assert len(results) == len(stride_ratios)
        
        # Check that all results are valid
        for result in results:
            assert isinstance(result, EvaluationResult)
            assert result.perplexity > 0
            assert result.chunk_params.stride_ratio in stride_ratios


class TestSlidingWindowValidation:
    """Test sliding window benefit validation."""
    
    def test_validate_sliding_window_benefit_positive(self, generate_evaluation_results):
        """Test validation when sliding window shows benefit."""
        # Create results where lower stride ratios have better (lower) perplexity
        results = []
        stride_ratios = [0.25, 0.5, 0.75, 1.0]
        base_perplexity = 10.0
        
        for i, ratio in enumerate(stride_ratios):
            chunk_params = ChunkParams(block_size=2048, stride_ratio=ratio, batch_size=1)
            result = EvaluationResult(
                model_name="test-model",
                dataset_name="test-dataset",
                chunk_params=chunk_params,
                avg_nll=2.0,
                perplexity=base_perplexity + i * 0.5,  # Higher perplexity for higher stride ratios
                num_tokens=1000,
                memory_used_mb=500.0,
                evaluation_time_seconds=30.0
            )
            results.append(result)
        
        assert validate_sliding_window_benefit(results) is True
    
    def test_validate_sliding_window_benefit_negative(self, generate_evaluation_results):
        """Test validation when sliding window doesn't show benefit."""
        # Create results where stride ratios don't correlate with perplexity
        results = []
        stride_ratios = [0.25, 0.5, 0.75, 1.0]
        perplexities = [12.0, 10.0, 15.0, 8.0]  # Random order
        
        for ratio, ppl in zip(stride_ratios, perplexities):
            chunk_params = ChunkParams(block_size=2048, stride_ratio=ratio, batch_size=1)
            result = EvaluationResult(
                model_name="test-model",
                dataset_name="test-dataset",
                chunk_params=chunk_params,
                avg_nll=2.0,
                perplexity=ppl,
                num_tokens=1000,
                memory_used_mb=500.0,
                evaluation_time_seconds=30.0
            )
            results.append(result)
        
        # This should return False as sliding window doesn't show consistent benefit
        assert validate_sliding_window_benefit(results) is False
    
    def test_validate_sliding_window_benefit_insufficient_data(self):
        """Test validation with insufficient data."""
        # Single result
        chunk_params = ChunkParams(block_size=2048, stride_ratio=0.5, batch_size=1)
        result = EvaluationResult(
            model_name="test-model",
            dataset_name="test-dataset",
            chunk_params=chunk_params,
            avg_nll=2.0,
            perplexity=10.0,
            num_tokens=1000,
            memory_used_mb=500.0,
            evaluation_time_seconds=30.0
        )
        
        # Should return True (can't compare, so assume valid)
        assert validate_sliding_window_benefit([result]) is True
    
    def test_validate_sliding_window_benefit_empty(self):
        """Test validation with empty results."""
        assert validate_sliding_window_benefit([]) is True


# Test edge cases and error handling
class TestEvaluatorEdgeCases:
    """Test edge cases and error handling."""
    
    def test_evaluator_with_invalid_device(self):
        """Test evaluator with invalid device."""
        # Should not raise error during initialization
        evaluator = PerplexityEvaluator(device="invalid")
        assert evaluator.device == "invalid"
    
    def test_tokenize_and_chunk_stride_larger_than_block(self, sample_text, mock_tokenizer):
        """Test stride larger than block size."""
        evaluator = PerplexityEvaluator()
        
        token_ids = list(range(100))
        mock_tokenizer.return_value = {"input_ids": torch.tensor([token_ids])}
        
        # Stride ratio > 1.0 should be handled gracefully
        chunk_params = ChunkParams(block_size=50, stride_ratio=1.5, batch_size=1)
        
        # Should adjust stride automatically
        samples, begin_locs = evaluator.tokenize_and_chunk(
            sample_text, mock_tokenizer, chunk_params, max_length=2048
        )
        
        assert len(samples) > 0  # Should still work
    
    def test_evaluate_model_with_error_chunks(self, mock_model, chunk_params):
        """Test evaluation when model throws errors on some chunks."""
        evaluator = PerplexityEvaluator()

        samples = [torch.tensor([[1, 2, 3]]), torch.tensor([[4, 5, 6]])]
        begin_locs = [(0, 3, 0), (3, 6, 3)]

        # Mock model to throw error on first chunk, succeed on second
        def mock_model_call(input_ids, labels=None):
            if input_ids[0, 0].item() == 1:  # First chunk
                raise RuntimeError("Mock error")
            output = Mock()
            output.loss = torch.tensor(2.0)
            return output

        mock_model.side_effect = mock_model_call

        result = evaluator.evaluate_model_on_chunks(
            samples, begin_locs, mock_model, "test-model", "test-dataset", chunk_params
        )

        # Should still return valid result from successful chunks
        assert result.num_tokens > 0
        assert result.perplexity > 0


class TestTokenIdValidation:
    """Test token ID range validation in tokenize_and_chunk."""

    def test_out_of_range_tokens_clamped(self):
        """Test that out-of-range token IDs are clamped to vocab size."""
        evaluator = PerplexityEvaluator()

        mock_tokenizer = Mock()
        mock_tokenizer.__len__ = Mock(return_value=100)  # vocab_size = 100
        # Return tokens with some IDs >= vocab_size
        token_ids = list(range(50)) + [150, 200, 99]  # 150, 200 are out of range
        mock_tokenizer.return_value = {"input_ids": torch.tensor([token_ids])}

        chunk_params = ChunkParams(block_size=128, stride_ratio=1.0, batch_size=1)

        samples, begin_locs = evaluator.tokenize_and_chunk(
            "test text", mock_tokenizer, chunk_params, max_length=2048
        )

        # All token IDs should be < vocab_size (100)
        for sample in samples:
            assert sample.max().item() < 100

    def test_valid_tokens_unchanged(self):
        """Test that valid token IDs are not modified."""
        evaluator = PerplexityEvaluator()

        mock_tokenizer = Mock()
        mock_tokenizer.__len__ = Mock(return_value=1000)
        token_ids = list(range(50))  # All valid
        mock_tokenizer.return_value = {"input_ids": torch.tensor([token_ids])}

        chunk_params = ChunkParams(block_size=128, stride_ratio=1.0, batch_size=1)

        samples, begin_locs = evaluator.tokenize_and_chunk(
            "test text", mock_tokenizer, chunk_params, max_length=2048
        )

        # Tokens should be unchanged
        all_tokens = torch.cat(samples, dim=1)
        assert all_tokens.tolist()[0] == token_ids

    def test_all_tokens_out_of_range(self):
        """Test edge case where all tokens are out of range."""
        evaluator = PerplexityEvaluator()

        mock_tokenizer = Mock()
        mock_tokenizer.__len__ = Mock(return_value=10)
        token_ids = [50, 60, 70, 80, 90]  # All out of range for vocab_size=10
        mock_tokenizer.return_value = {"input_ids": torch.tensor([token_ids])}

        chunk_params = ChunkParams(block_size=128, stride_ratio=1.0, batch_size=1)

        samples, begin_locs = evaluator.tokenize_and_chunk(
            "test text", mock_tokenizer, chunk_params, max_length=2048
        )

        # All should be clamped to 9 (vocab_size - 1)
        for sample in samples:
            assert sample.max().item() == 9