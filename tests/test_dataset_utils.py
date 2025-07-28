"""
Tests for dataset utilities and loading.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from src.perplexity.dataset_utils import (
    load_dataset_text, validate_text_length, get_text_stats, preview_text,
    load_wikitext2, load_ptb, load_shakespeare, DatasetLoadError,
    _load_text_from_config, _extract_text_from_dataset
)
from src.perplexity.models import DatasetConfig


class TestDatasetLoading:
    """Test dataset loading functionality."""
    
    @patch('src.perplexity.dataset_utils.load_dataset')
    @patch('src.perplexity.dataset_utils.get_dataset_config')
    def test_load_dataset_text_success(self, mock_get_config, mock_load_dataset):
        """Test successful dataset text loading."""
        # Setup dataset config
        config = DatasetConfig(
            name="test-dataset",
            hf_dataset="test/dataset",
            split="test",
            text_field="text"
        )
        mock_get_config.return_value = config
        
        # Setup mock dataset
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=100)
        mock_dataset.__getitem__ = Mock(side_effect=lambda key: ["Document 1 text", "Document 2 text", "Document 3 text"] if key == "text" else None)
        mock_load_dataset.return_value = mock_dataset
        
        result = load_dataset_text("test-dataset")
        
        assert isinstance(result, str)
        assert len(result) > 0
        mock_get_config.assert_called_once_with("test-dataset")
        mock_load_dataset.assert_called_once()
    
    @patch('src.perplexity.dataset_utils.get_dataset_config')
    def test_load_dataset_text_unknown_dataset(self, mock_get_config):
        """Test loading unknown dataset."""
        mock_get_config.side_effect = ValueError("Unknown dataset")
        
        with pytest.raises(DatasetLoadError):
            load_dataset_text("unknown-dataset")
    
    @patch('src.perplexity.dataset_utils.load_dataset')
    @patch('src.perplexity.dataset_utils.get_dataset_config')
    def test_load_dataset_text_with_sample_limit(self, mock_get_config, mock_load_dataset):
        """Test dataset loading with sample limit."""
        config = DatasetConfig(
            name="test-dataset",
            hf_dataset="test/dataset",
            split="test",
            text_field="text"
        )
        mock_get_config.return_value = config
        
        # Setup mock dataset with select method
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=1000)
        mock_dataset.__getitem__ = Mock(side_effect=lambda key: [f"Document {i}" for i in range(100)] if key == "text" else None)
        
        mock_subset = Mock()
        mock_subset.__len__ = Mock(return_value=10)
        mock_subset.__getitem__ = Mock(side_effect=lambda key: [f"Document {i}" for i in range(10)] if key == "text" else None)
        mock_dataset.select = Mock(return_value=mock_subset)
        
        mock_load_dataset.return_value = mock_dataset
        
        result = load_dataset_text("test-dataset", max_samples=10)
        
        # Should have called select with range(10)
        mock_dataset.select.assert_called_once()
        call_args = mock_dataset.select.call_args[0][0]
        assert list(call_args) == list(range(10))
    
    def test_extract_text_from_dataset_ptb(self):
        """Test text extraction for PTB dataset."""
        config = DatasetConfig(
            name="PTB",
            hf_dataset=("ptb_text_only", "penn_treebank"),
            split="test",
            text_field="sentence"
        )
        
        mock_dataset = Mock()
        mock_dataset.__getitem__ = Mock(side_effect=lambda key: ["Sentence 1.", "Sentence 2.", "Sentence 3."] if key == "sentence" else None)
        
        result = _extract_text_from_dataset(mock_dataset, config)
        
        # PTB should join with " \n "
        expected = "Sentence 1. \n Sentence 2. \n Sentence 3."
        assert result == expected
    
    def test_extract_text_from_dataset_wikitext(self):
        """Test text extraction for WikiText datasets."""
        config = DatasetConfig(
            name="Wikitext2",
            hf_dataset=("wikitext", "wikitext-2-raw-v1"),
            split="test",
            text_field="text"
        )
        
        mock_dataset = Mock()
        mock_dataset.__getitem__ = Mock(side_effect=lambda key: ["Article 1 text", "", "Article 2 text", "Article 3 text"] if key == "text" else None)
        
        result = _extract_text_from_dataset(mock_dataset, config)
        
        # WikiText should join with "\n\n" and filter empty strings
        expected = "Article 1 text\n\nArticle 2 text\n\nArticle 3 text"
        assert result == expected
    
    def test_extract_text_from_dataset_shakespeare(self):
        """Test text extraction for Shakespeare dataset."""
        config = DatasetConfig(
            name="Shakespeare",
            hf_dataset="karpathy/tiny_shakespeare",
            split="train",
            text_field="text"
        )
        
        mock_dataset = Mock()
        mock_dataset.__getitem__ = Mock(side_effect=lambda key: ["To be or not to be", "That is the question"] if key == "text" else None)
        
        result = _extract_text_from_dataset(mock_dataset, config)
        
        expected = "To be or not to be\n\nThat is the question"
        assert result == expected
    
    def test_extract_text_from_dataset_default(self):
        """Test text extraction for unknown dataset (default behavior)."""
        config = DatasetConfig(
            name="unknown-dataset",
            hf_dataset="test/unknown",
            split="test",
            text_field="text"
        )
        
        mock_dataset = Mock()
        mock_dataset.__getitem__ = Mock(side_effect=lambda key: ["Text 1", "Text 2", ""] if key == "text" else None)
        
        result = _extract_text_from_dataset(mock_dataset, config)
        
        # Default should join with "\n\n" and filter empty strings
        expected = "Text 1\n\nText 2"
        assert result == expected


class TestTextValidation:
    """Test text validation and statistics."""
    
    def test_validate_text_length_valid(self):
        """Test text length validation with valid text."""
        text = "This is a long enough text that should pass validation " * 20
        
        result = validate_text_length(text, min_length=100)
        assert result is True
    
    def test_validate_text_length_invalid(self):
        """Test text length validation with short text."""
        text = "Short"
        
        result = validate_text_length(text, min_length=100)
        assert result is False
    
    def test_validate_text_length_exact(self):
        """Test text length validation at exact minimum."""
        text = "a" * 100
        
        result = validate_text_length(text, min_length=100)
        assert result is True
    
    def test_get_text_stats_basic(self):
        """Test basic text statistics."""
        text = "Hello world.\nThis is a test.\nThird line here."
        
        stats = get_text_stats(text)
        
        assert stats['total_characters'] == len(text)
        assert stats['total_lines'] == 3
        assert stats['total_words'] == 9
        assert stats['avg_line_length'] > 0
        assert stats['avg_word_length'] > 0
    
    def test_get_text_stats_empty(self):
        """Test text statistics with empty text."""
        text = ""
        
        stats = get_text_stats(text)
        
        assert stats['total_characters'] == 0
        assert stats['total_lines'] == 1  # Split always returns at least one element
        assert stats['total_words'] == 0
        assert stats['avg_line_length'] == 0
        assert stats['avg_word_length'] == 0
    
    def test_preview_text_short(self):
        """Test text preview with short text."""
        text = "Short text"
        
        preview = preview_text(text, num_chars=20)
        assert preview == text
    
    def test_preview_text_long(self):
        """Test text preview with long text."""
        text = "This is a very long text that should be truncated"
        
        preview = preview_text(text, num_chars=20)
        assert preview == "This is a very long ..."
        assert len(preview) == 23  # 20 + "..."


class TestConvenienceFunctions:
    """Test convenience functions for specific datasets."""
    
    @patch('src.perplexity.dataset_utils.load_dataset_text')
    def test_load_wikitext2(self, mock_load):
        """Test WikiText-2 convenience function."""
        mock_load.return_value = "Sample WikiText-2 content"
        
        result = load_wikitext2(max_samples=100)
        
        mock_load.assert_called_once_with("Wikitext2", 100, None, False)
        assert result == "Sample WikiText-2 content"
    
    @patch('src.perplexity.dataset_utils.load_dataset_text')
    def test_load_ptb(self, mock_load):
        """Test PTB convenience function."""
        mock_load.return_value = "Sample PTB content"
        
        result = load_ptb(max_samples=50)
        
        mock_load.assert_called_once_with("PTB", 50, None, False)
        assert result == "Sample PTB content"
    
    @patch('src.perplexity.dataset_utils.load_dataset_text')
    def test_load_shakespeare(self, mock_load):
        """Test Shakespeare convenience function."""
        mock_load.return_value = "Sample Shakespeare content"
        
        result = load_shakespeare()
        
        mock_load.assert_called_once_with("Shakespeare", None, None, False)
        assert result == "Sample Shakespeare content"


class TestDatasetConfigIntegration:
    """Test dataset utilities with actual configurations."""
    
    @patch('src.perplexity.dataset_utils.load_dataset')
    def test_load_text_from_config_tuple_dataset(self, mock_load_dataset):
        """Test loading from tuple-based dataset configuration."""
        config = DatasetConfig(
            name="Wikitext2",
            hf_dataset=("wikitext", "wikitext-2-raw-v1"),
            split="test",
            text_field="text"
        )
        
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=100)
        mock_dataset.__getitem__ = Mock(side_effect=lambda key: ["Article 1", "Article 2"] if key == "text" else None)
        mock_dataset.select = Mock(return_value=mock_dataset)
        mock_load_dataset.return_value = mock_dataset
        
        result = _load_text_from_config(config, max_samples=10)
        
        # Should call load_dataset with tuple arguments
        mock_load_dataset.assert_called_once_with("wikitext", "wikitext-2-raw-v1", split="test")
        assert isinstance(result, str)
    
    @patch('src.perplexity.dataset_utils.load_dataset')
    def test_load_text_from_config_string_dataset(self, mock_load_dataset):
        """Test loading from string-based dataset configuration."""
        config = DatasetConfig(
            name="Shakespeare",
            hf_dataset="karpathy/tiny_shakespeare",
            split="train",
            text_field="text"
        )
        
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=100)
        mock_dataset.__getitem__ = Mock(side_effect=lambda key: ["Shakespeare text"] if key == "text" else None)
        mock_dataset.select = Mock(return_value=mock_dataset)
        mock_load_dataset.return_value = mock_dataset
        
        result = _load_text_from_config(config)
        
        # Should call load_dataset with string argument
        mock_load_dataset.assert_called_once_with("karpathy/tiny_shakespeare", split="train")
        assert isinstance(result, str)
    
    @patch('src.perplexity.dataset_utils.load_dataset')
    def test_load_text_from_config_with_config_max_samples(self, mock_load_dataset):
        """Test loading with max_samples from config."""
        config = DatasetConfig(
            name="test-dataset",
            hf_dataset="test/dataset",
            split="test",
            text_field="text",
            max_samples=20
        )
        
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=100)
        mock_dataset.__getitem__ = Mock(side_effect=lambda key: [f"Text {i}" for i in range(20)] if key == "text" else None)
        
        mock_subset = Mock()
        mock_subset.__len__ = Mock(return_value=20)
        mock_subset.__getitem__ = Mock(side_effect=lambda key: [f"Text {i}" for i in range(20)] if key == "text" else None)
        mock_dataset.select = Mock(return_value=mock_subset)
        
        mock_load_dataset.return_value = mock_dataset
        
        result = _load_text_from_config(config)
        
        # Should use max_samples from config
        mock_dataset.select.assert_called_once()
    
    @patch('src.perplexity.dataset_utils.load_dataset')
    def test_load_text_from_config_error_handling(self, mock_load_dataset):
        """Test error handling in dataset loading."""
        config = DatasetConfig(
            name="error-dataset",
            hf_dataset="error/dataset",
            split="test",
            text_field="text"
        )
        
        mock_load_dataset.side_effect = Exception("Dataset loading failed")
        
        with pytest.raises(Exception):
            _load_text_from_config(config)


# Edge cases and error handling
class TestDatasetUtilsEdgeCases:
    """Test edge cases and error handling."""
    
    def test_extract_text_empty_list(self):
        """Test text extraction with empty text list."""
        config = DatasetConfig(
            name="empty-dataset",
            hf_dataset="test/empty",
            split="test",
            text_field="text"
        )
        
        mock_dataset = Mock()
        mock_dataset.__getitem__ = Mock(side_effect=lambda key: [] if key == "text" else None)
        
        result = _extract_text_from_dataset(mock_dataset, config)
        assert result == ""
    
    def test_extract_text_all_empty_strings(self):
        """Test text extraction with all empty strings."""
        config = DatasetConfig(
            name="Wikitext2",
            hf_dataset=("wikitext", "wikitext-2-raw-v1"),
            split="test",
            text_field="text"
        )
        
        mock_dataset = Mock()
        mock_dataset.__getitem__ = Mock(side_effect=lambda key: ["", "   ", ""] if key == "text" else None)
        
        result = _extract_text_from_dataset(mock_dataset, config)
        assert result == ""  # Should filter out empty and whitespace-only strings
    
    def test_extract_text_single_string(self):
        """Test text extraction when text field contains single string."""
        config = DatasetConfig(
            name="single-string",
            hf_dataset="test/single",
            split="test",
            text_field="text"
        )
        
        mock_dataset = Mock()
        mock_dataset.__getitem__ = Mock(side_effect=lambda key: "Single string content" if key == "text" else None)
        
        result = _extract_text_from_dataset(mock_dataset, config)
        assert result == "Single string content"
    
    def test_get_text_stats_single_line(self):
        """Test text statistics with single line."""
        text = "Single line text"
        
        stats = get_text_stats(text)
        
        assert stats['total_lines'] == 1
        assert stats['total_words'] == 3
        assert stats['avg_line_length'] == len(text)
    
    def test_preview_text_exact_length(self):
        """Test text preview at exact character limit."""
        text = "Exactly twenty chars"  # 20 characters
        
        preview = preview_text(text, num_chars=20)
        assert preview == text  # Should not add "..." when exact length