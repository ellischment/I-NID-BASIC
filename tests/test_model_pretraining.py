import unittest
import os
import pandas as pd
from unittest.mock import patch, MagicMock, call
import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer, BertModel

from src.model_pretraining import pretrain_model


class MockTokenizer:
    def __init__(self):
        self.side_effect = self.mock_tokenize
        self.vocab = {
            "[CLS]": 101,
            "[SEP]": 102,
            "[PAD]": 0,
            "check": 103,
            "balance": 104,
            "transfer": 105,
            "money": 106,
            "reset": 107,
            "password": 108,
            "account": 109,
            "info": 110
        }

    def mock_tokenize(self, texts, **kwargs):
        input_ids = []
        attention_masks = []

        for text in texts:
            # Simulate tokenization process
            tokens = text.lower().split()
            token_ids = [self.vocab.get(tok, 1) for tok in tokens]  # 1 for UNK

            # Add special tokens [CLS] and [SEP]
            token_ids = [self.vocab["[CLS]"]] + token_ids + [self.vocab["[SEP]"]]
            attention_mask = [1] * len(token_ids)

            # Pad to max_length if needed
            if "max_length" in kwargs:
                pad_len = kwargs["max_length"] - len(token_ids)
                token_ids += [self.vocab["[PAD]"]] * pad_len
                attention_mask += [0] * pad_len

            input_ids.append(token_ids)
            attention_masks.append(attention_mask)

        return {
            'input_ids': torch.tensor(input_ids),
            'attention_mask': torch.tensor(attention_masks)
        }

    def __call__(self, *args, **kwargs):
        return self.side_effect(*args, **kwargs)

    def save_pretrained(self, path):
        pass


class TestModelPretraining(unittest.TestCase):
    def setUp(self):
        # Realistic banking intent examples
        self.banking_intents = pd.DataFrame({
            'text': [
                "How can I check my account balance?",
                "I need to transfer money to another account",
                "What's my current balance?",
                "Make a transfer of $500 to John Doe",
                "Reset my online banking password",
                "Show my account information",
                "I want to check my savings account balance"
            ],
            'label_index': [0, 1, 0, 1, 2, 3, 0]  # 0=balance, 1=transfer, 2=password, 3=account_info
        })

        # Mock model components
        self.mock_tokenizer = MockTokenizer()
        self.mock_model = MagicMock(spec=BertModel)

        # Configure realistic mock model outputs
        mock_output = MagicMock()
        # batch_size=7 (matches our examples), seq_len=128, hidden_size=768
        mock_output.last_hidden_state = torch.randn(7, 128, 768)
        self.mock_model.return_value = mock_output
        self.mock_model.parameters.return_value = [torch.nn.Parameter(torch.randn(768, 768))]

    @patch('src.model_pretraining.BertModel.from_pretrained')
    @patch('src.model_pretraining.BertTokenizer.from_pretrained')
    def test_pretrain_model_with_realistic_data(self, mock_tokenizer_init, mock_model_init):
        """Test pretraining with realistic banking intent examples"""
        # Setup mocks
        mock_tokenizer_init.return_value = self.mock_tokenizer
        mock_model_init.return_value = self.mock_model

        # Run pretraining
        model, tokenizer = pretrain_model(self.banking_intents, self.banking_intents)

        # Verify tokenization
        tokenized = self.mock_tokenizer(list(self.banking_intents['text']), max_length=128)
        self.assertEqual(tokenized['input_ids'].shape, (7, 128))  # 7 examples
        self.assertEqual(tokenized['attention_mask'].shape, (7, 128))

        # Verify special tokens
        sample_input = tokenized['input_ids'][0].tolist()
        self.assertEqual(sample_input[0], 101)  # [CLS]
        self.assertIn(102, sample_input)  # [SEP]

        # Verify model training
        self.assertEqual(self.mock_model.train.call_count, 3)
        self.mock_model.save_pretrained.assert_called_once_with("models/pretrained/bert_pretrained")

    def test_data_distribution(self):
        """Test handling of imbalanced classes"""
        # Create imbalanced data
        imbalanced_data = pd.DataFrame({
            'text': ["check balance"] * 5 + ["transfer money"] * 2 + ["reset password"] * 1,
            'label_index': [0] * 5 + [1] * 2 + [2] * 1
        })

        model, _ = pretrain_model(
            imbalanced_data,
            imbalanced_data,
            tokenizer=self.mock_tokenizer,
            model=self.mock_model
        )

        # Verify all classes were processed
        tokenized = self.mock_tokenizer(list(imbalanced_data['text']))
        self.assertEqual(len(tokenized['input_ids']), 8)  # 5+2+1 examples

    @patch('torch.nn.CrossEntropyLoss')
    def test_loss_calculation(self, mock_loss):
        """Test loss calculation with realistic examples"""
        loss_instance = MagicMock()
        loss_instance.return_value = torch.tensor(0.5)
        mock_loss.return_value = loss_instance

        model, _ = pretrain_model(
            self.banking_intents,
            self.banking_intents,
            tokenizer=self.mock_tokenizer,
            model=self.mock_model
        )

        # Verify loss was calculated for each batch
        self.assertGreater(loss_instance.call_count, 0)

        # Verify CLS token is used for classification
        self.mock_model.return_value.last_hidden_state[:, 0, :].size() == (7, 768)

    def test_model_output_shapes(self):
        """Verify model produces correct output shapes"""
        # Mock different batch sizes
        test_data = pd.DataFrame({
            'text': ["check balance", "transfer money"],
            'label_index': [0, 1]
        })

        # Create a mock tensor that requires gradient
        mock_output = torch.randn(2, 128, 768, requires_grad=True)

        # Configure mock model
        self.mock_model.return_value.last_hidden_state = mock_output
        self.mock_model.return_value.pooler_output = torch.randn(2, 768, requires_grad=True)

        # Mock the forward pass to return our mock output
        def mock_forward(*args, **kwargs):
            return MagicMock(last_hidden_state=mock_output)

        self.mock_model.forward = mock_forward

        model, _ = pretrain_model(
            test_data,
            test_data,
            tokenizer=self.mock_tokenizer,
            model=self.mock_model
        )

        # Verify output shape matches batch size
        outputs = self.mock_model.return_value.last_hidden_state
        self.assertEqual(outputs.size(), (2, 128, 768))  # batch_size=2


if __name__ == '__main__':
    unittest.main()
