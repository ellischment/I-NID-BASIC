import unittest
import os
import pandas as pd
from unittest.mock import patch, MagicMock
from src.model_pretraining import pretrain_model
from src.data_preparation import create_long_tailed_dataset
import torch
from torch.utils.data import TensorDataset, DataLoader


def mock_tokenizer(texts, **kwargs):
    # Mocking the tokenizer to return consistent input_ids and attention_mask
    input_ids = []
    attention_masks = []
    max_length = 8  # Define a maximum length for padding

    for text in texts:
        # Simulate tokenization
        num_tokens = min(len(text.split()) + 2, max_length)  # +2 for [CLS] and [SEP]
        input_id = [101] + [2023] * (num_tokens - 2) + [102]  # Example tokenized input
        input_id += [0] * (max_length - len(input_id))  # Pad with zeros
        input_ids.append(input_id)
        attention_mask = [1] * num_tokens + [0] * (max_length - num_tokens)  # Example attention mask
        attention_masks.append(attention_mask)

    return {
        'input_ids': torch.tensor(input_ids),  # Return as tensor
        'attention_mask': torch.tensor(attention_masks)  # Return as tensor
    }


class TestModelPretraining(unittest.TestCase):
    def setUp(self):
        # Create synthetic data
        self.data = {
            'text': [
                'How do I reset my password?',  # intent1
                'I want to check my balance',  # intent2
                'How can I transfer money?',  # intent3
                'What is my account number?',  # intent1
                'I need help with my card',  # intent2
                'How do I change my email?',  # intent3
                'I want to close my account',  # intent1
            ],
            'intent': [  # Ensure this column exists
                'intent1', 'intent2', 'intent3', 'intent1', 'intent2',
                'intent3', 'intent1'
            ]
        }
        self.df = pd.DataFrame(self.data)

        # Create long-tailed distribution
        self.long_tailed_df = create_long_tailed_dataset(self.df, gamma=3, num_classes=3)

        # Create the directory for saving the model
        os.makedirs("models/pretrained", exist_ok=True)

    @patch('src.model_pretraining.BertTokenizer.from_pretrained')
    @patch('src.model_pretraining.BertModel.from_pretrained')
    def test_pretrain_model(self, mock_bert_model, mock_tokenizer_class):
        """
        Test that the model pretraining function works correctly on long-tailed data.
        """
        # Mock the tokenizer
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.side_effect = mock_tokenizer  # Set side effect for the tokenizer instance
        mock_tokenizer_class.return_value = mock_tokenizer_instance

        # Mock the model
        mock_model = MagicMock()
        mock_param = torch.nn.Parameter(torch.randn(768, 768, requires_grad=True))  # Mock parameters
        mock_model.parameters.return_value = [mock_param]  # Return the mock parameter
        mock_model.return_value.last_hidden_state = torch.randn(5, 128, 768, requires_grad=True)  # Example model output
        mock_model.forward.return_value = mock_model.return_value  # Ensure forward method returns the mock output
        mock_model.save_pretrained = MagicMock()  # Mock the save_pretrained method
        mock_bert_model.return_value = mock_model

        # Call the pretraining function
        try:
            pretrain_model(self.long_tailed_df, self.long_tailed_df)  # Pass long-tailed data
        except Exception as e:
            self.fail(f"pretrain_model() raised an exception: {e}")

        # Debugging information
        print(f"Length of long-tailed DataFrame: {len(self.long_tailed_df)}")
        print(f"Tokenizer call count: {mock_tokenizer_instance.call_count}")

        # Check that the tokenizer and model were called
        mock_tokenizer_class.assert_called_once_with('bert-base-uncased')
        mock_bert_model.assert_called_once_with('bert-base-uncased')

        # Check that the tokenizer was used for processing data
        # Ожидаем, что токенизатор вызывается дважды: для labeled_data и test_data
        self.assertEqual(mock_tokenizer_instance.call_count, 2)

        # Check that the model was saved
        mock_model.save_pretrained.assert_called_once_with("models/pretrained/bert_pretrained")

    def test_create_tensor_dataset(self):
        """
        Test that TensorDataset is created correctly.
        """
        # Mock the tokenizer
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.side_effect = mock_tokenizer
        mock_tokenizer_class = MagicMock(return_value=mock_tokenizer_instance)

        # Mock the model
        mock_model = MagicMock()
        mock_bert_model = MagicMock(return_value=mock_model)

        # Call the pretraining function
        pretrain_model(self.long_tailed_df, self.long_tailed_df)

        # Check that TensorDataset is created with the correct data
        train_encodings = mock_tokenizer(list(self.long_tailed_df['text']), truncation=True, padding=True, max_length=128)
        train_labels = torch.tensor(self.long_tailed_df['label_index'].tolist())

        # Verify that TensorDataset is created with the correct input_ids, attention_mask, and labels
        self.assertTrue(isinstance(train_encodings['input_ids'], torch.Tensor))
        self.assertTrue(isinstance(train_encodings['attention_mask'], torch.Tensor))
        self.assertTrue(isinstance(train_labels, torch.Tensor))

    def test_create_data_loader(self):
        """
        Test that DataLoader is created correctly.
        """
        # Mock the tokenizer
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.side_effect = mock_tokenizer
        mock_tokenizer_class = MagicMock(return_value=mock_tokenizer_instance)

        # Mock the model
        mock_model = MagicMock()
        mock_bert_model = MagicMock(return_value=mock_model)

        # Call the pretraining function
        pretrain_model(self.long_tailed_df, self.long_tailed_df)

        # Check that DataLoader is created with the correct batch size and shuffle
        train_encodings = mock_tokenizer(list(self.long_tailed_df['text']), truncation=True, padding=True, max_length=128)
        train_labels = torch.tensor(self.long_tailed_df['label_index'].tolist())

        train_dataset = TensorDataset(
            train_encodings['input_ids'],
            train_encodings['attention_mask'],
            train_labels
        )

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        self.assertEqual(train_loader.batch_size, 32)
        self.assertTrue(train_loader.shuffle)

    def test_optimizer_and_device_initialization(self):
        """
        Test that the optimizer and device are initialized correctly.
        """
        # Mock the tokenizer
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.side_effect = mock_tokenizer
        mock_tokenizer_class = MagicMock(return_value=mock_tokenizer_instance)

        # Mock the model
        mock_model = MagicMock()
        mock_bert_model = MagicMock(return_value=mock_model)

        # Mock the optimizer
        mock_optimizer = MagicMock()
        with patch('src.model_pretraining.AdamW', return_value=mock_optimizer):
            # Call the pretraining function
            pretrain_model(self.long_tailed_df, self.long_tailed_df)

            # Check that the optimizer is initialized with the correct parameters
            mock_optimizer.assert_called_once_with(mock_model.parameters(), lr=2e-5)

            # Check that the model is moved to the correct device
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            mock_model.to.assert_called_once_with(device)

    def test_training_loop(self):
        """
        Test that the training loop works correctly.
        """
        # Mock the tokenizer
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.side_effect = mock_tokenizer
        mock_tokenizer_class = MagicMock(return_value=mock_tokenizer_instance)

        # Mock the model
        mock_model = MagicMock()
        mock_model.return_value.last_hidden_state = torch.randn(5, 128, 768, requires_grad=True)
        mock_bert_model = MagicMock(return_value=mock_model)

        # Mock the optimizer
        mock_optimizer = MagicMock()
        with patch('src.model_pretraining.AdamW', return_value=mock_optimizer):
            # Call the pretraining function
            pretrain_model(self.long_tailed_df, self.long_tailed_df)

            # Check that the training loop runs for 3 epochs
            self.assertEqual(mock_model.train.call_count, 3)

            # Check that the optimizer zero_grad is called
            self.assertEqual(mock_optimizer.zero_grad.call_count, 3 * len(self.long_tailed_df) // 32)

            # Check that the loss is computed and backpropagated
            self.assertEqual(mock_model.return_value.last_hidden_state[:, 0, :].shape, (5, 768))
            self.assertTrue(mock_optimizer.step.call_count > 0)

    def test_logging(self):
        """
        Test that logging works correctly.
        """
        # Mock the tokenizer
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.side_effect = mock_tokenizer
        mock_tokenizer_class = MagicMock(return_value=mock_tokenizer_instance)

        # Mock the model
        mock_model = MagicMock()
        mock_bert_model = MagicMock(return_value=mock_model)

        # Mock the logging
        with patch('src.model_pretraining.logging.info') as mock_logging:
            # Call the pretraining function
            pretrain_model(self.long_tailed_df, self.long_tailed_df)

            # Check that logging.info is called with the correct messages
            mock_logging.assert_any_call("Starting model pretraining")
            mock_logging.assert_any_call("Pretraining completed, model saved")


if __name__ == '__main__':
    unittest.main()