# tests/test_evaluation.py
import unittest
import os
import tempfile
import shutil
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import torch


class TestEvaluateModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Setup temp directories
        cls.temp_dir = tempfile.mkdtemp()
        cls.data_dir = os.path.join(cls.temp_dir, "data/processed")
        cls.model_dir = os.path.join(cls.temp_dir, "models/finetuned")
        cls.log_dir = os.path.join(cls.temp_dir, "logs")

        os.makedirs(cls.data_dir, exist_ok=True)
        os.makedirs(cls.model_dir, exist_ok=True)
        os.makedirs(cls.log_dir, exist_ok=True)

        # Create test data - 4 samples
        test_data = pd.DataFrame({
            'text': ['sample 1', 'sample 2', 'sample 3', 'sample 4'],
            'label_index': [0, 1, 0, 1]
        })
        test_data.to_csv(os.path.join(cls.data_dir, "test_data.csv"), index=False)

    def setUp(self):
        # Create mock model
        self.mock_model = MagicMock()

        # Configure model to return logits for binary classification
        def mock_forward(*args, **kwargs):
            mock_output = MagicMock()
            mock_output.logits = torch.tensor([[1.0, -1.0], [-1.0, 1.0]])  # Alternating predictions
            return mock_output

        self.mock_model.forward = mock_forward
        self.mock_model.eval.return_value = None

        # Mock tokenizer
        def mock_tokenize(texts, **kwargs):
            return {
                'input_ids': torch.ones(len(texts), 128, dtype=torch.long),
                'attention_mask': torch.ones(len(texts), 128, dtype=torch.long)
            }

        self.mock_tokenizer = MagicMock(side_effect=mock_tokenize)

        # Patch the imports
        self.model_patch = patch(
            'src.evaluation.BertForSequenceClassification.from_pretrained',
            return_value=self.mock_model
        )
        self.tokenizer_patch = patch(
            'src.evaluation.BertTokenizer.from_pretrained',
            return_value=self.mock_tokenizer
        )

        self.model_mock = self.model_patch.start()
        self.tokenizer_mock = self.tokenizer_patch.start()

    def test_evaluation_workflow(self):
        from src.evaluation import evaluate_model

        # Test evaluation
        metrics = evaluate_model(
            data_path=os.path.join(self.data_dir, "test_data.csv"),
            model_path=self.model_dir,
            batch_size=2
        )

        # Verify metrics
        self.assertIn('accuracy', metrics)
        self.assertIn('f1', metrics)
        self.assertIn('nmi', metrics)
        self.assertIn('ari', metrics)
        self.assertIsInstance(metrics['accuracy'], float)

    def test_error_handling(self):
        from src.evaluation import evaluate_model

        with patch('pandas.read_csv', side_effect=Exception("Test error")):
            with self.assertRaises(Exception) as context:
                evaluate_model()
            self.assertIn("Test error", str(context.exception))

    def tearDown(self):
        self.model_patch.stop()
        self.tokenizer_patch.stop()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir)


if __name__ == "__main__":
    unittest.main()