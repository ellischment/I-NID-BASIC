import unittest
import numpy as np
import pandas as pd
import torch
from unittest.mock import patch, MagicMock
import os
from transformers import BertTokenizer, BertModel


class TestPseudoLabelGeneration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_data = pd.DataFrame({
            'text': [
                'check my balance',  # 0 (banking)
                'transfer money',  # 0 (banking)
                'weather forecast',  # 1 (weather)
                'play jazz music'  # 2 (music)
            ],
            'true_label': [0, 0, 1, 2]  # Для валидации
        })
        os.makedirs("data/processed", exist_ok=True)
        cls.test_data.to_csv("data/processed/unlabeled.csv", index=False)

        # Создаем моки для модели и токенизатора
        cls.mock_tokenizer = MagicMock(spec=BertTokenizer)
        cls.mock_model = MagicMock(spec=BertModel)

        # Настраиваем поведение моков
        cls.mock_tokenizer.return_value = {
            'input_ids': torch.tensor([[1] * 128]),
            'attention_mask': torch.tensor([[1] * 128])
        }

        mock_output = MagicMock()
        mock_output.last_hidden_state = torch.rand(1, 128, 768)  # [batch_size, seq_len, hidden_size]
        cls.mock_model.return_value = mock_output

    def test_plg_initialization(self):
        from src.pseudo_labeling import PseudoLabelGenerator
        plg = PseudoLabelGenerator(tokenizer=self.mock_tokenizer, model=self.mock_model)
        self.assertIsNotNone(plg)

    def test_embedding_generation(self):
        from src.pseudo_labeling import PseudoLabelGenerator

        # Настраиваем мок для возврата эмбеддингов
        mock_output = MagicMock()
        mock_output.last_hidden_state = torch.rand(4, 128, 768)  # 4 samples
        self.mock_model.return_value = mock_output

        plg = PseudoLabelGenerator(tokenizer=self.mock_tokenizer, model=self.mock_model)
        embeddings = plg.get_embeddings(self.test_data['text'].tolist())
        self.assertEqual(embeddings.shape, (4, 768))  # 4 samples, 768 dim

    def test_rot_pseudo_labeling(self):
        from src.pseudo_labeling import PseudoLabelGenerator
        plg = PseudoLabelGenerator(tokenizer=self.mock_tokenizer, model=self.mock_model)

        test_embeddings = np.random.rand(4, 768)  # 4 samples, 768 dim

        with patch('ot.sinkhorn', return_value=np.array([
            [0.9, 0.05, 0.05],  # Высокая уверенность в классе 0
            [0.8, 0.1, 0.1],  # Класс 0
            [0.1, 0.8, 0.1],  # Класс 1
            [0.1, 0.1, 0.8]  # Класс 2
        ])) as mock_sinkhorn:
            labels = plg.generate_pseudo_labels(test_embeddings, reg=0.1)
            mock_sinkhorn.assert_called_once()
            self.assertEqual(labels.tolist(), [0, 0, 1, 2])

    def test_confidence_threshold(self):
        from src.pseudo_labeling import PseudoLabelGenerator
        plg = PseudoLabelGenerator(tokenizer=self.mock_tokenizer, model=self.mock_model)

        # Матрица с одним примером с низкой уверенностью
        Q = np.array([
            [0.9, 0.05, 0.05],
            [0.4, 0.3, 0.3]  # Низкая уверенность
        ])

        with patch('ot.sinkhorn', return_value=Q):
            labels = plg.generate_pseudo_labels(
                embeddings=np.random.rand(2, 768),
                confidence_thresh=0.5
            )
            self.assertEqual(labels.tolist(), [0, -1])  # -1 для "unknown"

    def test_full_pipeline(self):
        from src.pseudo_labeling import PseudoLabelGenerator

        # Настраиваем мок для возврата эмбеддингов
        mock_output = MagicMock()
        mock_output.last_hidden_state = torch.rand(4, 128, 768)  # 4 samples
        self.mock_model.return_value = mock_output

        with patch('ot.sinkhorn', return_value=np.eye(4)), \
                patch('pandas.DataFrame.to_csv') as mock_csv:
            plg = PseudoLabelGenerator(tokenizer=self.mock_tokenizer, model=self.mock_model)
            result = plg.process(
                "data/processed/unlabeled.csv",
                "data/processed/labeled.csv"
            )

            self.assertEqual(result.shape[0], 4)  # 4 строки
            self.assertIn('pseudo_labels', result.columns)
            mock_csv.assert_called_once()

    @classmethod
    def tearDownClass(cls):
        if os.path.exists("data/processed/unlabeled.csv"):
            os.remove("data/processed/unlabeled.csv")
        if os.path.exists("data/processed/labeled.csv"):
            os.remove("data/processed/labeled.csv")


if __name__ == '__main__':
    unittest.main()