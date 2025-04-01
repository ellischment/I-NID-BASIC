import unittest
import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from unittest.mock import patch, MagicMock
import os
import logging
from sklearn.metrics import accuracy_score

# Импортируем функцию, которую будем тестировать
from src.pseudo_labeling import generate_pseudo_labels

class TestGeneratePseudoLabels(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Создаем фиктивные данные для тестирования
        cls.balanced_data = pd.DataFrame({
            'text': [
                'I want to book a flight',  # Класс 0
                'Can you help me with my booking?',  # Класс 0
                'I need to cancel my reservation',  # Класс 1
                'How do I cancel my booking?',  # Класс 1
                'What is the status of my order?',  # Класс 2
                'Can you check my order status?'  # Класс 2
            ],
            'true_labels': [0, 0, 1, 1, 2, 2]  # Истинные метки
        })

        cls.imbalanced_data = pd.DataFrame({
            'text': [
                'I want to book a flight',  # Класс 0 (много примеров)
                'Can you help me with my booking?',  # Класс 0
                'I need to cancel my reservation',  # Класс 1 (мало примеров)
                'What is the status of my order?',  # Класс 2 (мало примеров)
                'Can you check my order status?'  # Класс 2
            ],
            'true_labels': [0, 0, 1, 2, 2]  # Истинные метки
        })

        cls.noisy_data = pd.DataFrame({
            'text': [
                'I want to book a flight',  # Класс 0
                'Can you help me with my booking?',  # Класс 0
                'I need to cancel my reservation',  # Класс 1
                'How do I cancel my booking?',  # Класс 1
                'What is the status of my order?',  # Класс 2
                'Can you check my order status?',  # Класс 2
                'I want to order a pizza'  # Шум (не относится ни к одному классу)
            ],
            'true_labels': [0, 0, 1, 1, 2, 2, -1]  # Истинные метки (-1 для шума)
        })

        # Создаем фиктивные директории и файлы
        os.makedirs("data/processed", exist_ok=True)
        os.makedirs("logs", exist_ok=True)

    def tearDown(self):
        # Закрываем логгер
        logging.shutdown()

        # Очищаем файлы после каждого теста
        if os.path.exists("data/processed/unlabeled_data.csv"):
            os.remove("data/processed/unlabeled_data.csv")
        if os.path.exists("data/processed/unlabeled_data_with_pseudo_labels.csv"):
            os.remove("data/processed/unlabeled_data_with_pseudo_labels.csv")
        if os.path.exists("logs/imbanid.log"):
            os.remove("logs/imbanid.log")

    @patch('src.pseudo_labeling.BertTokenizer.from_pretrained')
    @patch('src.pseudo_labeling.BertModel.from_pretrained')
    @patch('src.pseudo_labeling.torch.device')
    @patch('src.pseudo_labeling.DataLoader')
    @patch('src.pseudo_labeling.ot.sinkhorn')
    def test_generate_pseudo_labels_balanced_data(self, mock_sinkhorn, mock_dataloader, mock_device, mock_bert_model, mock_tokenizer):
        # Сохраняем сбалансированные данные
        self.balanced_data.to_csv("data/processed/unlabeled_data.csv", index=False)

        # Мокируем возвращаемые значения
        mock_tokenizer.return_value = MagicMock()
        mock_bert_model.return_value = MagicMock()
        mock_device.return_value = 'cpu'
        mock_dataloader.return_value = [
            (torch.tensor([[1, 2, 3]]), torch.tensor([[1, 1, 1]]))
        ]
        mock_sinkhorn.return_value = np.array([
            [0.9, 0.05, 0.05],  # Класс 0
            [0.9, 0.05, 0.05],  # Класс 0
            [0.05, 0.9, 0.05],  # Класс 1
            [0.05, 0.9, 0.05],  # Класс 1
            [0.05, 0.05, 0.9],  # Класс 2
            [0.05, 0.05, 0.9]   # Класс 2
        ])

        # Мокируем предсказания модели
        mock_bert_model.return_value.eval.return_value = None
        mock_bert_model.return_value.return_value = MagicMock(
            last_hidden_state=torch.tensor([[[0.1, 0.2], [0.3, 0.4]]])
        )

        # Вызываем функцию
        generate_pseudo_labels()

        # Проверяем, что файл с псевдо-метками был создан
        self.assertTrue(os.path.exists("data/processed/unlabeled_data_with_pseudo_labels.csv"))

        # Проверяем, что псевдо-метки были добавлены в данные
        saved_data = pd.read_csv("data/processed/unlabeled_data_with_pseudo_labels.csv")
        self.assertIn('pseudo_labels', saved_data.columns)

        # Проверяем, что псевдо-метки близки к истинным меткам
        pseudo_labels = saved_data['pseudo_labels'].values
        true_labels = saved_data['true_labels'].values
        accuracy = accuracy_score(true_labels, pseudo_labels)
        self.assertGreaterEqual(accuracy, 0.9)  # Ожидаем высокую точность

    @patch('src.pseudo_labeling.BertTokenizer.from_pretrained')
    @patch('src.pseudo_labeling.BertModel.from_pretrained')
    @patch('src.pseudo_labeling.torch.device')
    @patch('src.pseudo_labeling.DataLoader')
    @patch('src.pseudo_labeling.ot.sinkhorn')
    def test_generate_pseudo_labels_imbalanced_data(self, mock_sinkhorn, mock_dataloader, mock_device, mock_bert_model, mock_tokenizer):
        # Сохраняем несбалансированные данные
        self.imbalanced_data.to_csv("data/processed/unlabeled_data.csv", index=False)

        # Мокируем возвращаемые значения
        mock_tokenizer.return_value = MagicMock()
        mock_bert_model.return_value = MagicMock()
        mock_device.return_value = 'cpu'
        mock_dataloader.return_value = [
            (torch.tensor([[1, 2, 3]]), torch.tensor([[1, 1, 1]]))
        ]
        mock_sinkhorn.return_value = np.array([
            [0.9, 0.05, 0.05],  # Класс 0
            [0.9, 0.05, 0.05],  # Класс 0
            [0.05, 0.9, 0.05],  # Класс 1
            [0.05, 0.05, 0.9],  # Класс 2
            [0.05, 0.05, 0.9]   # Класс 2
        ])

        # Мокируем предсказания модели
        mock_bert_model.return_value.eval.return_value = None
        mock_bert_model.return_value.return_value = MagicMock(
            last_hidden_state=torch.tensor([[[0.1, 0.2], [0.3, 0.4]]])
        )

        # Вызываем функцию
        generate_pseudo_labels()

        # Проверяем, что файл с псевдо-метками был создан
        self.assertTrue(os.path.exists("data/processed/unlabeled_data_with_pseudo_labels.csv"))

        # Проверяем, что псевдо-метки были добавлены в данные
        saved_data = pd.read_csv("data/processed/unlabeled_data_with_pseudo_labels.csv")
        self.assertIn('pseudo_labels', saved_data.columns)

        # Проверяем, что псевдо-метки близки к истинным меткам
        pseudo_labels = saved_data['pseudo_labels'].values
        true_labels = saved_data['true_labels'].values
        accuracy = accuracy_score(true_labels, pseudo_labels)
        self.assertGreaterEqual(accuracy, 0.8)  # Ожидаем хорошую точность, но ниже, чем для сбалансированных данных

    @patch('src.pseudo_labeling.BertTokenizer.from_pretrained')
    @patch('src.pseudo_labeling.BertModel.from_pretrained')
    @patch('src.pseudo_labeling.torch.device')
    @patch('src.pseudo_labeling.DataLoader')
    @patch('src.pseudo_labeling.ot.sinkhorn')
    def test_generate_pseudo_labels_noisy_data(self, mock_sinkhorn, mock_dataloader, mock_device, mock_bert_model,
                                               mock_tokenizer):
        # Сохраняем данные с шумом
        self.noisy_data.to_csv("data/processed/unlabeled_data.csv", index=False)

        # Мокируем возвращаемые значения
        mock_tokenizer.return_value = MagicMock()
        mock_bert_model.return_value = MagicMock()
        mock_device.return_value = 'cpu'
        mock_dataloader.return_value = [
            (torch.tensor([[1, 2, 3]]), torch.tensor([[1, 1, 1]]))
        ]
        mock_sinkhorn.return_value = np.array([
            [0.9, 0.05, 0.05],  # Класс 0
            [0.9, 0.05, 0.05],  # Класс 0
            [0.05, 0.9, 0.05],  # Класс 1
            [0.05, 0.9, 0.05],  # Класс 1
            [0.05, 0.05, 0.9],  # Класс 2
            [0.05, 0.05, 0.9],  # Класс 2
            [0.33, 0.33, 0.34]  # Шум (не классифицирован)
        ])

        # Мокируем предсказания модели
        mock_bert_model.return_value.eval.return_value = None
        mock_bert_model.return_value.return_value = MagicMock(
            last_hidden_state=torch.tensor([[[0.1, 0.2], [0.3, 0.4]]])
        )

        # Вызываем функцию
        generate_pseudo_labels()

        # Проверяем, что файл с псевдо-метками был создан
        self.assertTrue(os.path.exists("data/processed/unlabeled_data_with_pseudo_labels.csv"))

        # Проверяем, что псевдо-метки были добавлены в данные
        saved_data = pd.read_csv("data/processed/unlabeled_data_with_pseudo_labels.csv")
        self.assertIn('pseudo_labels', saved_data.columns)

        # Проверяем, что шумный пример не был классифицирован
        pseudo_labels = saved_data['pseudo_labels'].values
        true_labels = saved_data['true_labels'].values
        noise_index = np.where(true_labels == -1)[0]

        # Шумный пример не должен быть классифицирован как 0, 1 или 2
        self.assertNotIn(pseudo_labels[noise_index], [0, 1, 2])

if __name__ == '__main__':
    unittest.main()