import unittest
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader, TensorDataset
import os
import logging
import tempfile
import shutil
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, normalized_mutual_info_score, adjusted_rand_score


class TestEvaluateModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Создаем временные директории для данных, логов и моделей
        cls.temp_data_dir = tempfile.mkdtemp()
        cls.temp_logs_dir = tempfile.mkdtemp()
        cls.temp_models_dir = tempfile.mkdtemp()

        # Создаем фиктивные данные для тестирования
        cls.test_data = pd.DataFrame({
            'text': [
                'This is a positive sentence',  # label_index 1
                'This is a negative sentence',  # label_index 0
                'This is another positive sentence',  # label_index 1
                'This is another negative sentence'  # label_index 0
            ],
            'label_index': [1, 0, 1, 0]
        })

        # Сохраняем фиктивные данные в файл
        os.makedirs(os.path.join(cls.temp_data_dir, "processed"), exist_ok=True)
        cls.test_data.to_csv(os.path.join(cls.temp_data_dir, "processed/test_data.csv"), index=False)

        # Инициализация логгера
        os.makedirs(cls.temp_logs_dir, exist_ok=True)
        logging.basicConfig(filename=os.path.join(cls.temp_logs_dir, "imbanid.log"), level=logging.INFO)

        # Сохраняем фиктивную модель
        os.makedirs(os.path.join(cls.temp_models_dir, "finetuned"), exist_ok=True)
        model = BertModel.from_pretrained('bert-base-uncased')
        model.save_pretrained(os.path.join(cls.temp_models_dir, "finetuned/bert_finetuned"))

    def test_data_loading(self):
        # Проверка загрузки тестовых данных
        test_data = pd.read_csv(os.path.join(self.temp_data_dir, "processed/test_data.csv"))
        self.assertEqual(len(test_data), 4)
        self.assertListEqual(test_data['label_index'].tolist(), [1, 0, 1, 0])

    def test_tokenization(self):
        # Проверка токенизации
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        test_encodings = tokenizer(list(self.test_data['text']), truncation=True, padding=True, max_length=128)

        self.assertIn('input_ids', test_encodings)
        self.assertIn('attention_mask', test_encodings)
        self.assertEqual(len(test_encodings['input_ids']), 4)

    def test_dataloader_creation(self):
        # Проверка создания DataLoader
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        test_encodings = tokenizer(list(self.test_data['text']), truncation=True, padding=True, max_length=128)
        test_labels = torch.tensor(self.test_data['label_index'].tolist())

        test_dataset = TensorDataset(
            torch.tensor(test_encodings['input_ids']),
            torch.tensor(test_encodings['attention_mask']),
            test_labels
        )
        test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

        self.assertEqual(len(test_loader), 2)  # 4 samples / batch_size=2 = 2 batches

    def test_model_predictions(self):
        # Проверка предсказаний модели
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained(os.path.join(self.temp_models_dir, "finetuned/bert_finetuned"))
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model.to(device)

        test_encodings = tokenizer(list(self.test_data['text']), truncation=True, padding=True, max_length=128)
        test_labels = torch.tensor(self.test_data['label_index'].tolist())

        test_dataset = TensorDataset(
            torch.tensor(test_encodings['input_ids']),
            torch.tensor(test_encodings['attention_mask']),
            test_labels
        )
        test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

        model.eval()
        predictions = []
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch[0].to(device)
                attention_mask = batch[1].to(device)
                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                predicted_labels = np.argmax(logits, axis=1)
                predictions.extend(predicted_labels)

        self.assertEqual(len(predictions), 4)  # Проверяем, что предсказания сделаны для всех примеров

    def test_metrics_calculation(self):
        # Проверка вычисления метрик
        true_labels = np.array([1, 0, 1, 0])
        predictions = np.array([1, 0, 1, 0])  # Идеальные предсказания

        accuracy = accuracy_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions, average='weighted')
        nmi = normalized_mutual_info_score(true_labels, predictions)
        ari = adjusted_rand_score(true_labels, predictions)

        self.assertEqual(accuracy, 1.0)  # Accuracy должна быть 1.0
        self.assertEqual(f1, 1.0)  # F1-score должна быть 1.0
        self.assertEqual(nmi, 1.0)  # NMI должна быть 1.0
        self.assertEqual(ari, 1.0)  # ARI должна быть 1.0

    @classmethod
    def tearDownClass(cls):
        # Закрываем логгер
        logging.shutdown()

        # Удаляем временные директории
        shutil.rmtree(cls.temp_data_dir)
        shutil.rmtree(cls.temp_logs_dir)
        shutil.rmtree(cls.temp_models_dir)


if __name__ == "__main__":
    unittest.main()