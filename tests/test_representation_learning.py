import unittest
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader, TensorDataset
import os
import logging
import tempfile
import shutil


class TestTrainModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Создаем временные директории для данных и логов
        cls.temp_data_dir = tempfile.mkdtemp()
        cls.temp_logs_dir = tempfile.mkdtemp()
        cls.temp_models_dir = tempfile.mkdtemp()

        # Создаем фиктивные данные для тестирования
        cls.labeled_data = pd.DataFrame({
            'text': ['This is a positive sentence', 'This is a negative sentence'],
            'pseudo_labels': [1, 0]
        })
        cls.unlabeled_data = pd.DataFrame({
            'text': ['This is another positive sentence', 'This is another negative sentence'],
            'pseudo_labels': [1, 0]
        })
        cls.test_data = pd.DataFrame({
            'text': ['This is a test sentence'],
            'pseudo_labels': [1]
        })

        # Сохраняем фиктивные данные в файлы
        os.makedirs(os.path.join(cls.temp_data_dir, "processed"), exist_ok=True)
        cls.labeled_data.to_csv(os.path.join(cls.temp_data_dir, "processed/labeled_data.csv"), index=False)
        cls.unlabeled_data.to_csv(os.path.join(cls.temp_data_dir, "processed/unlabeled_data_with_pseudo_labels.csv"), index=False)
        cls.test_data.to_csv(os.path.join(cls.temp_data_dir, "processed/test_data.csv"), index=False)

        # Инициализация логгера
        os.makedirs(cls.temp_logs_dir, exist_ok=True)
        logging.basicConfig(filename=os.path.join(cls.temp_logs_dir, "imbanid.log"), level=logging.INFO)

    def test_data_loading(self):
        # Проверка загрузки данных
        labeled_data = pd.read_csv(os.path.join(self.temp_data_dir, "processed/labeled_data.csv"))
        unlabeled_data = pd.read_csv(os.path.join(self.temp_data_dir, "processed/unlabeled_data_with_pseudo_labels.csv"))
        test_data = pd.read_csv(os.path.join(self.temp_data_dir, "processed/test_data.csv"))

        self.assertEqual(len(labeled_data), 2)
        self.assertEqual(len(unlabeled_data), 2)
        self.assertEqual(len(test_data), 1)

    def test_tokenization(self):
        # Проверка токенизации
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        train_encodings = tokenizer(list(self.labeled_data['text']), truncation=True, padding=True, max_length=128)

        self.assertIn('input_ids', train_encodings)
        self.assertIn('attention_mask', train_encodings)
        self.assertEqual(len(train_encodings['input_ids']), 2)

    def test_dataloader_creation(self):
        # Проверка создания DataLoader
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        train_encodings = tokenizer(list(self.labeled_data['text']), truncation=True, padding=True, max_length=128)
        train_labels = torch.tensor(self.labeled_data['pseudo_labels'].tolist())

        train_dataset = TensorDataset(
            torch.tensor(train_encodings['input_ids']),
            torch.tensor(train_encodings['attention_mask']),
            train_labels
        )
        train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

        self.assertEqual(len(train_loader), 1)  # 2 samples / batch_size=2 = 1 batch

    def test_model_training(self):
        # Проверка обучения модели
        model = BertModel.from_pretrained('bert-base-uncased')
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model.to(device)

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        train_encodings = tokenizer(list(self.labeled_data['text']), truncation=True, padding=True, max_length=128)
        train_labels = torch.tensor(self.labeled_data['pseudo_labels'].tolist())

        train_dataset = TensorDataset(
            torch.tensor(train_encodings['input_ids']),
            torch.tensor(train_encodings['attention_mask']),
            train_labels
        )
        train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            loss = torch.nn.CrossEntropyLoss()(outputs.last_hidden_state[:, 0, :], labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        self.assertGreater(total_loss, 0)  # Проверяем, что loss был вычислен

    def test_model_saving(self):
        # Проверка сохранения модели
        model = BertModel.from_pretrained('bert-base-uncased')
        os.makedirs(os.path.join(self.temp_models_dir, "finetuned"), exist_ok=True)
        model.save_pretrained(os.path.join(self.temp_models_dir, "finetuned/bert_finetuned"))

        self.assertTrue(os.path.exists(os.path.join(self.temp_models_dir, "finetuned/bert_finetuned")))

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