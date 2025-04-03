import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertModel, BertTokenizer, AdamW
import logging
import os
from src.config import Config

os.makedirs('logs', exist_ok=True)
logging.basicConfig(filename='logs/representation_learning.log', level=logging.INFO)


class RepresentationLearner:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = BertModel.from_pretrained(config.pretrained_model_path)
        self.tokenizer = BertTokenizer.from_pretrained(config.tokenizer_name)
        self.model.to(self.device)

    def _prepare_data(self, texts, labels=None):
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding='max_length',  # Changed to ensure fixed length
            max_length=self.config.max_seq_length,
            return_tensors='pt'
        )
        if labels is not None:
            return TensorDataset(
                encodings['input_ids'],
                encodings['attention_mask'],
                torch.tensor(labels)
            )
        return TensorDataset(
            encodings['input_ids'],
            encodings['attention_mask']
        )

    def train(self, labeled_data, pseudo_labeled_data):
        try:
            # Combine data
            all_texts = labeled_data['text'].tolist() + pseudo_labeled_data['text'].tolist()
            all_labels = labeled_data['label'].tolist() + pseudo_labeled_data['pseudo_label'].tolist()

            # Prepare datasets
            train_dataset = self._prepare_data(all_texts, all_labels)
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True
            )

            # Initialize optimizer and loss
            optimizer = AdamW(self.model.parameters(), lr=self.config.learning_rate)
            criterion = nn.CrossEntropyLoss()

            # Training loop
            self.model.train()
            for epoch in range(self.config.num_epochs):
                total_loss = 0
                for batch in train_loader:
                    optimizer.zero_grad()
                    input_ids = batch[0].to(self.device)
                    attention_mask = batch[1].to(self.device)
                    labels = batch[2].to(self.device)

                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )

                    # Use CLS token for classification
                    cls_embeddings = outputs.last_hidden_state[:, 0, :]
                    loss = criterion(cls_embeddings, labels)

                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                avg_loss = total_loss / len(train_loader)
                logging.info(f"Epoch {epoch + 1}/{self.config.num_epochs}, Loss: {avg_loss:.4f}")

            # Save model
            os.makedirs(self.config.finetuned_model_path, exist_ok=True)
            self.model.save_pretrained(self.config.finetuned_model_path)
            logging.info("Model training completed and saved")


        except Exception as e:
            logging.error(f"Error during training: {str(e)}")
            raise

    def get_embeddings(self, texts):
        self.model.eval()
        dataset = self._prepare_data(texts)
        loader = DataLoader(dataset, batch_size=self.config.batch_size)

        embeddings = []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch[0].to(self.device)
                attention_mask = batch[1].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                cls_embeddings = outputs.last_hidden_state[:, 0, :]
                embeddings.append(cls_embeddings.cpu())

        return torch.cat(embeddings, dim=0)