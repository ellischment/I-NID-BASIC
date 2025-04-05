import torch
import torch.nn as nn
from transformers import BertForSequenceClassification, BertTokenizer, AdamW
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from tqdm import tqdm
import logging
import os
from typing import Optional


@dataclass
class PretrainConfig:
    """Configuration for model pretraining stage"""
    model_name: str = 'bert-base-uncased'
    batch_size: int = 32  # Reduced from paper's 512 for Colab
    learning_rate: float = 5e-5
    num_epochs: int = 3  # Reduced from paper's 10
    max_seq_length: int = 64  # Reduced from 128
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    weight_decay: float = 0.01
    pretrained_model_path: str = 'models/pretrained'

    # ImbaNID specific additions
    projection_dim: int = 256  # For contrastive learning
    temperature: float = 0.07
    lambda1: float = 0.05  # ROT loss weights
    lambda2: float = 2.0
    omega: float = 0.5  # Contrastive loss weight


class IntentDataset(Dataset):
    """Dataset for intent classification with MLM support"""

    def __init__(self, df, tokenizer, config):
        self.texts = df[config.text_column].values
        self.labels = df[config.label_column].values if config.label_column in df.columns else None
        self.tokenizer = tokenizer
        self.max_length = config.max_seq_length
        self.mlm_probability = 0.15  # For MLM pretraining

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }

        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)

        # Add MLM masking for pretraining
        input_ids = item['input_ids'].clone()
        probability_matrix = torch.full(input_ids.shape, self.mlm_probability)
        special_tokens_mask = self.tokenizer.get_special_tokens_mask(input_ids.tolist(),
                                                                     already_has_special_tokens=True)
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()

        # 80% mask, 10% random, 10% original
        indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.mask_token_id

        indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), input_ids.shape, dtype=torch.long)
        input_ids[indices_random] = random_words[indices_random]

        item['masked_input_ids'] = input_ids
        item['masked_labels'] = item['input_ids'].clone()
        item['masked_labels'][~masked_indices] = -100  # Only compute loss on masked tokens

        return item


def pretrain_model(labeled_data, unlabeled_data, config):
    """Enhanced pretraining with MLM and intent classification"""
    # Validate input
    required_columns = {config.text_column}
    if config.label_column:
        required_columns.add(config.label_column)

    missing = required_columns - set(labeled_data.columns)
    if missing:
        raise ValueError(f"Missing columns in labeled data: {missing}")

    # Initialize model with projection head
    model = BertForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=len(labeled_data[config.label_column].unique())
    ).to(config.device)

    # Add projection head for contrastive learning
    model.projection = nn.Linear(768, config.projection_dim).to(config.device)
    tokenizer = BertTokenizer.from_pretrained(config.model_name)

    # Create datasets
    labeled_dataset = IntentDataset(labeled_data, tokenizer, config)
    unlabeled_dataset = IntentDataset(unlabeled_data.drop(columns=[config.label_column], errors='ignore'), tokenizer,
                                      config)

    # Combined dataloader
    combined_dataset = torch.utils.data.ConcatDataset([labeled_dataset, unlabeled_dataset])
    train_loader = DataLoader(combined_dataset, batch_size=config.batch_size, shuffle=True)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    # Training loop
    for epoch in range(config.num_epochs):
        model.train()
        epoch_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.num_epochs}"):
            # Move batch to device
            batch = {k: v.to(config.device) for k, v in batch.items()}

            # Supervised loss (labeled data)
            if 'labels' in batch:
                outputs = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels']
                )
                cls_loss = outputs.loss
            else:
                cls_loss = 0

            # MLM loss (all data)
            mlm_outputs = model(
                input_ids=batch['masked_input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['masked_labels']
            )
            mlm_loss = mlm_outputs.loss

            # Contrastive loss preparation
            with torch.no_grad():
                embeddings = model.bert(
                    batch['input_ids'],
                    attention_mask=batch['attention_mask']
                ).last_hidden_state[:, 0, :]
                projected = model.projection(embeddings)

            # Total loss
            loss = cls_loss + mlm_loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()

        logging.info(f"Epoch {epoch + 1} - Loss: {epoch_loss / len(train_loader):.4f}")

    # Save model
    os.makedirs(config.pretrained_model_path, exist_ok=True)
    model.save_pretrained(config.pretrained_model_path)
    tokenizer.save_pretrained(config.pretrained_model_path)

    return model, tokenizer
