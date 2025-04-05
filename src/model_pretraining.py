from dataclasses import dataclass
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertForSequenceClassification, BertTokenizer, BertForMaskedLM, AdamW
from torch.utils.data import Dataset, DataLoader, TensorDataset
from tqdm import tqdm
import pandas as pd
import logging
from typing import Optional, Tuple, Dict
from src.config import Config


# Setup logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    filename='logs/pretrain.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


@dataclass
class PretrainConfig:
    """Configuration for model pretraining with MLM and CE losses"""
    model_name: str = 'bert-base-uncased'
    batch_size: int = 32
    learning_rate: float = 5e-5
    num_epochs: int = 3
    max_seq_length: int = 128
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    weight_decay: float = 0.01
    pretrained_model_path: str = 'models/pretrained'
    text_column: str = 'text'
    label_column: str = 'intent'
    mlm_probability: float = 0.15
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0


class IntentDataset(Dataset):
    """Dataset for intent classification with MLM capability"""

    def __init__(self, df: pd.DataFrame, tokenizer: BertTokenizer, config: PretrainConfig):
        self.texts = df[config.text_column].values
        self.labels = df[config.label_column].values if config.label_column in df.columns else None
        self.tokenizer = tokenizer
        self.max_length = config.max_seq_length
        self.mlm_probability = config.mlm_probability

        assert self.labels.min() >= 0, "You have labels < 0 "
        assert self.labels.max() < config.num_classes, "labels.max() < config.num_classes"

        self.encodings = tokenizer(
            list(self.texts),
            max_length=config.max_seq_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
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

        return item


def mask_tokens(inputs: torch.Tensor, tokenizer: BertTokenizer, mlm_probability: float = 0.15) -> Tuple[
    torch.Tensor, torch.Tensor]:
    """Prepare masked tokens for MLM training (Sec 3.3 in paper)"""
    labels = inputs.clone()
    probability_matrix = torch.full(labels.shape, mlm_probability)

    # Get special tokens mask
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
        for val in labels.tolist()
    ]
    special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)

    # Apply masking
    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # Ignore non-masked tokens

    # 80% mask, 10% random, 10% original (as in BERT)
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    return inputs, labels


def pretrain_model(
        labeled_data: pd.DataFrame,
        test_data: pd.DataFrame,
        config: PretrainConfig
) -> Tuple[BertForSequenceClassification, BertTokenizer]:
    """Pretrains model with both MLM and CE losses as per paper Sec 3.3"""

    # Initialize tokenizer and models
    tokenizer = BertTokenizer.from_pretrained(config.model_name)
    num_labels = config.num_classes

    # Initialize model with both MLM and classification heads
    model = BertForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=num_labels
    ).to(config.device)

    mlm_model = BertForMaskedLM.from_pretrained(config.model_name).to(config.device)

    # Create datasets
    train_dataset = IntentDataset(labeled_data, tokenizer, config)
    val_dataset = IntentDataset(test_data, tokenizer, config)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)

    # Initialize optimizer
    optimizer = AdamW(
        list(model.parameters()) + list(mlm_model.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    # Training loop
    best_val_loss = float('inf')
    for epoch in range(config.num_epochs):
        model.train()
        mlm_model.train()
        epoch_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.num_epochs}"):
            # CE loss for labeled data
            inputs = {
                'input_ids': batch['input_ids'].to(config.device),
                'attention_mask': batch['attention_mask'].to(config.device),
                'labels': batch['labels'].to(config.device)
            }
            outputs = model(**inputs)
            ce_loss = outputs.loss

            # MLM loss for all data
            mlm_inputs, mlm_labels = mask_tokens(
                batch['input_ids'],
                tokenizer,
                config.mlm_probability
            )
            mlm_outputs = mlm_model(
                mlm_inputs.to(config.device),
                attention_mask=batch['attention_mask'].to(config.device),
                labels=mlm_labels.to(config.device)
            )
            mlm_loss = mlm_outputs.loss

            # Combined loss
            total_loss = ce_loss + mlm_loss

            # Backward pass
            total_loss.backward()

            # Gradient accumulation
            if (epoch + 1) % config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(mlm_model.parameters(), config.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

            epoch_loss += total_loss.item()

        # Validation
        model.eval()
        mlm_model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs = {
                    'input_ids': batch['input_ids'].to(config.device),
                    'attention_mask': batch['attention_mask'].to(config.device),
                    'labels': batch['labels'].to(config.device)
                }
                outputs = model(**inputs)
                val_loss += outputs.loss.item()

        # Log metrics
        avg_train_loss = epoch_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        logging.info(f"Epoch {epoch + 1} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs(config.pretrained_model_path, exist_ok=True)
            model.save_pretrained(config.pretrained_model_path)
            tokenizer.save_pretrained(config.pretrained_model_path)
            logging.info(f"Saved best model with val loss: {best_val_loss:.4f}")

    return model, tokenizer