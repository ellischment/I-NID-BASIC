from transformers import BertTokenizer, BertModel, AdamW
import torch
from torch.utils.data import DataLoader, TensorDataset
import logging
import pandas as pd
import os
from typing import Optional, Tuple
from tqdm import tqdm

os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    filename='logs/pretrain.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class Pretrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _prepare_datasets(self, labeled_data: pd.DataFrame) -> Tuple[DataLoader, DataLoader]:
        """Prepare training and validation datasets"""
        # Split data (90% train, 10% val)
        train_df = labeled_data.sample(frac=0.9, random_state=42)
        val_df = labeled_data.drop(train_df.index)

        # Initialize tokenizer
        tokenizer = BertTokenizer.from_pretrained(self.config.model_name)

        # Tokenization function
        def tokenize(texts):
            return tokenizer(
                texts.tolist(),
                truncation=True,
                padding='max_length',
                max_length=self.config.max_seq_length,
                return_tensors='pt'
            )

        # Tokenize data
        train_encodings = tokenize(train_df['text'])
        val_encodings = tokenize(val_df['text'])

        # Create datasets
        train_dataset = TensorDataset(
            train_encodings['input_ids'],
            train_encodings['attention_mask'],
            torch.tensor(train_df['label_index'].tolist())
        )
        val_dataset = TensorDataset(
            val_encodings['input_ids'],
            val_encodings['attention_mask'],
            torch.tensor(val_df['label_index'].tolist())
        )

        return train_dataset, val_dataset, tokenizer

    def train(self, labeled_data: pd.DataFrame) -> Tuple[BertModel, BertTokenizer]:
        """Full pretraining pipeline"""
        logging.info("Starting pretraining process")

        # Prepare data
        train_dataset, val_dataset, tokenizer = self._prepare_datasets(labeled_data)

        # Initialize model
        model = BertModel.from_pretrained(self.config.model_name)
        model.to(self.device)

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False
        )

        # Training setup
        optimizer = AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        criterion = torch.nn.CrossEntropyLoss()

        # Training loop
        for epoch in range(self.config.num_epochs):
            model.train()
            total_loss = 0

            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.config.num_epochs}")
            for batch in progress_bar:
                optimizer.zero_grad()

                inputs = {
                    'input_ids': batch[0].to(self.device),
                    'attention_mask': batch[1].to(self.device)
                }
                labels = batch[2].to(self.device)

                outputs = model(**inputs)
                cls_embeddings = outputs.last_hidden_state[:, 0, :]
                loss = criterion(cls_embeddings, labels)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                progress_bar.set_postfix({'loss': loss.item()})

            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    inputs = {
                        'input_ids': batch[0].to(self.device),
                        'attention_mask': batch[1].to(self.device)
                    }
                    labels = batch[2].to(self.device)

                    outputs = model(**inputs)
                    cls_embeddings = outputs.last_hidden_state[:, 0, :]
                    val_loss += criterion(cls_embeddings, labels).item()

            # Logging
            avg_train_loss = total_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            logging.info(
                f"Epoch {epoch + 1} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
            )

        # Save model
        os.makedirs(self.config.pretrained_model_path, exist_ok=True)
        model.save_pretrained(self.config.pretrained_model_path)
        tokenizer.save_pretrained(self.config.pretrained_model_path)
        logging.info(f"Model saved to {self.config.pretrained_model_path}")

        return model, tokenizer


def pretrain_model(
        labeled_data: pd.DataFrame,
        test_data: pd.DataFrame,
        config,
        tokenizer: Optional[BertTokenizer] = None,
        model: Optional[BertModel] = None
) -> Tuple[BertModel, BertTokenizer]:
    """
    Pretrain model wrapper for compatibility with existing code

    Args:
        labeled_data: Training DataFrame with 'text' and 'label_index'
        test_data: Test DataFrame (unused in pretraining)
        config: Configuration object
        tokenizer: Optional pre-loaded tokenizer
        model: Optional pre-loaded model

    Returns:
        Tuple of (model, tokenizer)
    """
    pretrainer = Pretrainer(config)
    return pretrainer.train(labeled_data)


if __name__ == "__main__":
    # Test with real data
    from src.config import Config

    # Load sample data
    data = pd.read_csv("data/processed/clinc_oos_gamma3/train_labeled.csv")

    # Initialize and run
    cfg = Config()
    model, tokenizer = pretrain_model(data, None, cfg)

    print("Pretraining completed successfully!")