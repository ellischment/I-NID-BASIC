import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertForSequenceClassification, BertTokenizer, AdamW, BertModel
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import pandas as pd
from dataclasses import dataclass
import logging
import os
from typing import Optional, Tuple, Dict

os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    filename='logs/pretrain.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


@dataclass
class PretrainConfig:
    model_name: str = 'bert-base-uncased'
    batch_size: int = 32
    learning_rate: float = 5e-5
    num_epochs: int = 3
    max_seq_length: int = 128
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    ce_weight: float = 1.0
    mlm_weight: float = 0.5
    mlm_probability: float = 0.15
    pretrained_model_path: str = 'models/pretrained'


class Pretrainer:
    def __init__(self, config: PretrainConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.tokenizer = BertTokenizer.from_pretrained(config.model_name)
        self.model = BertForSequenceClassification.from_pretrained(
            config.model_name,
            num_labels=len(pd.read_csv("data/processed/clinc_oos_gamma3/train_labeled.csv")['label_index'].unique())
        ).to(self.device)
        self._init_mlm_projection()

    def _mlm_loss(hidden_states: torch.Tensor, mlm_labels: torch.Tensor, vocab_size: int,
                 model: nn.Module, device: torch.device) -> torch.Tensor:
        """Compute MLM loss using projection head"""
        # Create projection layer if not exists
        if not hasattr(model, 'mlm_projection'):
            model.mlm_projection = nn.Linear(hidden_states.size(-1), vocab_size).to(device)
            nn.init.xavier_uniform_(model.mlm_projection.weight)

        logits = model.mlm_projection(hidden_states[mlm_labels != -100])
        return F.cross_entropy(logits, mlm_labels[mlm_labels != -100].view(-1))

    def _init_mlm_projection(self):
        """Initialize MLM projection head"""
        self.mlm_projection = nn.Linear(self.model.config.hidden_size, len(self.tokenizer)).to(self.device)
        nn.init.xavier_uniform_(self.mlm_projection.weight)

    def _mask_tokens(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare masked tokens for MLM"""
        labels = inputs.clone()
        probability_matrix = torch.full(labels.shape, self.config.mlm_probability)

        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
            for val in labels.tolist()
        ]
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100

        # 80% mask, 10% random, 10% original
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        return inputs, labels

    def _prepare_dataset(self, df: pd.DataFrame) -> TensorDataset:
        """Prepare single dataset with MLM"""
        encodings = self.tokenizer(
            df['text'].tolist(),
            truncation=True,
            padding='max_length',
            max_length=self.config.max_seq_length,
            return_tensors='pt',
            return_special_tokens_mask=True
        )

        input_ids, mlm_labels = self._mask_tokens(encodings['input_ids'])
        return TensorDataset(
            input_ids,
            encodings['attention_mask'],
            mlm_labels,
            torch.tensor(df['label_index'].tolist())
        )

    def train(self, labeled_data: pd.DataFrame) -> Tuple[BertForSequenceClassification, BertTokenizer]:
        """Full training pipeline"""
        # Prepare datasets
        train_df = labeled_data.sample(frac=0.9, random_state=42)
        val_df = labeled_data.drop(train_df.index)

        train_dataset = self._prepare_dataset(train_df)
        val_dataset = self._prepare_dataset(val_df)

        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size)

        # Optimizer setup
        optimizer = AdamW(
            list(self.model.parameters()) + list(self.mlm_projection.parameters()),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        # Training loop
        global_step = 0
        for epoch in range(self.config.num_epochs):
            self.model.train()
            epoch_loss = 0

            for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.config.num_epochs}")):
                # Prepare batch
                input_ids, attention_mask, mlm_labels, labels_cls = [x.to(self.device) for x in batch]

                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels_cls,
                    output_hidden_states=True
                )

                # Calculate losses
                loss_ce = outputs.loss
                last_hidden = outputs.hidden_states[-1]
                mlm_logits = self.mlm_projection(last_hidden[mlm_labels != -100])
                loss_mlm = F.cross_entropy(mlm_logits, mlm_labels[mlm_labels != -100].view(-1))

                total_loss = (self.config.ce_weight * loss_ce +
                              self.config.mlm_weight * loss_mlm)

                # Backward pass
                total_loss.backward()

                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        list(self.model.parameters()) + list(self.mlm_projection.parameters()),
                        self.config.max_grad_norm
                    )
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

                epoch_loss += total_loss.item()

            # Validation
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    input_ids, attention_mask, mlm_labels, labels_cls = [x.to(self.device) for x in batch]

                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels_cls,
                        output_hidden_states=True
                    )

                    last_hidden = outputs.hidden_states[-1]
                    mlm_logits = self.mlm_projection(last_hidden[mlm_labels != -100])
                    loss_mlm = F.cross_entropy(mlm_logits, mlm_labels[mlm_labels != -100].view(-1))

                    val_loss += (self.config.ce_weight * outputs.loss +
                                 self.config.mlm_weight * loss_mlm).item()

            # Save model
            os.makedirs(self.config.pretrained_model_path, exist_ok=True)
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'projection_state_dict': self.mlm_projection.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(self.config.pretrained_model_path, 'checkpoint.pt'))

            self.tokenizer.save_pretrained(self.config.pretrained_model_path)
            logging.info(
                f"Epoch {epoch + 1} | Train Loss: {epoch_loss / len(train_loader):.4f} | Val Loss: {val_loss / len(val_loader):.4f}")

        return self.model, self.tokenizer


def pretrain_model(
        labeled_data: pd.DataFrame,
        test_data: pd.DataFrame,
        config,
        tokenizer: Optional[BertTokenizer] = None,
        model: Optional[BertModel] = None
) -> Tuple[BertForSequenceClassification, BertTokenizer]:
    """
    Enhanced pretraining following ImbaNID paper with multi-task learning (CE + MLM)

    Args:
        labeled_data: DataFrame with 'text' and 'label_index' columns
        test_data: DataFrame used for validation during pretraining
        config: Configuration object with hyperparameters
        tokenizer: Optional pre-initialized tokenizer
        model: Optional pre-initialized model

    Returns:
        Tuple of (fine-tuned model, tokenizer)
    """
    # Initialize tokenizer and model if not provided
    tokenizer = tokenizer if tokenizer is not None else BertTokenizer.from_pretrained(config.model_name)
    model = model if model is not None else BertForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=len(labeled_data['label_index'].unique())
    )

    # Device setup
    device = torch.device(config.device)
    model.to(device)

    # Initialize best validation loss
    best_val_loss = float('inf')

    def mlm_loss(hidden_states: torch.Tensor, mlm_labels: torch.Tensor) -> torch.Tensor:
        """Compute MLM loss using projection head"""
        # Create projection layer if not exists
        if not hasattr(model, 'mlm_projection'):
            model.mlm_projection = nn.Linear(hidden_states.size(-1), len(tokenizer)).to(device)
            nn.init.xavier_uniform_(model.mlm_projection.weight)

        logits = model.mlm_projection(hidden_states[mlm_labels != -100])
        return F.cross_entropy(logits, mlm_labels[mlm_labels != -100].view(-1))

    # Create datasets - modified to include MLM
    def prepare_datasets(df: pd.DataFrame) -> Dict[str, torch.Tensor]:
        texts = df['text'].tolist()
        labels = df['label_index'].tolist()

        # Tokenize with MLM masks
        inputs = tokenizer(
            texts,
            truncation=True,
            padding='max_length',
            max_length=config.max_seq_length,
            return_tensors='pt',
            return_special_tokens_mask=True
        )

        # Add MLM masks - 15% probability
        inputs['input_ids'], inputs['mlm_labels'] = mask_tokens(
            inputs['input_ids'],
            tokenizer,
            mlm_probability=0.15,
            special_tokens_mask=inputs['special_tokens_mask']
        )

        if 'label_index' in df.columns:
            inputs['labels_cls'] = torch.tensor(labels)
        else:
            inputs['labels_cls'] = torch.zeros(len(df))  # Dummy labels for unlabeled data

        return inputs

    train_dataset = prepare_datasets(labeled_data)
    val_dataset = prepare_datasets(test_data)

    # Convert to TensorDataset
    train_tensor_dataset = TensorDataset(
        train_dataset['input_ids'],
        train_dataset['attention_mask'],
        train_dataset['mlm_labels'],
        train_dataset['labels_cls']
    )

    val_tensor_dataset = TensorDataset(
        val_dataset['input_ids'],
        val_dataset['attention_mask'],
        val_dataset['mlm_labels'],
        val_dataset['labels_cls']
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_tensor_dataset,
        batch_size=config.batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        val_tensor_dataset,
        batch_size=config.batch_size,
        shuffle=False
    )

    # Optimizer setup
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    # Training loop
    for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0
        epoch_loss = 0

        progress_bar = tqdm(enumerate(train_loader),
                            total=len(train_loader),
                            desc=f"Epoch {epoch + 1}/{config.num_epochs}")

        for step, batch in progress_bar:
            # Prepare batch
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            mlm_labels = batch[2].to(device)
            labels_cls = batch[3].to(device)

            # Forward passes
            with torch.set_grad_enabled(True):
                # Classification forward
                outputs_cls = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels_cls,
                    output_hidden_states=True
                )

                # Calculate losses
                loss_ce = outputs_cls.loss if torch.any(labels_cls != 0) else 0

                # MLM loss
                last_hidden_state = outputs_cls.hidden_states[-1]
                loss_mlm = mlm_loss(last_hidden_state, mlm_labels)

                total_batch_loss = (config.ce_weight * loss_ce +
                                    config.mlm_weight * loss_mlm)

            # Backward pass
            total_batch_loss.backward()

            # Gradient clipping and step
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += total_batch_loss.item()
            progress_bar.set_postfix({'loss': epoch_loss / (step + 1)})

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch[0].to(device)
                attention_mask = batch[1].to(device)
                mlm_labels = batch[2].to(device)
                labels_cls = batch[3].to(device)

                # Classification loss
                outputs_cls = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels_cls,
                    output_hidden_states=True
                )

                # MLM loss
                last_hidden_state = outputs_cls.hidden_states[-1]
                loss_mlm = mlm_loss(last_hidden_state, mlm_labels)

                val_loss += (config.ce_weight * outputs_cls.loss +
                             config.mlm_weight * loss_mlm).item()

        avg_train_loss = epoch_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        logging.info(
            f"Epoch {epoch + 1} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
        )

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.save_pretrained(config.pretrained_model_path)
            tokenizer.save_pretrained(config.pretrained_model_path)
            logging.info(f"New best model saved with val loss: {best_val_loss:.4f}")

    return model, tokenizer

def mask_tokens(inputs: torch.Tensor, tokenizer: BertTokenizer,
                mlm_probability: float = 0.15,
                special_tokens_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Prepare masked tokens for MLM (80% mask, 10% random, 10% original)"""
    labels = inputs.clone()
    probability_matrix = torch.full(labels.shape, mlm_probability)

    if special_tokens_mask is None:
        special_tokens_mask = [
            tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
            for val in labels.tolist()
        ]
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
    else:
        special_tokens_mask = special_tokens_mask.bool()

    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # Only compute loss on masked tokens

    # 80% of the time replace with [MASK]
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time replace with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    return inputs, labels


if __name__ == "__main__":
    # Test with real data
    from config import Config

    # Load sample data
    data = pd.read_csv("data/processed/clinc_oos_gamma3/train_labeled.csv")
    test_data = pd.read_csv("data/processed/clinc_oos_gamma3/test.csv")

    # Initialize and run
    cfg = Config()
    model, tokenizer = pretrain_model(data, test_data, cfg)

    print("Pretraining completed successfully!")
