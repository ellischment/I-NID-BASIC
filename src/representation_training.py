import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertModel, BertTokenizer
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
import logging
import os
from tqdm import tqdm
import numpy as np
from typing import Dict, List
from src.config import Config

config = Config()

class RepresentationLearner:
    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def contrastive_loss(self, z_i, z_j, temperature=0.07):
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)

        logits = (z_i @ z_j.T) / temperature
        labels = torch.arange(logits.shape[0]).to(self.device)

        loss_i = F.cross_entropy(logits, labels)
        loss_j = F.cross_entropy(logits.T, labels)
        return (loss_i + loss_j) / 2

    def class_contrastive_loss(self, features, labels, temperature=0.07):
        features = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features, features.T) / temperature

        mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float()
        self_mask = torch.eye(len(labels)).to(self.device)
        mask = mask * (1 - self_mask)

        exp_logits = torch.exp(similarity_matrix) * (1 - self_mask)
        log_prob = similarity_matrix - torch.log(exp_logits.sum(1, keepdim=True))

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1).clamp(min=1.0)
        loss = -mean_log_prob_pos.mean()
        return loss

    def noise_regularization(self, features, pseudo_labels, confidences):
        class_counts = torch.bincount(pseudo_labels)
        class_probs = class_counts.float() / class_counts.sum()

        high_conf_mask = confidences > self.config.tau_g
        clean_mask = high_conf_mask
        return features[clean_mask], pseudo_labels[clean_mask]

    def train_step(self, batch, optimizer):
        self.model.train()

        input_ids, attention_mask, labels = batch
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        labels = labels.to(self.device)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )

        embeddings = outputs.last_hidden_state[:, 0, :]

        loss_iwcl = self.contrastive_loss(embeddings, embeddings)
        loss_cwcl = self.class_contrastive_loss(embeddings, labels)

        loss = self.config.omega * (loss_iwcl + loss_cwcl) + (1 - self.config.omega) * F.cross_entropy(embeddings,
                                                                                                     labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return {
            'total_loss': loss.item(),
            'iwcl_loss': loss_iwcl.item(),
            'cwcl_loss': loss_cwcl.item()
        }

    def train(self, labeled_data, unlabeled_data, num_epochs=10):
        train_dataset = self._prepare_dataset(labeled_data, unlabeled_data)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        for epoch in range(num_epochs):
            epoch_losses = {'total_loss': 0, 'iwcl_loss': 0, 'cwcl_loss': 0}

            for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
                losses = self.train_step(batch, optimizer)

                for k in epoch_losses:
                    epoch_losses[k] += losses[k]

            for k in epoch_losses:
                epoch_losses[k] /= len(train_loader)

            logging.info(
                f"Epoch {epoch + 1} - "
                f"Total Loss: {epoch_losses['total_loss']:.4f}, "
                f"IWCL Loss: {epoch_losses['iwcl_loss']:.4f}, "
                f"CWCL Loss: {epoch_losses['cwcl_loss']:.4f}"
            )

    def _prepare_dataset(self, labeled_df, unlabeled_df):
        texts = labeled_df['text'].tolist() + unlabeled_df['text'].tolist()
        encodings = self.tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=self.config.max_seq_length,
            return_tensors='pt'
        )

        labels = torch.cat([
            torch.tensor(labeled_df['label'].values),
            torch.tensor(unlabeled_df['pseudo_label'].values)
        ])

        return TensorDataset(
            encodings['input_ids'],
            encodings['attention_mask'],
            labels
        )

    def evaluate(self, test_loader):
        self.model.eval()
        all_embeddings = []
        all_labels = []

        with torch.no_grad():
            for batch in test_loader:
                input_ids, attention_mask, labels = batch
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                embeddings = outputs.last_hidden_state[:, 0, :].cpu()

                all_embeddings.append(embeddings)
                all_labels.append(labels)

        embeddings = torch.cat(all_embeddings)
        labels = torch.cat(all_labels)

        preds = embeddings.argmax(dim=1)
        acc = (preds == labels).float().mean().item()
        nmi = normalized_mutual_info_score(labels.numpy(), preds.numpy())
        ari = adjusted_rand_score(labels.numpy(), preds.numpy())

        return {
            'accuracy': acc,
            'nmi': nmi,
            'ari': ari
        }