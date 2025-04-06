import numpy as np
import ot
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
import pandas as pd
import logging
import os
from typing import Optional
from src.config import Config

config = Config()

class PseudoLabelGenerator:
    def __init__(self, model, tokenizer, lambda1=0.05, lambda2=2.0):
        self.encoder = model.bert if hasattr(model, 'bert') else model.base_model
        self.tokenizer = tokenizer
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder.to(self.device)

    def get_embeddings(self, texts, batch_size=32):
        self.encoder.eval()
        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=config.max_seq_length,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.encoder(**inputs)
                embeddings.append(outputs.last_hidden_state[:, 0, :].cpu().numpy())

        return np.concatenate(embeddings)

    def solve_rot(self, C, a, b, max_iter=1000, tol=1e-6):
        K = np.exp(-C / self.lambda1)
        u = np.ones_like(a)

        for _ in range(max_iter):
            v = (a / (K.T @ u)) ** (self.lambda1 / (self.lambda1 + self.lambda2))
            v *= (1 / len(b)) ** (self.lambda2 / (self.lambda1 + self.lambda2))

            u_new = b / (K @ v)

            if np.linalg.norm(u_new - u) < tol:
                break

            u = u_new

        Q = np.diag(u) @ K @ np.diag(v)
        return Q

    def generate_pseudo_labels(self, embeddings, n_classes):
        sim_matrix = cosine_similarity(embeddings)
        C = 1 - sim_matrix

        a = np.ones(embeddings.shape[0]) / embeddings.shape[0]
        b = np.ones(n_classes) / n_classes

        Q = self.solve_rot(C, a, b)
        pseudo_labels = np.argmax(Q, axis=1)
        return pseudo_labels

    def process(self, data_df, output_path=None):
        texts = data_df['text'].tolist()
        embeddings = self.get_embeddings(texts)
        n_classes = self.estimate_num_classes(embeddings)
        data_df['pseudo_label'] = self.generate_pseudo_labels(embeddings, n_classes)

        if output_path:
            data_df.to_csv(output_path, index=False)
        return data_df

    def estimate_num_classes(self, embeddings, max_k=50):
        return min(20, max_k)

def main():
    model = BertModel.from_pretrained(config.model_name)
    tokenizer = BertTokenizer.from_pretrained(config.tokenizer_name)

    plg = PseudoLabelGenerator(model, tokenizer, lambda1=config.lambda1, lambda2=config.lambda2)

    unlabeled_df = pd.read_csv(os.path.join(config.data_processed_path, "unlabeled_data.csv"))
    labeled_df = plg.process(unlabeled_df, os.path.join(config.data_processed_path, "unlabeled_data_with_pseudo_labels.csv"))

if __name__ == "__main__":
    main()
