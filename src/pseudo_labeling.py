import numpy as np
import ot
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
import pandas as pd
import logging
import os

os.makedirs('logs', exist_ok=True)
logging.basicConfig(filename='logs/pseudo_labeling.log', level=logging.INFO)


class PseudoLabelGenerator:
    def __init__(self, model, tokenizer, lambda1=0.05, lambda2=2):
        # Изменяем тип модели для получения эмбеддингов
        self.encoder = model.bert if hasattr(model, 'bert') else model.base_model
        self.tokenizer = tokenizer
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder.to(self.device)

    def get_embeddings(self, texts, batch_size=32):
        """Get BERT embeddings for texts"""
        self.encoder.eval()
        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.encoder(**inputs)
                embeddings.append(outputs.last_hidden_state[:, 0, :].cpu().numpy())

        return np.concatenate(embeddings)

    def solve_rot(self, C, a, b, max_iter=1000, tol=1e-6):
        """
        Solve Relaxed Optimal Transport problem with Sinkhorn-Knopp algorithm
        Args:
            C: Cost matrix (n_samples x n_classes)
            a: Sample distribution (n_samples,)
            b: Initial class distribution (n_classes,)
        Returns:
            Q: Optimal transport matrix
        """
        K = np.exp(-C / self.lambda1)
        u = np.ones_like(a)

        for _ in range(max_iter):
            # Update v with KL constraint
            v = (a / (K.T @ u)) ** (self.lambda1 / (self.lambda1 + self.lambda2))
            v *= (1 / (len(b))) ** (self.lambda2 / (self.lambda1 + self.lambda2))

            # Update u
            u_new = b / (K @ v)

            if np.linalg.norm(u_new - u) < tol:
                break

            u = u_new

        Q = np.diag(u) @ K @ np.diag(v)
        return Q

    def generate_pseudo_labels(self, embeddings, n_classes):
        # Calculate cost matrix (negative cosine similarity)
        sim_matrix = cosine_similarity(embeddings)
        C = 1 - sim_matrix

        # Uniform sample distribution
        a = np.ones(embeddings.shape[0]) / embeddings.shape[0]

        # Initial class distribution (uniform)
        b = np.ones(n_classes) / n_classes

        # Solve ROT problem
        Q = self.solve_rot(C, a, b)

        # Get pseudo-labels
        pseudo_labels = np.argmax(Q, axis=1)
        return pseudo_labels

    def process(self, data_df, output_path=None):
        """Process unlabeled data to generate pseudo-labels"""
        # Get embeddings
        texts = data_df['text'].tolist()
        embeddings = self.get_embeddings(texts)

        # Estimate number of classes (you may need to implement this)
        n_classes = self.estimate_num_classes(embeddings)

        # Generate pseudo-labels
        data_df['pseudo_labels'] = self.generate_pseudo_labels(embeddings, n_classes)

        if output_path:
            data_df.to_csv(output_path, index=False)
        return data_df

    def estimate_num_classes(self, embeddings, max_k=50):
        """Simple method to estimate number of classes (replace with better method)"""
        # This is a placeholder - implement proper estimation as per your needs
        return min(20, max_k)  # Example: cap at 20 classes


def main():
    # Initialize model and tokenizer
    model = BertModel.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Create pseudo-label generator
    plg = PseudoLabelGenerator(model, tokenizer)

    # Process data
    unlabeled_df = pd.read_csv("data/processed/unlabeled_data.csv")
    labeled_df = plg.process(unlabeled_df, "data/processed/unlabeled_data_with_pseudo_labels.csv")


if __name__ == "__main__":
    main()