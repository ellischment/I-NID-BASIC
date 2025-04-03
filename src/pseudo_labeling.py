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
    def __init__(self, model_path="bert-base-uncased", tokenizer=None, model=None):
        self.tokenizer = tokenizer if tokenizer is not None else BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = model if model is not None else BertModel.from_pretrained(model_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def load_data(self, data_path):
        return pd.read_csv(data_path)

    def get_embeddings(self, texts, batch_size=32):
        encodings = self.tokenizer(texts, truncation=True, padding=True, max_length=128)
        dataset = TensorDataset(
            torch.tensor(encodings['input_ids']),
            torch.tensor(encodings['attention_mask'])
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        self.model.eval()
        embeddings = []
        with torch.no_grad():
            for batch in loader:
                inputs = {k: v.to(self.device) for k, v in zip(['input_ids', 'attention_mask'], batch)}
                outputs = self.model(**inputs)
                embeddings.append(outputs.last_hidden_state[:, 0, :].cpu().numpy())
        return np.concatenate(embeddings)

    def generate_pseudo_labels(self, embeddings):
        a = np.ones(embeddings.shape[0]) / embeddings.shape[0]  # Uniform sample distribution
        b = np.ones(embeddings.shape[1]) / embeddings.shape[1]  # Initial class distribution

        # Cost matrix
        C = 1 - cosine_similarity(embeddings)

        # Решить проблему ROT с Sinkhorn-Knopp
        Q = self.solve_rot(C, a, b, lambda1=0.05, lambda2=2)

        return np.argmax(Q, axis=1)

    def process(self, data_df, output_path=None):
        embeddings = self.get_embeddings(data_df['text'].tolist())
        data_df['pseudo_labels'] = self.generate_pseudo_labels(embeddings)
        if output_path:
            data_df.to_csv(output_path, index=False)
        return data_df

def main():
    plg = PseudoLabelGenerator()
    plg.process(
        "data/processed/unlabeled_data.csv",
        "data/processed/unlabeled_data_with_pseudo_labels.csv"
    )

if __name__ == "__main__":
    main()
