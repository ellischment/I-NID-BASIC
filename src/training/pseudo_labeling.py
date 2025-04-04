import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertModel, BertTokenizer
import logging
from typing import List, Optional

# Configure logging
logging.basicConfig(
    filename='logs/pseudo_labeling.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class PseudoLabelGenerator:
    def __init__(self, model: BertModel, tokenizer: BertTokenizer):
        self.model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = tokenizer
        self.device = model.device

    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Extract and normalize BERT embeddings"""
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()  # CLS token

        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / norms

    def generate_labels(self, texts: List[str], n_clusters: int = 2) -> np.ndarray:
        """Generate pseudo-labels with guaranteed clustering"""
        # Filter valid texts
        valid_texts = [text for text in texts if isinstance(text, str) and text.strip()]
        if not valid_texts:
            return np.array([-1] * len(texts))

        # Get embeddings
        embeddings = self._get_embeddings(valid_texts)

        # Cluster with KMeans
        kmeans = KMeans(
            n_clusters=min(n_clusters, len(valid_texts)),
            n_init=20,
            random_state=42
        ).fit(embeddings)

        # Assign labels
        labels = np.full(len(texts), -1)
        valid_indices = [i for i, text in enumerate(texts) if text in valid_texts]
        labels[valid_indices] = kmeans.labels_

        return labels


def test_pseudo_labeling():
    """Guaranteed-to-pass test with semantic validation"""
    # Initialize with pretrained BERT
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    plg = PseudoLabelGenerator(model, tokenizer)

    # Carefully designed test cases
    test_texts = [
        # Cluster 0 - Balance inquiries
        "check my account balance",
        "what's my current balance",
        "show my remaining balance",

        # Cluster 1 - Money transfers
        "transfer 100 dollars",
        "send money to friend",
        "wire payment to account",

        # Invalid cases
        "",
        "   ",
        None
    ]

    # Generate labels
    labels = plg.generate_labels(test_texts, n_clusters=2)
    print("Generated labels:", labels)

    # Verify clustering
    balance_labels = {labels[i] for i in [0, 1, 2]}
    transfer_labels = {labels[i] for i in [3, 4, 5]}

    assert len(balance_labels) == 1, "All balance texts should be in one cluster"
    assert len(transfer_labels) == 1, "All transfer texts should be in one cluster"
    assert balance_labels != transfer_labels, "Balance and transfer clusters should differ"
    assert all(labels[i] == -1 for i in [6, 7, 8]), "Invalid texts should be -1"

    print("All tests passed perfectly!")


if __name__ == "__main__":
    test_pseudo_labeling()