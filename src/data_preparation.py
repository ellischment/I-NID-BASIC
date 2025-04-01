import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import load_dataset
import logging
import os

# Define base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Root of the project
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

# Create logs directory if it doesn't exist
os.makedirs(LOGS_DIR, exist_ok=True)
logging.basicConfig(filename=os.path.join(LOGS_DIR, "imbanid.log"), level=logging.INFO)


def create_long_tailed_dataset(df, gamma, num_classes):
    """
    Create a long-tailed distribution from the dataset:
    - Assigns a label index to each intent.
    - Samples data based on the imbalance ratio (gamma).
    """
    logging.info(f"Creating long-tailed distribution with gamma={gamma} and num_classes={num_classes}")
    df['label_index'] = df['intent'].astype('category').cat.codes
    n_max = df['label_index'].value_counts().max()
    sampled_data = []
    for i in range(num_classes):
        n_k = int(n_max * (gamma ** (-i / (num_classes - 1))))
        class_data = df[df['label_index'] == i]
        if len(class_data) > 0:
            sampled_class_data = class_data.sample(n=min(n_k, len(class_data)), random_state=42)
            sampled_data.append(sampled_class_data)
            logging.info(f"Class {i}: sampled {len(sampled_class_data)} instances")

    long_tailed_df = pd.concat(sampled_data)
    logging.info(f"Long-tailed distribution created, total instances: {len(long_tailed_df)}")
    return long_tailed_df


def split_data(df, known_intent_ratio=0.75, labeled_ratio=0.1):
    """
    Split data into labeled, unlabeled, and test sets:
    - known_intent_ratio: Ratio of known intents to total intents.
    - labeled_ratio: Ratio of labeled data to known intents.
    """
    logging.info(f"Splitting data: known_intent_ratio={known_intent_ratio}, labeled_ratio={labeled_ratio}")
    all_intents = df['intent'].unique()
    num_known_intents = int(known_intent_ratio * len(all_intents))
    known_intents = np.random.choice(all_intents, size=num_known_intents, replace=False)
    known_intent_df = df[df['intent'].isin(known_intents)]
    unknown_intent_df = df[~df['intent'].isin(known_intents)]

    if len(known_intent_df) == 0:
        logging.warning("No data for known intents")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Ensure labeled_ratio is feasible given the dataset size
    min_labeled_samples = 1
    if len(known_intent_df) * labeled_ratio < min_labeled_samples:
        labeled_ratio = min_labeled_samples / len(known_intent_df)

    labeled_df, unlabeled_df = train_test_split(known_intent_df, test_size=1 - labeled_ratio, random_state=42)
    test_df = unknown_intent_df

    logging.info(f"Split completed: labeled={len(labeled_df)}, unlabeled={len(unlabeled_df)}, test={len(test_df)}")
    return labeled_df, unlabeled_df, test_df


def prepare_data():
    """
    Prepare data for training:
    - Load datasets (CLINC150, Banking77).
    - Create long-tailed distributions.
    - Split data into labeled, unlabeled, and test sets.
    - Save processed data to data/processed/.
    """
    logging.info("Starting data preparation")

    # Load datasets
    logging.info("Loading CLINC150 dataset")
    clinc_dataset = load_dataset("clinc_oos", "plus")
    logging.info("Loading Banking77 dataset")
    banking_dataset = load_dataset("banking77")

    # Convert to DataFrames
    clinc_df = pd.DataFrame(clinc_dataset['train'])
    banking_df = pd.DataFrame(banking_dataset['train'])

    # Create long-tailed distributions
    logging.info("Creating long-tailed distribution for CLINC150")
    clinc150_lt = create_long_tailed_dataset(clinc_df, gamma=3, num_classes=150)
    logging.info("Creating long-tailed distribution for Banking77")
    banking77_lt = create_long_tailed_dataset(banking_df, gamma=5, num_classes=77)

    # Split data into labeled, unlabeled, and test sets
    logging.info("Splitting CLINC150 data")
    clinc_labeled, clinc_unlabeled, clinc_test = split_data(clinc150_lt)
    logging.info("Splitting Banking77 data")
    banking_labeled, banking_unlabeled, banking_test = split_data(banking77_lt)

    # Save processed data
    os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)

    # Save CLINC150 data
    clinc_labeled.to_csv(os.path.join(DATA_PROCESSED_DIR, "clinc_labeled.csv"), index=False)
    clinc_unlabeled.to_csv(os.path.join(DATA_PROCESSED_DIR, "clinc_unlabeled.csv"), index=False)
    clinc_test.to_csv(os.path.join(DATA_PROCESSED_DIR, "clinc_test.csv"), index=False)

    # Save Banking77 data
    banking_labeled.to_csv(os.path.join(DATA_PROCESSED_DIR, "banking_labeled.csv"), index=False)
    banking_unlabeled.to_csv(os.path.join(DATA_PROCESSED_DIR, "banking_unlabeled.csv"), index=False)
    banking_test.to_csv(os.path.join(DATA_PROCESSED_DIR, "banking_test.csv"), index=False)

    # Uncomment this section if StackOverflow dataset is available
    # logging.info("Loading StackOverflow dataset")
    # stackoverflow_dataset = load_dataset("stackoverflow")
    # stackoverflow_df = pd.DataFrame(stackoverflow_dataset['train'])
    # logging.info("Creating long-tailed distribution for StackOverflow")
    # stackoverflow20_lt = create_long_tailed_dataset(stackoverflow_df, gamma=10, num_classes=20)
    # logging.info("Splitting StackOverflow data")
    # stackoverflow_labeled, stackoverflow_unlabeled, stackoverflow_test = split_data(stackoverflow20_lt)
    # stackoverflow_labeled.to_csv(os.path.join(DATA_PROCESSED_DIR, "stackoverflow_labeled.csv"), index=False)
    # stackoverflow_unlabeled.to_csv(os.path.join(DATA_PROCESSED_DIR, "stackoverflow_unlabeled.csv"), index=False)
    # stackoverflow_test.to_csv(os.path.join(DATA_PROCESSED_DIR, "stackoverflow_test.csv"), index=False)

    logging.info("Data preparation completed and saved to data/processed/")
    breakpoint()  # Debugging checkpoint


if __name__ == "__main__":
    prepare_data()