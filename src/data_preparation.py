import numpy as np
import pandas as pd
from datasets import load_dataset
import logging
import os
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOGS_DIR, "data_prep.log"),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def create_long_tailed_distribution(df, gamma, num_classes):
    """Create long-tailed distribution following the paper's formula"""
    n_max = df['intent'].value_counts().max()
    sampled_data = []

    for i, intent in enumerate(sorted(df['intent'].unique())):
        n_k = int(n_max * (gamma ** (-i / (num_classes - 1))))
        class_data = df[df['intent'] == intent]
        sampled_data.append(class_data.sample(n=min(n_k, len(class_data)), random_state=42))

    return pd.concat(sampled_data)


def split_data_known_novel(df, known_ratio=0.75, labeled_ratio=0.1, min_samples_per_class=1):
    """
    Split data into labeled (known), unlabeled (known + novel) and balanced test sets
    with guaranteed minimum samples.

    Args:
        df: Input DataFrame with 'text' and 'intent' columns
        known_ratio: Proportion of intents to treat as known (default: 0.75)
        labeled_ratio: Proportion of known intents to label (default: 0.1)
        min_samples_per_class: Minimum samples per test class (default: 1)

    Returns:
        tuple: (labeled_df, unlabeled_df, test_df)
    """
    # Handle empty input
    if len(df) == 0:
        return pd.DataFrame(columns=df.columns), pd.DataFrame(columns=df.columns), pd.DataFrame(columns=df.columns)

    # Get unique intents
    all_intents = df['intent'].unique()
    num_known = max(1, int(known_ratio * len(all_intents)))  # At least 1 known intent
    known_intents = np.random.choice(all_intents, size=num_known, replace=False)

    # Split known data
    known_df = df[df['intent'].isin(known_intents)]
    novel_df = df[~df['intent'].isin(known_intents)]

    # Ensure minimum labeled samples
    min_labeled = max(1, int(len(known_df) * labeled_ratio))
    labeled_df = known_df.sample(n=min(min_labeled, len(known_df)), random_state=42)

    # Create unlabeled set (remaining known + all novel)
    unlabeled_known = known_df[~known_df.index.isin(labeled_df.index)]
    unlabeled_df = pd.concat([unlabeled_known, novel_df])

    # Create balanced test set
    test_samples = []
    for intent in all_intents:
        intent_samples = df[df['intent'] == intent]
        samples = intent_samples.sample(
            n=min(min_samples_per_class, len(intent_samples)),
            random_state=42
        )
        test_samples.append(samples)
    test_df = pd.concat(test_samples)

    return labeled_df, unlabeled_df, test_df


def prepare_dataset(dataset_name, dataset_config, num_classes, gamma_values):
    try:
        raw_data = load_dataset(dataset_name, dataset_config)['train']
        df = pd.DataFrame(raw_data)

        for gamma in gamma_values:
            save_path = os.path.join(PROCESSED_DIR, f"{dataset_name}_gamma{gamma}")
            os.makedirs(save_path, exist_ok=True)

            lt_data = create_long_tailed_distribution(df, gamma, num_classes)
            labeled, unlabeled, test = split_data_known_novel(lt_data)

            labeled.to_csv(os.path.join(save_path, "train_labeled.csv"), index=False)
            unlabeled.to_csv(os.path.join(save_path, "train_unlabeled.csv"), index=False)
            test.to_csv(os.path.join(save_path, "test.csv"), index=False)

    except Exception as e:
        logging.error(f"Error processing {dataset_name}: {str(e)}")
        raise

def main():
    """Main data preparation pipeline"""
    datasets = [
        ("clinc_oos", "plus", 150, [3, 5, 10]),
        ("banking77", None, 77, [3, 5, 10]),
    ]

    for name, config, classes, gammas in datasets:
        logging.info(f"Starting {name} processing")
        prepare_dataset(name, config, classes, gammas)

    # StackOverflow dataset (kept commented as in original)
    # logging.info("Loading StackOverflow dataset")
    # stackoverflow_dataset = load_dataset("stackoverflow")
    # stackoverflow_df = pd.DataFrame(stackoverflow_dataset['train'])
    # logging.info("Creating long-tailed distribution for StackOverflow")
    # stackoverflow20_lt = create_long_tailed_distribution(stackoverflow_df, gamma=10, num_classes=20)
    # logging.info("Splitting StackOverflow data")
    # stackoverflow_labeled, stackoverflow_unlabeled, stackoverflow_test = split_data_known_novel(stackoverflow20_lt)
    #
    # stackoverflow_path = os.path.join(PROCESSED_DIR, "stackoverflow_gamma10")
    # os.makedirs(stackoverflow_path, exist_ok=True)
    # stackoverflow_labeled.to_csv(os.path.join(stackoverflow_path, "train_labeled.csv"), index=False)
    # stackoverflow_unlabeled.to_csv(os.path.join(stackoverflow_path, "train_unlabeled.csv"), index=False)
    # stackoverflow_test.to_csv(os.path.join(stackoverflow_path, "test.csv"), index=False)
    # logging.info(f"Processed StackOverflow: labeled={len(stackoverflow_labeled)}, "
    #              f"unlabeled={len(stackoverflow_unlabeled)}, test={len(stackoverflow_test)}")


if __name__ == "__main__":
    main()