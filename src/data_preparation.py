import numpy as np
import pandas as pd
from datasets import load_dataset
import logging
import os
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from src.config import Config

config = Config()

def create_long_tailed_distribution(df, gamma, num_classes):
    if gamma == 1:
        return df

    class_counts = df['intent'].value_counts().sort_values(ascending=False)
    n_max = class_counts.max()

    sampled_data = []
    for i, (intent, count) in enumerate(class_counts.items()):
        n_k = max(1, int(n_max * (gamma ** (-i / (num_classes - 1)))))
        class_data = df[df['intent'] == intent]
        sampled_data.append(class_data.sample(n=min(n_k, len(class_data)),
                                          random_state=42))

    return pd.concat(sampled_data).sample(frac=1, random_state=42)

def split_data_known_novel(df, known_ratio=0.75, labeled_ratio=0.1):
    if len(df) == 0:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    all_intents = df['intent'].unique()
    num_known = max(1, int(known_ratio * len(all_intents)))
    known_intents = np.random.choice(all_intents, size=num_known, replace=False)

    known_df = df[df['intent'].isin(known_intents)]
    novel_df = df[~df['intent'].isin(known_intents)]

    labeled_df = known_df.groupby('intent').apply(
        lambda x: x.sample(frac=labeled_ratio, random_state=42)
    ).reset_index(drop=True)

    unlabeled_known = known_df[~known_df.index.isin(labeled_df.index)]
    unlabeled_df = pd.concat([unlabeled_known, novel_df])

    test_samples = []
    for intent in all_intents:
        intent_samples = df[df['intent'] == intent]
        samples = intent_samples.sample(n=min(20, len(intent_samples)),
                                      random_state=42)
        test_samples.append(samples)
    test_df = pd.concat(test_samples)

    return labeled_df, unlabeled_df, test_df

def prepare_dataset(dataset_name, dataset_config, num_classes, gamma_values):
    try:
        logging.info(f"Loading {dataset_name} dataset")
        dataset = load_dataset(dataset_name, dataset_config)
        df = pd.DataFrame(dataset['train'])

        if 'intent' not in df.columns:
            if 'label' in df.columns:
                df = df.rename(columns={'label': 'intent'})
            else:
                raise ValueError(f"Dataset {dataset_name} missing 'intent' column")

        for gamma in gamma_values:
            logging.info(f"Processing gamma={gamma}")
            save_path = os.path.join(config.data_processed_path, f"{dataset_name}_gamma{gamma}")
            os.makedirs(save_path, exist_ok=True)

            le = LabelEncoder()
            df['intent'] = le.fit_transform(df['intent'])
            num_classes = len(le.classes_)

            lt_data = create_long_tailed_distribution(df, gamma, num_classes)

            labeled, unlabeled, test = split_data_known_novel(lt_data)

            for data, name in [(labeled, "train_labeled"),
                             (unlabeled, "train_unlabeled"),
                             (test, "test")]:
                path = os.path.join(save_path, f"{name}.csv")
                data.to_csv(path, index=False)
                logging.info(f"Saved {name}: {len(data)} samples")

            log_distributions(labeled, unlabeled, test, save_path)
        import joblib
        joblib.dump(le, os.path.join(save_path, 'label_encoder.pkl'))

    except Exception as e:
        logging.error(f"Error processing {dataset_name}: {str(e)}", exc_info=True)
        raise

def log_distributions(labeled, unlabeled, test, save_path):
    with open(os.path.join(save_path, "distributions.log"), 'w') as f:
        f.write("=== Labeled Data Distribution ===\n")
        f.write(str(Counter(labeled['intent'])) + "\n\n")

        f.write("=== Unlabeled Data Distribution ===\n")
        f.write(f"Known: {Counter(unlabeled[unlabeled['intent'].isin(labeled['intent'].unique())]['intent'])}\n")
        f.write(f"Novel: {Counter(unlabeled[~unlabeled['intent'].isin(labeled['intent'].unique())]['intent'])}\n\n")

        f.write("=== Test Data Distribution ===\n")
        f.write(str(Counter(test['intent'])) + "\n")

def test_data_pipeline():
    try:
        print("=== Testing Data Pipeline ===")

        test_data = pd.DataFrame({
            'text': ['check balance'] * 15 + ['transfer money'] * 10 +
                   ['open account'] * 8 + ['loan request'] * 5,
            'intent': [0] * 15 + [1] * 10 + [2] * 8 + [3] * 5
        })

        print("\nOriginal distribution:")
        print(test_data['intent'].value_counts())

        long_tail = create_long_tailed_distribution(test_data, gamma=3, num_classes=4)
        print("\nAfter long-tail (gamma=3):")
        print(long_tail['intent'].value_counts())

        labeled, unlabeled, test = split_data_known_novel(test_data)
        print(f"\nSplit result:")
        print(f"Labeled: {len(labeled)} samples, {labeled['intent'].nunique()} classes")
        print(f"Unlabeled: {len(unlabeled)} samples")
        print(f"Test: {len(test)} samples, balanced")

        assert set(labeled['intent']).issubset(set(unlabeled['intent'])), "Label leakage!"
        assert len(set(test['intent']).difference(set(unlabeled['intent']))) == 0, "Test leakage!"

        print("\n✅ All pipeline tests passed!")
        return True
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}", exc_info=True)
        return False

def main():
    if not test_data_pipeline():
        raise ValueError("Data pipeline test failed! Fix before proceeding.")

    datasets = [
        ("clinc_oos", "plus", 150, [3, 5, 10]),
        ("banking77", None, 77, [3, 5, 10]),
    ]

    for name, config, classes, gammas in datasets:
        try:
            logging.info(f"\n{'=' * 50}\nProcessing {name}\n{'=' * 50}")
            prepare_dataset(name, config, classes, gammas)
        except Exception as e:
            logging.error(f"Failed to process {name}: {str(e)}", exc_info=True)
            continue

if __name__ == "__main__":
    main()
