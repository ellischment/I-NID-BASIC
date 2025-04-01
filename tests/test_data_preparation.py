import unittest
import pandas as pd
import numpy as np
from src.data_preparation import create_long_tailed_dataset, split_data


class TestDataPreparation(unittest.TestCase):
    def setUp(self):
        # Инициализируем синтетические данные для тестов
        self.data = {
            'text': [
                'How do I reset my password?',  # intent1
                'I want to check my balance',  # intent2
                'How can I transfer money?',  # intent3
                'What is my account number?',  # intent1
                'I need help with my card',  # intent2
                'How do I change my email?',  # intent3
                'I want to close my account',  # intent1
                'How can I open a new account?',  # intent2
                'What are the fees for transfers?',  # intent3
                'I need help with my loan'  # intent1
            ],
            'intent': [
                'intent1', 'intent2', 'intent3', 'intent1', 'intent2',
                'intent3', 'intent1', 'intent2', 'intent3', 'intent1'
            ]
        }
        self.df = pd.DataFrame(self.data)

    def test_create_long_tailed_dataset(self):
        # Call the function
        gamma = 3
        num_classes = 3
        long_tailed_df = create_long_tailed_dataset(self.df, gamma, num_classes)

        # Check that the DataFrame is not empty
        self.assertFalse(long_tailed_df.empty)
        # Check that the number of classes matches the expected value
        self.assertEqual(len(long_tailed_df['label_index'].unique()), num_classes)
        # Check that the distribution follows the gamma parameter
        class_counts = long_tailed_df['label_index'].value_counts().sort_index()
        print("Class counts:", class_counts)  # Debugging output
        self.assertTrue(class_counts[0] > class_counts[1] > class_counts[2])

    def test_split_data(self):
        # Adjust ratios to ensure the dataset is large enough for the split
        known_intent_ratio = 0.75
        labeled_ratio = 0.5  # Increase labeled_ratio to ensure enough samples

        # Call the function
        labeled_df, unlabeled_df, test_df = split_data(self.df, known_intent_ratio, labeled_ratio)

        # Check that the data is split correctly
        self.assertFalse(labeled_df.empty)
        self.assertFalse(unlabeled_df.empty)
        self.assertFalse(test_df.empty)
        # Check that labeled_df is approximately 50% of known_intent_df
        self.assertAlmostEqual(len(labeled_df) / (len(labeled_df) + len(unlabeled_df)), labeled_ratio, delta=0.1)


if __name__ == '__main__':
    unittest.main()