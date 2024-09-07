import unittest
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import sys
import os

# Adjust the sys.path to correctly point to the KNN module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'KNN')))

from KNN import create_engagement_variable, optimize_knn, evaluate_model, create_new_features, prepare_data

class TestKNNPipeline(unittest.TestCase):

    def setUp(self):
        """Set up the environment for testing."""
        # Set a random seed for reproducibility
        np.random.seed(42)

        # Simulate a small dataset for testing
        self.df = pd.DataFrame({
            'Subscribers count': np.random.randint(100, 1000, size=100),
            'Views avg.': np.random.randint(1000, 5000, size=100),
            'Comments avg.': np.random.randint(10, 100, size=100),
            'Shares avg.': np.random.randint(5, 50, size=100),
            'Likes avg.': np.random.randint(100, 1000, size=100)
        })

        # Create an engagement variable
        self.df = create_engagement_variable(self.df)

        # Features to use
        self.features = [
            'Subscribers count', 'Views avg.', 'Comments avg.', 'Shares avg.'
        ]

        # Prepare the data (split into training and testing sets)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.df[self.features], self.df['Engagement'], test_size=0.2, random_state=42)

        # Standardize the features
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

        # Add polynomial features
        self.poly = PolynomialFeatures(degree=2, include_bias=False)
        self.X_train_poly = self.poly.fit_transform(self.X_train)
        self.X_test_poly = self.poly.transform(self.X_test)

        # Performance log
        self.performance_log = []

        # Correct output directory to point to the KNN/outcomes folder
        self.output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'KNN', 'outcomes'))
        os.makedirs(self.output_dir, exist_ok=True)


    def test_create_engagement_variable(self):
        """Test if the engagement variable is created correctly."""
        self.assertIn('Engagement', self.df.columns)
        self.assertEqual(set(self.df['Engagement'].unique()), {0, 1})

    def test_create_new_features(self):
        """Test if new features are created correctly."""
        df_with_features = create_new_features(self.df)
        self.assertIn('Likes_per_view', df_with_features.columns)

    def test_prepare_data(self):
        """Test if the data is split correctly into training and testing sets."""
        X_train, X_temp, y_train, y_temp = prepare_data(self.df, self.features, target='Engagement')
        # Check the sizes of the splits
        self.assertEqual(len(X_train), len(y_train))
        self.assertEqual(len(X_temp), len(y_temp))
        # Ensure that the splits are not empty
        self.assertGreater(len(X_train), 0)
        self.assertGreater(len(X_temp), 0)

    def test_optimize_knn(self):
        """Test if KNN optimization runs without errors."""
        knn = optimize_knn(self.X_train_poly, self.y_train)
        self.assertIsInstance(knn, KNeighborsClassifier)

    def test_evaluate_model(self):
        """Test if model evaluation runs without errors."""
        knn = optimize_knn(self.X_train_poly, self.y_train)

        # Define a unique filename 
        unique_filename = 'knn_cm_test_set_test_script.png'

        # Pass the unique filename to evaluate_model
        y_pred = evaluate_model(knn, self.X_test_poly, self.y_test, 'Test Set', self.output_dir, custom_filename=unique_filename)

        # Performance metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)

        # Log the performance
        self.performance_log.append({
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'params': knn.get_params()
        })

        self.assertGreaterEqual(accuracy, 0.3, f"Model accuracy {accuracy} is below the acceptable threshold.")

    def test_pipeline(self):
        """Test the entire pipeline."""
        try:
            knn = optimize_knn(self.X_train_poly, self.y_train)
            # Define a unique filename for this test
            unique_filename = 'knn_cm_test_set_test_script.png'
            y_pred = evaluate_model(knn, self.X_test_poly, self.y_test, 'Test Set', self.output_dir, custom_filename= unique_filename)
            self.assertIsNotNone(y_pred)
        except Exception as e:
            self.fail(f"Pipeline failed with exception: {e}")

    def tearDown(self):
        """Print the performance log after tests."""
        if self.performance_log:
            print("\nPerformance Log:")
            for entry in self.performance_log:
                print(f"Params: {entry['params']}")
                print(f"Accuracy: {entry['accuracy']}, Precision: {entry['precision']}, Recall: {entry['recall']}, F1: {entry['f1']}\n")

if __name__ == '__main__':
    unittest.main()
