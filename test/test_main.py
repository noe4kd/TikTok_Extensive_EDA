import unittest
import os
import sys
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Ensure the src directory is in the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.main import load_and_preprocess_data, create_new_features, prepare_data, evaluate_model

class TestMainScript(unittest.TestCase):

    def test_load_and_preprocess_data(self):
        """Test that data loads without error."""
        try:
            data = load_and_preprocess_data()
            self.assertIsInstance(data, pd.DataFrame)
        except Exception as e:
            self.fail(f"load_and_preprocess_data() raised an exception: {e}")
    
    def test_create_new_features(self):
        """Test that features are created without error."""
        data = load_and_preprocess_data()
        try:
            data_with_features = create_new_features(data)
            self.assertIn('Engagement', data_with_features.columns)
        except Exception as e:
            self.fail(f"create_new_features() raised an exception: {e}")
    
    def test_prepare_data(self):
        """Test that data preparation works without error."""
        data = load_and_preprocess_data()
        data_with_features = create_new_features(data)
        features = ['Likes_per_view', 'Comments_per_view', 'Shares_per_view', 'Engagement_rate']
        try:
            X_train, X_test, y_train, y_test = prepare_data(data_with_features, features, target='Engagement')
            self.assertEqual(len(X_train), len(y_train))
            self.assertEqual(len(X_test), len(y_test))
        except Exception as e:
            self.fail(f"prepare_data() raised an exception: {e}")
    
    def test_evaluate_model(self):
        """Test that model evaluation works without error."""
        data = load_and_preprocess_data()
        data_with_features = create_new_features(data)
        features = ['Likes_per_view', 'Comments_per_view', 'Shares_per_view', 'Engagement_rate']
        X_train, X_test, y_train, y_test = prepare_data(data_with_features, features, target='Engagement')
        model = LogisticRegression()
        model.fit(X_train, y_train)
        try:
            evaluate_model(model, X_test, y_test, model_name='LogisticRegression', save_path=None)
        except Exception as e:
            self.fail(f"evaluate_model() raised an exception: {e}")

if __name__ == '__main__':
    unittest.main()
