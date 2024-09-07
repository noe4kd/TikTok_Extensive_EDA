import unittest
import sys
import os
import pandas as pd

# Add the root directory of the project to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_preprocessing import preprocess_data

class TestDataPreprocessing(unittest.TestCase):

    def setUp(self):
        """Set up the data for testing."""
        self.processed_data = preprocess_data()

    def test_preprocess_data(self):
        """Test cases to verify the output of preprocess_data."""
        # Ensure processed_data is not None and not empty
        self.assertIsNotNone(self.processed_data, "Processed data should be defined.")
        self.assertFalse(self.processed_data.empty, "Processed data should not be empty.")

        # Verify the number of columns
        expected_columns = 3015  # Adjust this based on your actual expected columns
        self.assertEqual(self.processed_data.shape[1], expected_columns, f"Processed data should have {expected_columns} columns.")

        # Verify the presence of the 'Account' column
        self.assertIn('Account', self.processed_data.columns, "Processed data should contain the 'Account' column.")

        # Check for missing values
        self.assertFalse(self.processed_data.isnull().any().any(), "Processed data should not have any missing values.")
        
    def test_no_missing_values(self):
        """Ensure there are no missing values after processing."""
        # This test is redundant now as the check is already part of test_preprocess_data
        # But if you want to keep it separate, ensure self.processed_data is defined in setUp
        self.assertFalse(self.processed_data.isnull().any().any(), "Processed data should not have any missing values.")

if __name__ == '__main__':
    unittest.main()
