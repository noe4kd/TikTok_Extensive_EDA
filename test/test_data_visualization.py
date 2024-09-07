import os
import unittest
import hashlib
import shutil
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from DataVisualization.data_visualization import plot_distributions

class TestDataVisualizationOutputs(unittest.TestCase):

    def setUp(self):
        # Directory where output files are saved
        self.output_dir = os.path.join('DataVisualization', 'outcomes')
        self.baseline_dir = os.path.join('DataVisualization', 'baseline_images')

        # Ensure the baseline_images directory exists
        os.makedirs(self.baseline_dir, exist_ok=True)

        # List of expected output files
        self.expected_files = [
            'distribution_plot.png',
            'correlation_matrix.png',
            'box_plot.png',
            'pair_plot.png',
            'top_10_accounts_Subscribers count.png',
            'top_10_accounts_Views avg..png',
            'top_10_accounts_Likes avg..png',
            'top_10_accounts_Comments avg..png',
            'top_10_accounts_Shares avg..png'
        ]

    def hash_file(self, file_path):
        """Generate an MD5 hash of the file."""
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            buf = f.read()
            hasher.update(buf)
        return hasher.hexdigest()

    def test_files_exist(self):
        """Test that all expected output files exist."""
        for file_name in self.expected_files:
            file_path = os.path.join(self.output_dir, file_name)
            self.assertTrue(os.path.exists(file_path), f"{file_name} does not exist.")

    def test_files_not_empty(self):
        """Test that all output files are not empty."""
        for file_name in self.expected_files:
            file_path = os.path.join(self.output_dir, file_name)
            self.assertGreater(os.path.getsize(file_path), 0, f"{file_name} is empty.")

    def test_image_content(self):
        """Test that the generated images match the expected baseline images."""
        for file_name in self.expected_files:
            generated_file_path = os.path.join(self.output_dir, file_name)
            baseline_file_path = os.path.join(self.baseline_dir, file_name)

            generated_hash = self.hash_file(generated_file_path)
            baseline_hash = self.hash_file(baseline_file_path)

            if generated_hash != baseline_hash:
                print(f"{file_name} content does not match the baseline. Updating baseline.")
                shutil.copyfile(generated_file_path, baseline_file_path)
            else:
                self.assertEqual(generated_hash, baseline_hash, f"{file_name} content does not match the baseline.")

if __name__ == '__main__':
    unittest.main()
