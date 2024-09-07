import unittest
import subprocess
import os
import sys
import time
from PIL import Image

# Add the root directory of the project to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestPostCVFeatureEng(unittest.TestCase):

    def setUp(self):
        # Define the output directory
        self.output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'EnsembleModels', 'outcomes'))
        print(f"Output directory: {self.output_dir}")

    def test_run_post_cv_feature_eng_script(self):
        # Define the script path
        script_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'EnsembleModels'))
        script_path = os.path.join(script_dir, 'post_cv_feature_eng.py')
        print(f"Running script: {script_path}")
        print(f"Current working directory: {os.getcwd()}")

        # Run the script
        start_time = time.time()
        result = subprocess.run([sys.executable, script_path], capture_output=True, text=True, cwd=script_dir)
        elapsed_time = time.time() - start_time

        # Print STDOUT and STDERR for debugging
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)

        # Check that the script ran without errors
        self.assertEqual(result.returncode, 0, "post_cv_feature_eng.py script did not run successfully")

        # Ensure the script runs within a reasonable time limit (e.g., 5 minutes)
        self.assertLess(elapsed_time, 300, "The script took too long to run (more than 5 minutes).")

        print("Script ran successfully.")

    def test_output_files_exist_and_valid(self):
        print(f"Looking for files in: {self.output_dir}")
        expected_files = [
            'Stacking Classifier (Test Set)_confusion_matrix.png',
            'roc_curve_Stacking Classifier (Test Set).png'
        ]

        for file_name in expected_files:
            file_path = os.path.join(self.output_dir, file_name)
            self.assertTrue(os.path.isfile(file_path), f"Expected output file {file_name} was not created in {self.output_dir}")

            # Validate the content of the image file
            with Image.open(file_path) as img:
                self.assertTrue(img.format in ['PNG', 'JPEG'], f"File {file_name} is not a valid image format.")
                self.assertGreater(img.size[0], 0, f"File {file_name} appears to be empty (width = 0).")
                self.assertGreater(img.size[1], 0, f"File {file_name} appears to be empty (height = 0).")

        print("Output files exist and are valid.")

    def test_data_integrity(self):
        from src.main import load_and_preprocess_data, create_new_features
        
        # Load and preprocess data
        data = load_and_preprocess_data()
        data = create_new_features(data)

        # Check that the data is not empty and has the expected features
        self.assertGreater(len(data), 0, "The dataset is empty after preprocessing.")
        expected_columns = [
            'Subscribers count', 'Views avg.', 'Comments avg.', 'Shares avg.', 
            'Likes_per_view', 'Comments_per_view', 'Shares_per_view', 'Engagement_rate', 'Engagement'
        ]
        for col in expected_columns:
            self.assertIn(col, data.columns, f"Expected column {col} not found in the dataset.")

        print("Data integrity test passed.")

if __name__ == '__main__':
    unittest.main()
