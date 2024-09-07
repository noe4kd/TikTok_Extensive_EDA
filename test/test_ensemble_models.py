import unittest
import subprocess
import os
import sys
import time

# Add the root directory of the project to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestEnsembleModels(unittest.TestCase):

    def setUp(self):
        self.output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'EnsembleModels', 'outcomes'))
        print(f"Output directory: {self.output_dir}")

    def test_run_ensemble_models_script(self):
        script_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'EnsembleModels'))
        script_path = os.path.join(script_dir, 'EnsembleModels.py')
        print(f"Running script: {script_path}")
        print(f"Current working directory: {os.getcwd()}")
        result = subprocess.run([sys.executable, script_path], capture_output=True, text=True, cwd=script_dir)

        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)

        self.assertEqual(result.returncode, 0, "EnsembleModels.py script did not run successfully")

        print("Files in output directory after running the script:")
        for file in os.listdir(self.output_dir):
            print(file)

    def test_output_files_exist(self):
        print(f"Looking for files in: {self.output_dir}")
        expected_files = [
            'Stacking Classifier_confusion_matrix.png',  
            'roc_curve_Stacking Classifier.png'          
        ]

        time.sleep(2)

        print("Files in output directory before checking:")
        for file in os.listdir(self.output_dir):
            print(file)

        for file_name in expected_files:
            file_path = os.path.join(self.output_dir, file_name)
            self.assertTrue(os.path.isfile(file_path), f"Expected output file {file_name} was not created in {self.output_dir}")

            
    def test_classification_report_content(self):
        # Run the script
        script_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'EnsembleModels'))
        script_path = os.path.join(script_dir, 'EnsembleModels.py')
        result = subprocess.run([sys.executable, script_path], capture_output=True, text=True, cwd=script_dir)

        # Check that the script ran successfully
        self.assertEqual(result.returncode, 0, "EnsembleModels.py script did not run successfully")

        # Check for expected metrics in the classification report output
        expected_keywords = ['precision', 'recall', 'f1-score', 'accuracy']
        for keyword in expected_keywords:
            self.assertIn(keyword, result.stdout, f"{keyword} not found in the classification report output")

        # Optionally, check specific values if you have known expected results
        expected_accuracy = "accuracy"
        self.assertIn(expected_accuracy, result.stdout, "Expected accuracy value not found in the report")

        print("Classification report content test passed.")

    def test_performance(self):
        # Track the start time
        start_time = time.time()

        # Run the script (or the key parts of it)
        script_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'EnsembleModels'))
        script_path = os.path.join(script_dir, 'EnsembleModels.py')
        result = subprocess.run([sys.executable, script_path], capture_output=True, text=True, cwd=script_dir)

        # Track the end time
        end_time = time.time()

        # Calculate the elapsed time
        elapsed_time = end_time - start_time

        # Print the elapsed time
        print(f"Performance Test: Time taken to run EnsembleModels.py: {elapsed_time:.2f} seconds")

        # Assert that the performance is within an acceptable range (e.g., less than 60 seconds)
        self.assertLess(elapsed_time, 60, "Performance test failed: Script took too long to run")

        print("Performance test passed.")        

if __name__ == '__main__':
    unittest.main()
