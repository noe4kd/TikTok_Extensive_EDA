import unittest
import subprocess
import os
import sys
import fnmatch
import time

# Add the root directory of the project to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestModelWithFeatureEngineering(unittest.TestCase):

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
        expected_files_patterns = [
            'roc_curve_Stacking Classifier with New Features.png',
            'Stacking Classifier with New Features_confusion_matrix.png'
        ]

        output_files = os.listdir(self.output_dir)

        for pattern in expected_files_patterns:
            matching_files = fnmatch.filter(output_files, pattern)
            print(f"Pattern {pattern}: Found {matching_files}")
            self.assertGreater(len(matching_files), 0, f"No files matching pattern {pattern} were created.")

if __name__ == '__main__':
    unittest.main()
