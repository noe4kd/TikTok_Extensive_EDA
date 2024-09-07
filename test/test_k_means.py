import unittest
import subprocess
import os
import sys
from PIL import Image

# Adding the parent directory to the system path so we can import the necessary scripts
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestKMeans(unittest.TestCase):

    def setUp(self):
        # Set the output directory where the image and data files are expected to be saved
        self.output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'K_means', 'outcomes'))
        print(f"Output directory: {self.output_dir}")

        # Define the script directory and script path
        script_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'K_means'))
        script_path = os.path.join(script_dir, 'k_means.py')
        print(f"Running script: {script_path}")
        print(f"Current working directory: {os.getcwd()}")

        # Run the K-Means script once and capture the output
        self.result = subprocess.run([sys.executable, script_path], capture_output=True, text=True, cwd=script_dir)
        print("STDOUT:", self.result.stdout)
        print("STDERR:", self.result.stderr)

        # Ensure the script ran successfully
        self.assertEqual(self.result.returncode, 0, f"Script failed with return code {self.result.returncode}")

        # Path to the output image
        self.output_image = os.path.join(self.output_dir, 'k-means_clusters.png')
        self.assertTrue(os.path.exists(self.output_image), f"Output image {self.output_image} not found.")

    def test_output_image_exists(self):
        # Validate that the output image file exists and is not empty
        self.assertGreater(os.path.getsize(self.output_image), 0, f"Output image {self.output_image} is empty.")

    def test_image_is_valid(self):
        # Validate that the output image is a valid image file
        try:
            with Image.open(self.output_image) as img:
                img.verify()  # Verifies the file can be opened and is a valid image
        except Exception as e:
            self.fail(f"The output image {self.output_image} is not a valid image file or cannot be opened. Error: {str(e)}")

    def test_silhouette_score_in_output(self):
        # Check that the silhouette score is mentioned in the script output
        self.assertIn("Silhouette Score", self.result.stdout, "Silhouette Score not found in script output.")

    def test_reproducibility(self):
        # Run the K-Means script again to check for reproducibility
        script_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'K_means'))
        new_result = subprocess.run([sys.executable, os.path.join(script_dir, 'k_means.py')], capture_output=True, text=True, cwd=script_dir)
        
        # Ensure the second run also completed successfully
        self.assertEqual(new_result.returncode, 0, "Second run of script failed, affecting reproducibility.")
        
        # Check if the results of the second run match the first run
        self.assertEqual(self.result.stdout, new_result.stdout, "Results are not reproducible across runs.")

if __name__ == '__main__':
    unittest.main()
