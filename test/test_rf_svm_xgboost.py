import unittest
import sys
import os
import time
import joblib  # For loading saved models
from PIL import Image  
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Add the root directory of the project to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestRF_SVM_XGBoostOutputs(unittest.TestCase):

    def setUp(self):
        # Set paths for outcome files and models
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.outcomes_dir = os.path.join(script_dir, 'RF_SVM_XGBOOST', 'outcomes')
        
        # Paths to the expected output files
        self.expected_confusion_matrices = {
            'XGBoost': os.path.join(self.outcomes_dir, 'XGBoost_confusion_matrix.png'),
            'Optimized XGBoost': os.path.join(self.outcomes_dir, 'Optimized XGBoost_confusion_matrix.png')
        }
        self.expected_roc_curves = {
            'XGBoost': os.path.join(self.outcomes_dir, 'roc_curve_XGBoost.png'),
            'Optimized XGBoost': os.path.join(self.outcomes_dir, 'roc_curve_Optimized XGBoost.png')
        }
        self.model_paths = {
            'XGBoost': os.path.join(self.outcomes_dir, 'xgboost_model.pkl'),
            'Optimized XGBoost': os.path.join(self.outcomes_dir, 'optimized_xgboost_model.pkl')
        }

    def test_confusion_matrix_files_exist(self):
        # Check that confusion matrix images have been created
        for model_name, path in self.expected_confusion_matrices.items():
            with self.subTest(model=model_name):
                self.assertTrue(os.path.exists(path), f"Confusion matrix for {model_name} does not exist at {path}")

    def test_roc_curve_files_exist(self):
        # Check that ROC curve images have been created
        for model_name, path in self.expected_roc_curves.items():
            with self.subTest(model=model_name):
                self.assertTrue(os.path.exists(path), f"ROC curve for {model_name} does not exist at {path}")

    def test_confusion_matrix_contents(self):
        # Validate the contents of the confusion matrix images
        for model_name, path in self.expected_confusion_matrices.items():
            with self.subTest(model=model_name):
                with Image.open(path) as img:
                    width, height = img.size
                    self.assertGreater(width, 0, f"Confusion matrix image for {model_name} is empty")
                    self.assertGreater(height, 0, f"Confusion matrix image for {model_name} is empty")

    def test_roc_curve_contents(self):
        # Validate the contents of the ROC curve images
        for model_name, path in self.expected_roc_curves.items():
            with self.subTest(model=model_name):
                with Image.open(path) as img:
                    width, height = img.size
                    self.assertGreater(width, 0, f"ROC curve image for {model_name} is empty")
                    self.assertGreater(height, 0, f"ROC curve image for {model_name} is empty")

    def test_model_performance_metrics(self):
        # Validate the performance of the saved models
        for model_name, model_path in self.model_paths.items():
            if os.path.exists(model_path):
                with self.subTest(model=model_name):
                    model = joblib.load(model_path)
                    # Assuming X_test and y_test are saved somewhere or can be reloaded
                    X_test = ...  # Load or define X_test based on how you save it in rf_svm_xgboost.py
                    y_test = ...  # Load or define y_test based on how you save it in rf_svm_xgboost.py
                    y_pred = model.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred)
                    recall = recall_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred)
                    self.assertGreaterEqual(accuracy, 0.7, f"{model_name} failed on accuracy.")
                    self.assertGreaterEqual(precision, 0.7, f"{model_name} failed on precision.")
                    self.assertGreaterEqual(recall, 0.7, f"{model_name} failed on recall.")
                    self.assertGreaterEqual(f1, 0.7, f"{model_name} failed on F1-score.")

if __name__ == '__main__':
    unittest.main()