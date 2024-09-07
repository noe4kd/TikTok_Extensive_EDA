import unittest
import os
import sys
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from matplotlib import pyplot as plt
import joblib
from PIL import Image 

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
class TestRFAndSVMOutputs(unittest.TestCase):

    def setUp(self):
        # Set the paths to the outcomes directory and expected outputs
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.outcomes_dir = os.path.join(script_dir, 'RF_SVM', 'outcomes')
        
        # Load expected outputs
        self.expected_confusion_matrices = {
            'Random Forest': os.path.join(self.outcomes_dir, 'Random Forest_confusion_matrix.png'),
            'Optimized Random Forest': os.path.join(self.outcomes_dir, 'Optimized Random Forest_confusion_matrix.png'),
            'SVM': os.path.join(self.outcomes_dir, 'Support Vector Machine_confusion_matrix.png'),
            'Optimized SVM': os.path.join(self.outcomes_dir, 'Optimized SVM_confusion_matrix.png')
        }
        self.expected_roc_curves = {
            'Random Forest': os.path.join(self.outcomes_dir, 'roc_curve_Random Forest.png'),
            'Optimized Random Forest': os.path.join(self.outcomes_dir, 'roc_curve_Optimized RF with New Features.png'),
            'SVM': os.path.join(self.outcomes_dir, 'roc_curve_SVM.png'),
            'Optimized SVM': os.path.join(self.outcomes_dir, 'roc_curve_Optimized SVM with New Features.png')
        }
        self.feature_importance_path = os.path.join(self.outcomes_dir, 'feature_importance.png')

        # Optionally, load a saved model for deeper tests
        self.saved_rf_model_path = os.path.join(self.outcomes_dir, 'best_rf_model.pkl')
        self.saved_svm_model_path = os.path.join(self.outcomes_dir, 'best_svm_model.pkl')

        self.rf_model = joblib.load(self.saved_rf_model_path) if os.path.exists(self.saved_rf_model_path) else None
        self.svm_model = joblib.load(self.saved_svm_model_path) if os.path.exists(self.saved_svm_model_path) else None

        # Load the test data to check performance metrics
        self.test_data_path = os.path.join(script_dir, 'data', 'tiktok_preprocessed.csv')
        self.test_data = pd.read_csv(self.test_data_path)
        self.features = ['Subscribers count', 'Views avg.', 'Comments avg.', 'Shares avg.']
        self.X_test = self.test_data[self.features]
        self.y_test = (self.test_data['Likes avg.'] > self.test_data['Likes avg.'].median()).astype(int)

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

    def test_feature_importance_file_exists(self):
        # Check that feature importance plot has been created
        self.assertTrue(os.path.exists(self.feature_importance_path), f"Feature importance plot does not exist at {self.feature_importance_path}")

    def test_confusion_matrix_contents(self):
        # Load the expected confusion matrix image and check dimensions or other properties
        for model_name, path in self.expected_confusion_matrices.items():
            with self.subTest(model=model_name):
                with Image.open(path) as img:
                    width, height = img.size
                    self.assertGreater(width, 0, f"Confusion matrix image for {model_name} is empty")
                    self.assertGreater(height, 0, f"Confusion matrix image for {model_name} is empty")
                # Add more checks as needed, e.g., pixel comparison if you have a baseline image

    def test_roc_curve_contents(self):
        # Load the expected ROC curve image and check dimensions or other properties
        for model_name, path in self.expected_roc_curves.items():
            with self.subTest(model=model_name):
                with Image.open(path) as img:
                    width, height = img.size
                    self.assertGreater(width, 0, f"ROC curve image for {model_name} is empty")
                    self.assertGreater(height, 0, f"ROC curve image for {model_name} is empty")
                # Add more checks as needed

    def test_model_performance_metrics(self):
        # Ensure the loaded model achieves expected performance metrics
        if self.rf_model is not None:
            y_pred_rf = self.rf_model.predict(self.X_test)
            self.assertGreaterEqual(accuracy_score(self.y_test, y_pred_rf), 0.7)
            self.assertGreaterEqual(precision_score(self.y_test, y_pred_rf), 0.7)
            self.assertGreaterEqual(recall_score(self.y_test, y_pred_rf), 0.7)
            self.assertGreaterEqual(f1_score(self.y_test, y_pred_rf), 0.7)

        if self.svm_model is not None:
            y_pred_svm = self.svm_model.predict(self.X_test)
            self.assertGreaterEqual(accuracy_score(self.y_test, y_pred_svm), 0.7)
            self.assertGreaterEqual(precision_score(self.y_test, y_pred_svm), 0.7)
            self.assertGreaterEqual(recall_score(self.y_test, y_pred_svm), 0.7)
            self.assertGreaterEqual(f1_score(self.y_test, y_pred_svm), 0.7)

if __name__ == '__main__':
    unittest.main()