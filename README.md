
# TikTok EDA Project

## Overview
This project involves an extensive exploratory data analysis (EDA) and machine learning model development on a TikTok dataset. The goal is to uncover insights from the data and build predictive models.

## Project Structure

The project is organized into the following directories and scripts:

```
Tiktok_EDA/
├── README.md
├── requirements.txt
├── update_readme.py
├── .pytest_cache/
│   ├── CACHEDIR.TAG
│   ├── README.md
│   ├── v/
│   │   ├── cache/
│   │   │   ├── lastfailed
│   │   │   ├── nodeids
│   │   │   ├── stepwise
├── .vscode/
│   ├── settings.json
├── data/
│   ├── data_preprocessing.py
│   ├── tiktok_preprocessed.csv
│   ├── tiktok_top_1000.csv
├── DataVisualization/
│   ├── __init__.py
│   ├── data_visualization.py
│   ├── baseline_images/
│   │   ├── box_plot.png
│   │   ├── correlation_matrix.png
│   │   ├── distribution_plot.png
│   │   ├── pair_plot.png
│   │   ├── top_10_accounts_Comments avg..png
│   │   ├── top_10_accounts_Likes avg..png
│   │   ├── top_10_accounts_Shares avg..png
│   │   ├── top_10_accounts_Subscribers count.png
│   │   ├── top_10_accounts_Views avg..png
│   ├── outcomes/
│   │   ├── box_plot.png
│   │   ├── correlation_matrix.png
│   │   ├── distribution_plot.png
│   │   ├── pair_plot.png
│   │   ├── test_plot.png
│   │   ├── top_10_accounts_Comments avg..png
│   │   ├── top_10_accounts_Likes avg..png
│   │   ├── top_10_accounts_Shares avg..png
│   │   ├── top_10_accounts_Subscribers count.png
│   │   ├── top_10_accounts_Views avg..png
├── EnsembleModels/
│   ├── EnsembleModels.py
│   ├── Model_feature_eng_and_validation.py
│   ├── Post_cv_feature_eng.py
│   ├── __init__.py
│   ├── outcomes/
│   │   ├── Stacking Classifier (Test Set)_confusion_matrix.png
│   │   ├── Stacking Classifier with New Features_confusion_matrix.png
│   │   ├── Stacking Classifier_confusion_matrix.png
│   │   ├── learning_curves.png
│   │   ├── roc_curve_Stacking Classifier (Test Set).png
│   │   ├── roc_curve_Stacking Classifier with New Features.png
│   │   ├── roc_curve_Stacking Classifier.png
├── KNN/
│   ├── KNN.py
│   ├── __init__.py
│   ├── outcomes/
│   │   ├── knn_cm_test_set_test_script.png
│   │   ├── knn_cm_validation_set.png
│   │   ├── knn_confusion_matrix_combined.png
│   │   ├── knn_roc_test.png
│   │   ├── knn_roc_validation.png
├── K_means/
│   ├── __init__.py
│   ├── k_means.py
│   ├── outcomes/
│   │   ├── k-means_clusters.png
├── RF_SVM/
│   ├── RF_SVM.py
│   ├── __init__.py
│   ├── outcomes/
│   │   ├── Optimized Random Forest with New Features_confusion_matrix.png
│   │   ├── Optimized Random Forest_confusion_matrix.png
│   │   ├── Optimized SVM with New Features_confusion_matrix.png
│   │   ├── Optimized SVM_confusion_matrix.png
│   │   ├── Random Forest_confusion_matrix.png
│   │   ├── Support Vector Machine_confusion_matrix.png
│   │   ├── feature_importance.png
│   │   ├── roc_curve_Optimized RF with New Features.png
│   │   ├── roc_curve_Optimized SVM with New Features.png
│   │   ├── roc_curve_Random Forest.png
│   │   ├── roc_curve_SVM.png
├── RF_SVM_XGBOOST/
│   ├── RF_SVM_XGBOOST.py
│   ├── __init__.py
│   ├── outcomes/
│   │   ├── Optimized XGBoost_confusion_matrix.png
│   │   ├── XGBoost_confusion_matrix.png
│   │   ├── roc_curve_Optimized XGBoost.png
│   │   ├── roc_curve_XGBoost.png
├── src/
│   ├── __init__.py
│   ├── main.py
├── test/
│   ├── test_data_preprocessing.py
│   ├── test_data_visualization.py
│   ├── test_ensemble_models.py
│   ├── test_ensemble_models_mev.py
│   ├── test_ensemble_models_post_cv.py
│   ├── test_k_means.py
│   ├── test_knn.py
│   ├── test_main.py
│   ├── test_rf_svm.py
│   ├── test_rf_svm_xgboost.py
```

## Installation
To run this project, you'll need to have Python installed along with several libraries. You can install the required libraries using the following command:

```sh
pip install -r requirements.txt