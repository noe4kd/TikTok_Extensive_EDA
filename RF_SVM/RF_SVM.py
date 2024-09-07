import os
import sys
import pandas as pd
import matplotlib
matplotlib.use('agg') 
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

# Add the root directory of the project to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.main import load_and_preprocess_data, create_new_features, prepare_data, evaluate_model, plot_roc_curve

def train_model(model, X_train, y_train):
    """Train a model."""
    model.fit(X_train, y_train)
    return model

def hyperparameter_tuning(model, param_grid, X_train, y_train):
    """Perform hyperparameter tuning using GridSearchCV."""
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

def plot_feature_importance(model, features, save_path=None):
    """Plot feature importance of the model and save the plot if save_path is provided."""
    feature_importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances}).sort_values(by='Importance', ascending=False)
    plt.figure(figsize=(10, 6))
    colors = plt.cm.viridis(feature_importance_df['Importance'] / max(feature_importance_df['Importance']))
    plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color=colors)
    plt.xlabel('Feature Importance')
    plt.title('Feature Importance for Optimized Random Forest Model')
    plt.gca().invert_yaxis()
    if save_path:
        plt.savefig(os.path.join(save_path, 'feature_importance.png'))
    plt.clf()
    plt.close()
    return feature_importance_df

def main():
    # Set output directory with an absolute path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "outcomes")
    os.makedirs(output_dir, exist_ok=True)

    # Load and preprocess data
    tiktok_data = load_and_preprocess_data()

    # Create new features and the binary target variable 'Engagement'
    tiktok_data = create_new_features(tiktok_data)
    tiktok_data['Engagement'] = (tiktok_data['Likes avg.'] > tiktok_data['Likes avg.'].median()).astype(int)

    # Select features for modeling
    features = ['Subscribers count', 'Views avg.', 'Comments avg.', 'Shares avg.']
    new_features = features + ['Likes_per_view', 'Comments_per_view', 'Shares_per_view']

    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data(tiktok_data, features)
    X_train_new, X_test_new, y_train_new, y_test_new = prepare_data(tiktok_data, new_features)

    # Train and evaluate Random Forest model
    rf_model = train_model(RandomForestClassifier(random_state=42), X_train, y_train)
    evaluate_model(rf_model, X_test, y_test, "Random Forest", output_dir)
    
    # Train and evaluate SVM model
    svm_model = train_model(SVC(probability=True, random_state=42), X_train, y_train)
    evaluate_model(svm_model, X_test, y_test, "Support Vector Machine", output_dir)

    # Plot ROC curves for initial models
    plot_roc_curve({'Random Forest': rf_model, 'SVM': svm_model}, X_test, y_test, output_dir)

    # Hyperparameter tuning for Random Forest
    param_grid_rf = {
        'n_estimators': [100, 200, 300],
        'max_features': ['sqrt', 'log2'],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }
    best_rf = hyperparameter_tuning(rf_model, param_grid_rf, X_train, y_train)
    evaluate_model(best_rf, X_test, y_test, "Optimized Random Forest", output_dir)

    # Train and evaluate the optimized Random Forest model with new features
    best_rf_new = train_model(best_rf, X_train_new, y_train_new)
    evaluate_model(best_rf_new, X_test_new, y_test_new, "Optimized Random Forest with New Features", output_dir)

    # Plot ROC curve for the optimized Random Forest model with new features
    plot_roc_curve({'Optimized RF with New Features': best_rf_new}, X_test_new, y_test_new, output_dir)

    # Hyperparameter tuning for SVM
    param_grid_svm = {
        'C': [0.1, 1, 10, 100],
        'gamma': [1, 0.1, 0.01, 0.001],
        'kernel': ['rbf', 'linear']
    }
    best_svm = hyperparameter_tuning(SVC(probability=True, random_state=42), param_grid_svm, X_train, y_train)
    evaluate_model(best_svm, X_test, y_test, "Optimized SVM", output_dir)

    # Train and evaluate the optimized SVM model with new features
    best_svm_new = train_model(best_svm, X_train_new, y_train_new)
    evaluate_model(best_svm_new, X_test_new, y_test_new, "Optimized SVM with New Features", output_dir)

    # Plot ROC curve for the optimized SVM model with new features
    plot_roc_curve({'Optimized SVM with New Features': best_svm_new}, X_test_new, y_test_new, output_dir)

    # Plot feature importance for Random Forest model
    feature_importance_df = plot_feature_importance(best_rf_new, new_features, output_dir)
    print(feature_importance_df)

if __name__ == "__main__":
    main()