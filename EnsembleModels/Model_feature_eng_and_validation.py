import os
import sys
import matplotlib
matplotlib.use('agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, learning_curve
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif

# Add the root directory of the project to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.main import load_and_preprocess_data, create_new_features, prepare_data, evaluate_model, plot_roc_curve

def perform_feature_selection(X_train, y_train, X_test, k=5):
    """Select top k features based on ANOVA F-value."""
    selector = SelectKBest(f_classif, k=k)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    return X_train_selected, X_test_selected

def optimize_random_forest(X_train, y_train):
    """Perform grid search to optimize Random Forest hyperparameters."""
    rf = RandomForestClassifier(random_state=42)
    param_grid_rf = {
        'n_estimators': [100, 200, 300],
        'max_features': ['sqrt', 'log2'],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }
    grid_search_rf = GridSearchCV(estimator=rf, param_grid=param_grid_rf, 
                                  cv=StratifiedKFold(n_splits=3), n_jobs=-1, verbose=2)
    grid_search_rf.fit(X_train, y_train)
    return grid_search_rf.best_estimator_

def create_stacking_classifier(best_rf):
    """Create a stacking classifier with RF, SVM, and XGBoost."""
    svm = SVC(probability=True, random_state=42)
    xgb = XGBClassifier(random_state=42)
    return StackingClassifier(
        estimators=[('rf', best_rf), ('svm', svm), ('xgb', xgb)],
        final_estimator=LogisticRegression()
    )

def plot_learning_curves(estimator, X, y, output_dir):
    """Plot and save learning curves for the estimator."""
    cv = StratifiedKFold(n_splits=10)
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10)
    )

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    val_scores_mean = np.mean(val_scores, axis=1)
    val_scores_std = np.std(val_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training score')
    plt.plot(train_sizes, val_scores_mean, 'o-', color='g', label='Cross-validation score')
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color='r')
    plt.fill_between(train_sizes, val_scores_mean - val_scores_std, val_scores_mean + val_scores_std, alpha=0.1, color='g')
    plt.xlabel('Training examples')
    plt.ylabel('Score')
    plt.title('Learning Curves')
    plt.legend(loc='best')
    plt.savefig(os.path.join(output_dir, 'learning_curves.png'))
    plt.close()

def main():
    # Set output directory with an absolute path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "outcomes")
    os.makedirs(output_dir, exist_ok=True)

    # Load and prepare data
    tiktok_data = load_and_preprocess_data()
    tiktok_data = create_new_features(tiktok_data)    
    features = ['Subscribers count', 'Views avg.', 'Comments avg.', 'Shares avg.', 
                'Likes_per_view', 'Comments_per_view', 'Shares_per_view', 'Engagement_rate']
    tiktok_data['Engagement'] = (tiktok_data['Likes avg.'] > tiktok_data['Likes avg.'].median()).astype(int)
    
    X_train, X_test, y_train, y_test = prepare_data(tiktok_data, features)

    # Perform feature selection
    X_train, X_test = perform_feature_selection(X_train, y_train, X_test)

    # Optimize Random Forest
    best_rf = optimize_random_forest(X_train, y_train)

    # Create and train Stacking Classifier
    stacking_clf = create_stacking_classifier(best_rf)
    stacking_clf.fit(X_train, y_train)

    # Evaluate the model on the test set (includes saving confusion matrix)
    evaluate_model(stacking_clf, X_test, y_test, "Stacking Classifier (Test Set)", save_path=output_dir)

    # Plot ROC curve
    plot_roc_curve({"Stacking Classifier (Test Set)": stacking_clf}, X_test, y_test, save_path=output_dir)

    # Plot learning curves
    plot_learning_curves(stacking_clf, X_train, y_train, output_dir)

    print("Model training and evaluation completed. Results saved in:", output_dir)

if __name__ == "__main__":
    main()