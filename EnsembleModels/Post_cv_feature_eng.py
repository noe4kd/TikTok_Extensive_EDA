import os
import sys
import matplotlib
matplotlib.use('agg')  # Use 'agg' backend to save figures without displaying them
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Add the root directory of the project to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.main import load_and_preprocess_data, create_new_features, prepare_data, evaluate_model, plot_roc_curve

def perform_grid_search_rf(X_train, y_train):
    """Perform hyperparameter tuning for Random Forest."""
    rf = RandomForestClassifier(random_state=42)
    
    param_grid_rf = {
        'n_estimators': [100, 200, 300],
        'max_features': ['sqrt', 'log2'],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }
    
    grid_search_rf = GridSearchCV(estimator=rf, param_grid=param_grid_rf, cv=3, n_jobs=-1, verbose=2)
    grid_search_rf.fit(X_train, y_train)
    
    return grid_search_rf.best_estimator_

def initialize_stacking_classifier(best_rf):
    """Initialize the stacking classifier with tuned Random Forest."""
    svm = SVC(probability=True, random_state=42)
    xgb = XGBClassifier(random_state=42)
    
    stacking_clf = StackingClassifier(
        estimators=[('rf', best_rf), ('svm', svm), ('xgb', xgb)], 
        final_estimator=LogisticRegression()
    )
    
    return stacking_clf

def main():
    # Set output directory with an absolute path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "outcomes")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and prepare data
    tiktok_data = load_and_preprocess_data()
    tiktok_data = create_new_features(tiktok_data)
    
    new_features = [
        'Subscribers count', 'Views avg.', 'Comments avg.', 'Shares avg.', 
        'Likes_per_view', 'Comments_per_view', 'Shares_per_view', 'Engagement_rate'
    ]
    tiktok_data['Engagement'] = (tiktok_data['Likes avg.'] > tiktok_data['Likes avg.'].median()).astype(int)
    X_train, X_test, y_train, y_test = prepare_data(tiktok_data, new_features)
    
    # Perform grid search for Random Forest
    best_rf = perform_grid_search_rf(X_train, y_train)
    
    # Initialize and train stacking classifier
    stacking_clf = initialize_stacking_classifier(best_rf)
    stacking_clf.fit(X_train, y_train)
    
    # Evaluate model and plot results
    evaluate_model(stacking_clf, X_test, y_test, "Stacking Classifier with New Features", save_path=output_dir)
    plot_roc_curve({"Stacking Classifier with New Features": stacking_clf}, X_test, y_test, save_path=output_dir)
    
    print("Model evaluation and plots saved to:", output_dir)

if __name__ == "__main__":
    main()
