import os
import sys
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

# Add the root directory of the project to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.main import load_and_preprocess_data, create_new_features, prepare_data, evaluate_model, plot_roc_curve

def main():
    # Set output directory with an absolute path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "outcomes")
    os.makedirs(output_dir, exist_ok=True)

    # Load and preprocess data
    tiktok_data = load_and_preprocess_data()

    # Create new features
    tiktok_data = create_new_features(tiktok_data)
    
    # Select features for modeling
    features = ['Subscribers count', 'Views avg.', 'Comments avg.', 'Shares avg.']
    new_features = features + ['Likes_per_view', 'Comments_per_view', 'Shares_per_view']

    # Add target variable
    tiktok_data['Engagement'] = (tiktok_data['Likes avg.'] > tiktok_data['Likes avg.'].median()).astype(int)

    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data(tiktok_data, new_features)

    # Train XGBoost model
    xgb = XGBClassifier(random_state=42, eval_metric='logloss')
    xgb.fit(X_train, y_train)
    evaluate_model(xgb, X_test, y_test, "XGBoost", output_dir)
    plot_roc_curve({'XGBoost': xgb}, X_test, y_test, output_dir)

    # Hyperparameter tuning for XGBoost
    param_grid_xgb = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9]
    }

    grid_search_xgb = GridSearchCV(estimator=xgb, param_grid=param_grid_xgb, cv=3, n_jobs=-1, verbose=2)
    grid_search_xgb.fit(X_train, y_train)

    # Best estimator
    best_xgb = grid_search_xgb.best_estimator_
    evaluate_model(best_xgb, X_test, y_test, "Optimized XGBoost", output_dir)
    plot_roc_curve({'Optimized XGBoost': best_xgb}, X_test, y_test, output_dir)

if __name__ == "__main__":
    main()
