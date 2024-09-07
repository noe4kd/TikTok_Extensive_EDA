import os
import sys
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Add the root directory of the project to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.main import load_and_preprocess_data, create_new_features, prepare_data, plot_roc_curve

def create_engagement_variable(df, column='Likes avg.'):
    """Creates a binary target variable for classification based on median."""
    median_value = df[column].median()
    df['Engagement'] = (df[column] > median_value).astype(int)
    return df

def plot_confusion_matrix(cm, title, filename, output_dir):
    """Plots and saves the confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(os.path.join(output_dir, filename))
    plt.clf()

def evaluate_model(model, X, y, dataset_name, output_dir, custom_filename=None):
    """Evaluates the model and saves confusion matrix and classification report."""
    y_pred = model.predict(X)
    print(f"{model.__class__.__name__} Report on {dataset_name}:\n")
    print(classification_report(y, y_pred))
    cm = confusion_matrix(y, y_pred)
    
    filename = custom_filename if custom_filename else f'knn_cm_{dataset_name.lower().replace(" ", "_")}.png'
    
    plot_confusion_matrix(cm, f'{model.__class__.__name__} Confusion Matrix ({dataset_name})', filename, output_dir)
    return y_pred

def optimize_knn(X_train, y_train):
    """Perform GridSearchCV to find the best hyperparameters for KNN."""
    param_grid = {
        'n_neighbors': range(1, 31),
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski']
    }
    knn = KNeighborsClassifier()
    grid_search = GridSearchCV(knn, param_grid, cv=2, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    print(f"Best parameters found: {grid_search.best_params_}")
    return grid_search.best_estimator_

def optimize_knn(X_train, y_train):
    """Perform RandomizedSearchCV to find the best hyperparameters for KNN."""
    param_grid = {
        'n_neighbors': range(1, 31),  # Larger range for n_neighbors
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski'],
        'p': [1, 2],  # Minkowski parameter
        'leaf_size': np.arange(20, 50, 5),  # Exploring different leaf sizes
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']  # Different algorithms
    }
    
    knn = KNeighborsClassifier()
    random_search = RandomizedSearchCV(knn, param_distributions=param_grid, n_iter=50, cv=5, scoring='accuracy', n_jobs=-1, random_state=42)
    random_search.fit(X_train, y_train)
    print(f"Best parameters found: {random_search.best_params_}")
    return random_search.best_estimator_

def main():
    # Set output directory with an absolute path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "outcomes")
    os.makedirs(output_dir, exist_ok=True)

    # Load and preprocess data
    tiktok_data = load_and_preprocess_data()

    # Create new features
    tiktok_data = create_new_features(tiktok_data)

    # Create engagement variable
    tiktok_data = create_engagement_variable(tiktok_data)

    # Select features for modeling
    features = [
        'Subscribers count', 'Views avg.', 'Comments avg.', 'Shares avg.', 
        'Likes_per_view', 'Comments_per_view', 'Shares_per_view', 'Engagement_rate'
    ]

    # Prepare the data (split into training, testing, and validation sets)
    X_train, X_temp, y_train, y_temp = prepare_data(tiktok_data, features, target='Engagement')
    X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_val = scaler.transform(X_val)

    # Add polynomial features
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    X_val_poly = poly.transform(X_val)

    # Optimize KNN classifier with GridSearchCV on polynomial features
    knn = optimize_knn(X_train_poly, y_train)
    
    # Evaluate model on test set
    y_pred_knn_test = evaluate_model(knn, X_test_poly, y_test, 'Test Set', output_dir)

    # Evaluate model on validation set
    y_pred_knn_val = evaluate_model(knn, X_val_poly, y_val, 'Validation Set', output_dir)

    # Combine predictions and actuals from both test and validation sets
    y_combined = pd.concat([pd.Series(y_test), pd.Series(y_val)])
    y_pred_combined = pd.concat([pd.Series(y_pred_knn_test), pd.Series(y_pred_knn_val)])

    # Generate and plot combined confusion matrix
    cm_combined = confusion_matrix(y_combined, y_pred_combined)
    plot_confusion_matrix(cm_combined, 'k-NN Classifier Confusion Matrix (Combined)', 
                          'knn_confusion_matrix_combined.png', output_dir)

    # Plot ROC curve for k-NN Classifier on test and validation sets
    plot_roc_curve({'k-NN Classifier': knn}, X_test_poly, y_test, save_path=os.path.join(output_dir, 'knn_roc_test.png'))
    plot_roc_curve({'k-NN Classifier': knn}, X_val_poly, y_val, save_path=os.path.join(output_dir, 'knn_roc_validation.png'))

if __name__ == "__main__":
    main()
