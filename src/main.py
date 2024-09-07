import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, RocCurveDisplay
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_preprocess_data():
    """Load the preprocessed data."""
    # Construct the correct path to the preprocessed CSV file
    current_dir = os.path.dirname(__file__)
    file_path = os.path.join(current_dir, '..', 'data', 'tiktok_preprocessed.csv')
    
    # Load the preprocessed data
    try:
        tiktok_data = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        raise
    except pd.errors.EmptyDataError:
        print(f"Error: The file {file_path} is empty.")
        raise
    
    return tiktok_data


def create_new_features(data):
    """Create new features for the dataset."""
    data['Likes_per_view'] = data['Likes avg.'] / data['Views avg.']
    data['Comments_per_view'] = data['Comments avg.'] / data['Views avg.']
    data['Shares_per_view'] = data['Shares avg.'] / data['Views avg.']
    data['Engagement_rate'] = (data['Likes avg.'] + data['Comments avg.'] + data['Shares avg.']) / data['Views avg.']
    
    # Convert Engagement_rate into a categorical variable for classification (e.g., quartiles)
    data['Engagement'] = pd.qcut(data['Engagement_rate'], q=4, labels=False) 
    return data

def prepare_data(data, features, target='Engagement'):
    """Prepare data for training and testing."""
    X = data[features]
    y = data[target]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_scaled, y)
    return train_test_split(X_res, y_res, test_size=0.2, random_state=42)

def evaluate_model(model, X_test, y_test, model_name, save_path=None):
    """Evaluate the model and save the confusion matrix plot if save_path is provided."""
    y_pred = model.predict(X_test)
    print(f"{model_name} Classifier Report:\n")
    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{model_name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    if save_path:
        plt.savefig(os.path.join(save_path, f'{model_name}_confusion_matrix.png'))
    plt.close()

def plot_roc_curve(models, X_test, y_test, save_path=None):
    """Plot ROC curve for given models and save the plot if save_path is provided."""
    for model_name, model in models.items():
        plt.figure(figsize=(10, 6))
        RocCurveDisplay.from_estimator(model, X_test, y_test, name=model_name)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC - {model_name}')
        plt.legend(loc='lower right')
        if save_path:
            if os.path.isdir(save_path):
                plt.savefig(os.path.join(save_path, f'roc_curve_{model_name}.png'))
            else:
                plt.savefig(save_path)
        plt.clf()
        plt.close()

def plot_confusion_matrix(y_test, y_pred, model_name, save_path=None, class_names=['0', '1']):
    """Plot and save the confusion matrix with color bar and class labels."""
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, 
                xticklabels=class_names, yticklabels=class_names)

    plt.title(f'{model_name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    if save_path:
        plt.savefig(os.path.join(save_path, f'{model_name}_confusion_matrix.png'))
    
    plt.show()
    plt.close()
