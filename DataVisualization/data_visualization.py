import warnings
import sys
import os
import matplotlib
matplotlib.use('agg') 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Add the root directory of the project to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.main import load_and_preprocess_data

# Suppress warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def initial_analysis(data):
    """Perform initial analysis: display head, summary statistics, and missing values."""
    summary_stats = data.describe()
    missing_values = data.isnull().sum()
    return summary_stats, missing_values

def plot_distributions(data, numerical_columns, save_path):
    """Plot and save distributions of key numerical columns."""
    colors = plt.cm.tab20(np.linspace(0, 1, len(numerical_columns)))
    plt.figure(figsize=(14, 9))
    for i, (col, color) in enumerate(zip(numerical_columns, colors), 1):
        plt.subplot(3, 2, i)
        sns.histplot(data[col], bins=45, kde=False, color=color, edgecolor='black')
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'distribution_plot.png'))
    plt.close()

def plot_correlations(data, numerical_columns, save_path):
    """Plot and save a heatmap of correlations between numerical variables."""
    correlations = data[numerical_columns].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlations, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Matrix')
    plt.savefig(os.path.join(save_path, 'correlation_matrix.png'))
    plt.close()

def plot_top_10_accounts(data, metric, save_path):
    """Plot and save top 10 accounts by a specified metric."""
    top_accounts = data.nlargest(10, metric)[['Account', metric]]
    plt.figure(figsize=(10, 6))
    sns.barplot(x=metric, y='Account', data=top_accounts, palette='viridis')
    plt.xlabel(f'{metric.replace("avg.", "Average")}')
    plt.title(f'Top 10 Accounts by {metric.replace("avg.", "Average")}')
    plt.gca().invert_yaxis()
    plt.savefig(os.path.join(save_path, f'top_10_accounts_{metric}.png'))
    plt.close()

def plot_boxplots(data, numerical_columns, save_path):
    """Plot and save box plots of key numerical columns."""
    colors = sns.color_palette("husl", len(numerical_columns)) 
    plt.figure(figsize=(14, 9))
    for i, (col, color) in enumerate(zip(numerical_columns, colors), 1):
        plt.subplot(3, 2, i)
        sns.boxplot(x=data[col], color=color)
        plt.title(f'Box Plot of {col}')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'box_plot.png'))
    plt.close()

def plot_pairplot(data, numerical_columns, save_path):
    """Plot and save pair plot of numerical columns."""
    sns.pairplot(data[numerical_columns], height=2)
    plt.savefig(os.path.join(save_path, 'pair_plot.png'))
    plt.close()

def main():
    save_path = os.path.join('DataVisualization', 'outcomes')
    os.makedirs(save_path, exist_ok=True)
    
    # Load the preprocessed dataset using the function from main.py
    tiktok_data = load_and_preprocess_data()
    
    # Initial analysis
    initial_analysis(tiktok_data)
    
    # Plot distributions
    numerical_columns = ['Subscribers count', 'Views avg.', 'Likes avg.', 'Comments avg.', 'Shares avg.']
    plot_distributions(tiktok_data, numerical_columns, save_path)
    
    # Plot correlations
    plot_correlations(tiktok_data, numerical_columns, save_path)
    
    # Plot top 10 accounts by various metrics
    for metric in numerical_columns:
        plot_top_10_accounts(tiktok_data, metric, save_path)
    
    # Additional visualizations
    plot_boxplots(tiktok_data, numerical_columns, save_path)
    plot_pairplot(tiktok_data, numerical_columns, save_path)

if __name__ == "__main__":
    main()
