import os
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures

def review_csv_file(file_path):
    """Provides an overview of the CSV file, including total rows, column names, and missing values."""
    df = pd.read_csv(file_path)
    
    # Display total number of rows and column names
    total_rows = df.shape[0]
    print(f"Total rows: {total_rows}")
    print("Column names:", df.columns.tolist())

    # Display the number of missing values in each column
    missing_values = df.isnull().sum()
    print("\nMissing values in each column:")
    print(missing_values[missing_values > 0])
    
    # Summary of the numerical and categorical features
    print("\nNumerical features summary:")
    print(df.describe().T)
    
    print("\nCategorical features summary:")
    print(df.select_dtypes(include=['object']).describe().T)

    print(f"\nTotal missing values in the dataset: {df.isnull().sum().sum()}\n")
    
    return df

def preprocess_data():
    current_dir = os.path.dirname(__file__)

    # Load the dataset
    file_path = os.path.join(current_dir, 'tiktok_top_1000.csv')
    print(f"Loading data from: {file_path}")
    df = review_csv_file(file_path)

    # Preserve the 'Account' column for later use
    account_column = df['Account']

    # 1. Handling Missing Values
    num_features = df.select_dtypes(include=['int64', 'float64']).columns
    cat_features = df.select_dtypes(include=['object']).columns

    print("Initial summary of numerical features before imputation:")
    print(df[num_features].describe().T)

    num_imputer = SimpleImputer(strategy='mean')
    cat_imputer = SimpleImputer(strategy='most_frequent')

    df[num_features] = num_imputer.fit_transform(df[num_features])
    df[cat_features] = cat_imputer.fit_transform(df[cat_features])
    print("Missing values have been handled.")
    print(f"Number of missing values after imputation: {df.isnull().sum().sum()}")

    # 2. Data Cleaning
    # Remove duplicate rows
    df.drop_duplicates(inplace=True)
    print(f"Duplicates removed. {len(df)} rows remaining.")

    # 3. Feature Engineering
    # Create interaction terms between numerical features
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    interaction_terms = poly.fit_transform(df[num_features])
    interaction_df = pd.DataFrame(interaction_terms, columns=poly.get_feature_names_out(num_features))
    df = pd.concat([df, interaction_df], axis=1)
    print(f"Interaction terms created and added to the dataset. New shape: {df.shape}")
    print(f"Summary of interaction terms:")
    print(interaction_df.describe().T)

    # 4. Handle Infinite and Extremely Large Values
    # Replace infinity values with NaN, then impute or drop them
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    print(f"Number of NaN values introduced after replacing infinities: {df.isnull().sum().sum()}")
    
    if df.isnull().sum().sum() > 0:
        df.dropna(inplace=True) 
        print(f"Number of missing values after dropping rows with NaN: {df.isnull().sum().sum()}")
    print("Handled infinity or NaN values in the dataset.")

    # Normalize numerical features using StandardScaler
    scaler = StandardScaler()
    print("Using StandardScaler for normalization.")
    df[num_features] = scaler.fit_transform(df[num_features])
    print("Numerical features normalized.")
    print(f"Summary of numerical features after normalization:")
    print(df[num_features].describe().T)

    # 5. Data Transformation
    # Encode categorical features using OneHotEncoder
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    encoded_cat = encoder.fit_transform(df[cat_features])
    encoded_cat_df = pd.DataFrame(encoded_cat, columns=encoder.get_feature_names_out(cat_features))
    df = pd.concat([df, encoded_cat_df], axis=1).drop(columns=cat_features)
    print("Categorical variables encoded.")
    print(f"Shape of the dataframe after encoding categorical variables: {df.shape}")

    # Add back the Account column to the DataFrame
    df['Account'] = account_column

    return df

if __name__ == "__main__":
    processed_df = preprocess_data()
    current_dir = os.path.dirname(__file__)
    output_file_path = os.path.join(current_dir, 'tiktok_preprocessed.csv')
    processed_df.to_csv(output_file_path, index=False)
    print(f"Processed data saved to: {output_file_path}")
