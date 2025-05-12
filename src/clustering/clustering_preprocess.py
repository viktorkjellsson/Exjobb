from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go

def load_data(path):
    """
    Description: Load data from a csv file containing strain distributions.

    Args:
        path (path object): The path to the csv file.

    Returns:
        df (pd DataFrame): The data loaded from the csv file.
    """
    df = pd.read_csv(path)
    df.isna().sum().sum()

    return df


def drop_columns_by_header_rules(df, threshold):
    first_col = df.columns[0]
    cols_to_drop = []

    for col in df.columns[1:]:  # Skip first column
        # Check if the column name contains more than one dot
        if str(col).count('.') > 1:
            cols_to_drop.append(col)
            continue

        # Try to convert to float and check if above threshold
        try:
            if float(col) < threshold:
                cols_to_drop.append(col)
        except ValueError:
            continue  # Skip if not convertible

    return df.drop(columns=cols_to_drop)


def remove_outliers(df, threshold, individual_threshold):
    """
    Description: Remove outliers from the data in three ways:
        1. Remove rows where the mean exceeds the threshold deviation from the overall mean based on absolute values.
        2. Remove rows where any value in the row exceeds the threshold based on its own mean (using absolute values).
        3. Remove individual values in rows that are too large in relation to the rest of the values along the same row.

    Args:
        df (pd DataFrame): The data loaded from the csv file.
        threshold (float): The threshold for determining outliers based on the mean. Default is 1.
        individual_threshold (float): The threshold for individual values compared to the row's mean. Default is 10.

    Returns:
        df_strain (pd DataFrame): The data cleaned of outliers, without the timestamp column.
        df (pd DataFrame): The data cleaned of outliers, with the timestamp column.
    """

    # Drop rows with any NaNs
    df = df.dropna()
    
    df_strain = df.drop(columns='Timestamp')
    abs_df = df_strain.abs()  # Take the absolute values of the dataframe

    # 1. Remove rows where the mean is above the threshold deviation from the overall mean (based on absolute values)
    row_means = abs_df.mean(axis=1)
    Q1 = row_means.quantile(0.25)
    Q3 = row_means.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    mean_outliers = row_means[(row_means < lower_bound) | (row_means > upper_bound)]

    # print("Mean-based outliers using IQR:")
    # print(mean_outliers)

    # 2. Remove rows where any value exceeds the threshold based on the row's mean (using absolute values)
    row_outliers = []
    for idx, row in abs_df.iterrows():
        row_mean = row.mean()  # Mean of the absolute values in the row
        row_std = row.std()  # Standard deviation of the absolute values in the row
        threshold_value = row_mean + threshold * row_std  # Define threshold for each row based on absolute values
        
        # Check if any value in the row exceeds the calculated threshold (absolute value)
        if (row > threshold_value).any():
            row_outliers.append(idx)  # Keep track of the outlier row indices
    
    # print("Outliers based on row threshold (absolute values):")
    # print(df_strain.loc[row_outliers])

    # 3. Remove individual values that are too large in relation to the row mean
    large_value_outliers = []
    for idx, row in abs_df.iterrows():
        row_mean = row.mean()  # Mean of the absolute values in the row
        row_max = row.max()  # Max value in the row
        
        # Compare the maximum value to the row mean; if it exceeds the threshold, flag it
        if row_max > individual_threshold * row_mean:
            large_value_outliers.append(idx)
    
    # print("Outliers based on individual large values in relation to row mean:")
    # print(df_strain.loc[large_value_outliers])

    # Combine all outliers (mean-based, row-based, and individual large value-based) and drop them
    all_outliers = mean_outliers.index.union(row_outliers).union(large_value_outliers)
    
    # Remove the rows from the data
    df_strain = df_strain.drop(all_outliers)  # Removed outliers without timestamp
    df = df.drop(all_outliers)  # Removed outliers with timestamp
    
    print(f"Total number of outliers removed: {len(all_outliers)}")

    return df_strain, df

def explain_variance(df_strain) -> None:  
    """
    Descripition: Plots the explained variance by number of components for PCA.

    Args:
        df_strain (pd DataFrame): The data.

    Returns:
        None    
    """
    # Fit PCA on the entire strain data (matrix-wise)
    # Set the number of components directly (e.g., 5 components)
    pca = PCA(n_components=10)
    pca.fit(df_strain)
    # 
    # Get the explained variance ratio
    per_var = np.round(pca.explained_variance_ratio_ * 100, decimals=1)

    # Plot the cumulative explained variance
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(per_var) + 1), per_var.cumsum(), marker="o", linestyle="--")
    plt.grid()
    plt.ylabel("Percentage Cumulative of Explained Variance")
    plt.xlabel("Number of Principal Components")
    plt.xticks(range(1, len(per_var) + 1, 1))
    plt.title("Explained Variance by Number of Components")
    plt.show()


def do_pca(n_components, df_strain, df):
    """
    Description: Perform PCA on the data.

    Args:
        n_components (int): The number of principal components to keep.
        df_strain (pd DataFrame): The data.

    Returns:
        df_pca (pd DataFrame): The principal components for each timestamp.
        normalized_pca_components (np array) : The normalized principal components.
    """
    # Perform PCA
    pca = PCA(n_components=n_components)

    # Fit PCA on the entire strain data (matrix-wise)
    pca.fit(df_strain)

    # Apply PCA to the entire strain data (matrix-wise)
    pca_results = pca.transform(df_strain)

    # Normalize the results
    normalized_pca_components = StandardScaler().fit_transform(pca_results)
    # normalized_pca_components = MinMaxScaler().fit_transform(pca_results)

    # Convert results into a DataFrame
    df_pca = pd.DataFrame(normalized_pca_components, columns=[f'PC{i+1}' for i in range(n_components)])

    # Add timestamps back
    df_pca.insert(0, 'Timestamp', df['Timestamp'].values)       #2009-11-30 040000
    # df_pca.insert(0, '2009-11-30 040000', df['2009-11-30 040000'].values)

    return normalized_pca_components, df_pca