import numpy as np

'''
A function that finds the outliers in the raw data and replaces them with NaN. It also replaces all values that are equal to zero with NaN aswell.

'''


def clean_zeros_outliers(df, k, window, step):

    """
    Removes zero-strain entries and replaces IQR-based outliers in both 'Strain' and 'Temperature' with NaN.
    
    Parameters:
    - df (pd.DataFrame): Input dataframe with 'Strain' and 'Temperature' columns
    - k (float): Multiplier for the IQR to determine outlier bounds
    - window (int): Size of the rolling window
    - step (int): Step size for the rolling window
    """

    # Remove rows with strain values of zero by replacing with NaN
    df = df.copy()
    df.loc[df["Strain"] == 0, "Strain"] = np.nan

    outlier_indices_strain = []
    outlier_indices_temp = []

    for start_idx in range(0, len(df) - window + 1, step):

        end_idx = start_idx + window
        window_data = df.iloc[start_idx:end_idx]

        # Compute IQR for the 'Strain' column
        Q1_strain = window_data['Strain'].quantile(0.25)  # 25th percentile
        Q3_strain = window_data['Strain'].quantile(0.75)  # 75th percentile
        IQR_strain = Q3_strain - Q1_strain

        # Define bounds for extreme outliers
        lower_bound_strain = Q1_strain - k * IQR_strain
        upper_bound_strain = Q3_strain + k * IQR_strain

        # Compute IQR for the 'Temperature' column
        Q1_temp = window_data['Temperature'].quantile(0.25)  # 25th percentile
        Q3_temp = window_data['Temperature'].quantile(0.75)  # 75th percentile
        IQR_temp = Q3_temp - Q1_temp

        # Define bounds for extreme outliers
        lower_bound_temp = Q1_temp - k * IQR_temp
        upper_bound_temp = Q3_temp + k * IQR_temp

        outliers_strain = window_data[(window_data['Strain'] < lower_bound_strain) | (window_data['Strain'] > upper_bound_strain)]
        outliers_temp = window_data[(window_data['Temperature'] < lower_bound_temp) | (window_data['Temperature'] > upper_bound_temp)]

        outlier_indices_strain += outliers_strain.index.tolist()
        outlier_indices_temp += outliers_temp.index.tolist()
    
    # Deduplicate indices before replacing
    outlier_indices_strain = list(set(outlier_indices_strain))
    outlier_indices_temp = list(set(outlier_indices_temp))

    # Replace outliers with NaN
    df.loc[outlier_indices_strain, 'Strain'] = np.nan  # Using np.nan to replace the outlier values
    df.loc[outlier_indices_temp, 'Temperature'] = np.nan  # Using np.nan to replace the outlier values
    print(f'Number of mild outliers replaced with NaN: \n  For strain: {len(outlier_indices_strain)}\n  For temperature: {len(outlier_indices_temp)}')

    return df
