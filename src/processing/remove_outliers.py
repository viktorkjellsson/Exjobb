import numpy as np

'''
A function that finds the outliers in the raw data and replaces them with NaN. It also replaces all values that are equal to zero with NaN aswell.

'''


def clean_zeros_outliers(df):

    # Remove rows with strain values of zero by replacing with NaN
    df = df.copy()
    df.loc[df["Strain"] == 0, "Strain"] = np.nan

    # Compute IQR for the 'Strain' column
    Q1_strain = df['Strain'].quantile(0.25)  # 25th percentile
    Q3_strain = df['Strain'].quantile(0.75)  # 75th percentile
    IQR_strain = Q3_strain - Q1_strain

    # Define bounds for extreme outliers
    lower_bound_strain = Q1_strain - 1.5 * IQR_strain
    lower_bound_extreme_strain = Q1_strain - 3 * IQR_strain
    upper_bound_strain = Q3_strain + 1.5 * IQR_strain
    upper_bound_extreme_strain = Q3_strain + 3 * IQR_strain

    # Compute IQR for the 'Temperature' column
    Q1_temp = df['Temperature'].quantile(0.25)  # 25th percentile
    Q3_temp = df['Temperature'].quantile(0.75)  # 75th percentile
    IQR_temp = Q3_temp - Q1_temp

    # Define bounds for extreme outliers
    lower_bound_temp = Q1_temp - 1.5 * IQR_temp
    lower_bound_extreme_temp = Q1_temp - 3 * IQR_temp
    upper_bound_temp = Q3_temp + 1.5 * IQR_temp
    upper_bound_extreme_temp = Q3_temp + 3 * IQR_temp

    # Count mild outliers (outside 1.5 * IQR)
    mild_outliers_strain = df[(df['Strain'] < lower_bound_strain) | (df['Strain'] > upper_bound_strain)]
    mild_outliers_temp = df[(df['Temperature'] < lower_bound_temp) | (df['Temperature'] > upper_bound_temp)]
    mild_outlier_indices_strain = mild_outliers_strain.index
    mild_outlier_indices_temp = mild_outliers_temp.index

    # Count extreme outliers (outside 3 * IQR)
    extreme_outliers_strain = df[(df['Strain'] < lower_bound_extreme_strain) | (df['Strain'] > upper_bound_extreme_strain)]
    extreme_outliers_temp = df[(df['Temperature'] < lower_bound_extreme_temp) | (df['Temperature'] > upper_bound_extreme_temp)]
    extreme_outlier_indices_strain = extreme_outliers_strain.index 
    extreme_outlier_indices_temp = extreme_outliers_temp.index

    # Replace outliers with NaN
    df.loc[mild_outlier_indices_strain, 'Strain'] = np.nan  # Using np.nan to replace the outlier values
    df.loc[mild_outlier_indices_temp, 'Temperature'] = np.nan  # Using np.nan to replace the outlier values
    print(f'Number of mild outliers replaced with NaN: \n  For strain: {len(mild_outlier_indices_strain)}\n  For temperature: {len(mild_outlier_indices_temp)}')
    # print(f'Number of outliers replaced with NaN: {len(mild_outlier_indices)}')


    return df
