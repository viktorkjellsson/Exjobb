import numpy as np
import nan_regions #nan_regions.py
import interpolate_nan #interpolate_nan.py

def clean_zeros_outliers_interpolate(df, interpolate_threshold):

    #Find values of strain that are exactly zero
    df_zero = df[df["Strain"] == 0]
    print(f'{df_zero.shape[0]} zeros to replace with NaN')

    # Remove rows with strain values of zero by replacing with NaN
    df = df.copy()
    df.loc[df["Strain"] == 0, "Strain"] = np.nan

    # Compute IQR
    Q1 = df['Strain'].quantile(0.25)  # 25th percentile
    Q3 = df['Strain'].quantile(0.75)  # 75th percentile
    IQR = Q3 - Q1

    # Define bounds for extreme outliers
    lower_bound = Q1 - 1.5 * IQR
    lower_bound_extreme = Q1 - 3 * IQR
    upper_bound = Q3 + 1.5 * IQR
    upper_bound_extreme = Q3 + 3 * IQR

    mild_outliers = df[(df['Strain'] < lower_bound) | (df['Strain'] > upper_bound)]
    mild_outlier_indices = mild_outliers.index  # Save the indices of mild outliers
    num_mild_outliers = mild_outliers.shape[0]

    # Count extreme outliers (outside 3 * IQR)
    extreme_outliers = df[(df['Strain'] < lower_bound_extreme) | (df['Strain'] > upper_bound_extreme)]
    extreme_outlier_indices = extreme_outliers.index  # Save the indices of extreme outliers
    num_extreme_outliers = extreme_outliers.shape[0]

    # Print results
    print(f'Number of mild outliers (1.5 × IQR): {num_mild_outliers}')
    print(f'Number of extreme outliers (3 × IQR): {num_extreme_outliers}')

    # Replace extreme outliers with NaN
    df.loc[mild_outlier_indices, 'Strain'] = np.nan  # Using np.nan to replace the outlier values
    print(f'Number of outliers replaced with NaN: {len(mild_outlier_indices)}')

    consecutive_nan_regions, nan_regions_sorted = nan_regions.find_nan_regions(df, threshold=1)

    df_filled = interpolate_nan.interpolate(df, nan_regions_sorted, interpolate_threshold)
    df = df_filled

    return df