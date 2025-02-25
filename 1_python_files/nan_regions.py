import pandas as pd

def find_nan_regions(df, threshold):
    """
    Find regions with more than 'threshold' consecutive NaNs in the 'Strain' column of the DataFrame.

    Args: 
        df (pd.DataFrame): The input DataFrame.
        threshold (int): Minimum number of consecutive NaN values to consider as a region.

    Returns:
        consecutive_nan_regions (list): List of tuples containing start and end indices of consecutive NaN regions.
        nan_regions_sorted (list): Sorted list including start & end times, indices, and length.
    """
    nan_indices = df[df['Strain'].isna()].index.tolist()
    consecutive_nan_regions = []
    start_idx = None

    for i in range(len(nan_indices)):
        if start_idx is None:
            start_idx = nan_indices[i]
        if i == len(nan_indices) - 1 or nan_indices[i] + 1 != nan_indices[i + 1]:
            end_idx = nan_indices[i]
            if (end_idx - start_idx + 1) >= threshold:
                consecutive_nan_regions.append((start_idx, end_idx))
            start_idx = None

    nan_regions_sorted = [
        (start, end, df.loc[start, 'Time'], df.loc[end, 'Time'], end - start + 1)
        for start, end in consecutive_nan_regions
    ]
    
    nan_regions_sorted.sort(key=lambda x: x[4], reverse=True)

    return consecutive_nan_regions, nan_regions_sorted
