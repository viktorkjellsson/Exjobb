
import numpy as np
from src.processing import remove_outliers
from src.processing import nan_regions #nan_regions.py
from src.processing import shift_region
from src.processing import interpolate_nan #interpolate_nan.py

def preprocessing_pipeline(df):

    # Remove zeros and outliers
    df = remove_outliers.clean_zeros_outliers(df)

    # Find consecutive NaN regions
    threshold = 1 # Choose the threshold(s) for consecutive NaNs
    consecutive_nan_regions, nan_regions_sorted = nan_regions.find_nan_regions(df, threshold)

    print(nan_regions_sorted)

    # Shift regions 
    n_points = 15
    std_multiplier = 4.6
    min_region_size = 5

    df = shift_region.shift_scale_diff(df, std_multiplier, n_points, n_points, min_region_size)

    # Interpolate NaN regions shorter than the threshold
    interpolate_threshold = 5

    df_filled = interpolate_nan.interpolate(df, nan_regions_sorted, interpolate_threshold)

    df = df_filled

    # Add function that scales the data here
    # df = scale_data(df)

    return df
