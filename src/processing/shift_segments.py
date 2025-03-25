"""
Handles the shift along the y-axis and scale change after an interruption by translating each segment 
to a common baseline and scaling each segment to allign the data to a common range.
 
"""
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def shift_scale(df):    #, scale_range=(0, 1)):
    # Finding the regions with valid values
    valid_indices = df[df['Strain'].notna()].index.tolist()
    consecutive_valid_regions = []
    start_idx = None

    for i in range(len(valid_indices)):
        if start_idx is None:
            start_idx = valid_indices[i]
        if i == len(valid_indices) - 1 or valid_indices[i] + 1 != valid_indices[i + 1]:
            end_idx = valid_indices[i]
            if (end_idx - start_idx + 1) >= 1:  # Threshold for region length
                consecutive_valid_regions.append((start_idx, end_idx))
            start_idx = None



    # Shifting and scaling each region
    for region in consecutive_valid_regions:
        start_idx, end_idx = region


        # Calculate the mean of the region
        mean_of_region = np.mean(df.loc[start_idx:end_idx, 'Strain'])           # redo this so that it calculates the difference in strain 
        
        # Subtract the mean to shift the data
        df.loc[start_idx:end_idx, 'Strain'] -= mean_of_region
        

    
    return df



'''





    # Initialize MinMaxScaler
    scaler = MinMaxScaler(feature_range=scale_range)

            # Apply Min-Max scaling using sklearn's MinMaxScaler
        region_values = df.loc[start_idx:end_idx, 'Strain'].values.reshape(-1, 1)  # Reshaping for the scaler
        df.loc[start_idx:end_idx, 'Strain'] = scaler.fit_transform(region_values).flatten()  # Flatten back to original shape
        
        # Check the new min and max values (it should be within the desired range)
        new_min = df.loc[start_idx:end_idx, 'Strain'].min()
        new_max = df.loc[start_idx:end_idx, 'Strain'].max()'






'''