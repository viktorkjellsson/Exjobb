
import pandas as pd



def shift_scale_diff(df, std_multiplier, n_points_prev, n_points_curr, min_region_size):
    # Finding the regions with valid values
    valid_indices = df[df['Strain'].notna()].index.tolist()
    consecutive_valid_regions = []
    start_idx = None

    for i in range(len(valid_indices)):
        if start_idx is None:
            start_idx = valid_indices[i]
        if i == len(valid_indices) - 1 or valid_indices[i] + 1 != valid_indices[i + 1]:
            end_idx = valid_indices[i]
            consecutive_valid_regions.append((start_idx, end_idx))
            start_idx = None

    total_shift = 0  # Accumulated shift applied to all following regions

    for i in range(1, len(consecutive_valid_regions)):
        prev_start_idx, prev_end_idx = consecutive_valid_regions[i - 1]
        start_idx, end_idx = consecutive_valid_regions[i]

        print(df.loc[start_idx:end_idx, 'Strain'])

        # Apply the total accumulated shift to the current region at the start
        df.loc[start_idx:end_idx, 'Strain'] += total_shift  

        # Reset delta_shift for this iteration
        delta_shift = 0  

        # Get size of previous and current regions
        prev_region_size = len(df.loc[prev_start_idx:prev_end_idx])
        curr_region_size = len(df.loc[start_idx:end_idx])

        # Compute mean for the last `n_points_prev` of the previous region
        prev_region_indices = df.loc[prev_start_idx:prev_end_idx].index
        selected_prev_indices = prev_region_indices[-n_points_prev:] if len(prev_region_indices) > n_points_prev else prev_region_indices
        selected_prev_values = df.loc[selected_prev_indices, 'Strain']

        previous_region_mean = selected_prev_values.mean()

        
        if prev_region_size < min_region_size and i > 1:  # If previous region has 1 point and there's a region before it
            prev_prev_start_idx, prev_prev_end_idx = consecutive_valid_regions[i - 2]
            prev_prev_region_indices = df.loc[prev_prev_start_idx:prev_prev_end_idx].index
            selected_prev_prev_indices = prev_prev_region_indices[-n_points_prev:] if len(prev_prev_region_indices) > n_points_prev else prev_prev_region_indices
            previous_region_std = df.loc[selected_prev_prev_indices, 'Strain'].std()
        else:
            previous_region_std = selected_prev_values.std()

        # Handle cases where the current region has only 1 point
        if curr_region_size == 1:
            current_region_mean = df.loc[start_idx, 'Strain']  # Use single point as mean
        else:
            # Compute mean for the first `n_points_curr` of the current region (after applying shift)
            curr_region_indices = df.loc[start_idx:end_idx].index
            selected_curr_indices = curr_region_indices[:n_points_curr] if len(curr_region_indices) > n_points_curr else curr_region_indices
            selected_curr_values = df.loc[selected_curr_indices, 'Strain']

            current_region_mean = selected_curr_values.mean()

        # Compute bounds
        lower_bound = previous_region_mean - std_multiplier * previous_region_std
        upper_bound = previous_region_mean + std_multiplier * previous_region_std

        # Only shift if the mean of the current region is outside the threshold
        if current_region_mean < lower_bound or current_region_mean > upper_bound:
            delta_shift = previous_region_mean - current_region_mean  # Compute shift amount
        else:
            delta_shift = 0

        # Apply the delta shift to the current region
        df.loc[start_idx:end_idx, 'Strain'] += delta_shift  

        # Update the total shift for future regions
        total_shift += delta_shift  

        start_time =df.loc[start_idx, 'Time']
        end_time = df.loc[end_idx, 'Time']

        # Print debug table
        table_data = [
            [start_time, end_time, previous_region_mean, previous_region_std, lower_bound, upper_bound, current_region_mean, delta_shift, total_shift]
        ]

        headers = ["Start time", "End time", "Previous Mean", "Previous Std", "Lower Bound", "Upper Bound", "Current Mean", "Delta Shift", "Total Shift"]
        df_output = pd.DataFrame(table_data, columns=headers)
        print(df_output.to_string(index=False))  # Print table without row index

    return df


'''
Make it so that the total_shift is only applied to the current region at the start of the loop. Then the mean of the current region and the standard deviation of the previous region should be calculated. Calculate if the mean is in
range and if it needs to be shifted. If it is outside the range of standard deviation calculate the shift as delta_shift = previous_region_mean - current_region_mean, if the mean is inside the range delta_shit should remain 0. 
Then apply the shift to the current region and add the delta shift to the total shift. The delta shift should be reset to 0 at the start of every loop, if the mean is inside the range delta_shit should remain 0. For the next loop the 
total shift should be applied to the current region and the process should be repeated.
'''

