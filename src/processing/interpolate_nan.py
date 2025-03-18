import pandas as pd


def interpolate(df, nan_regions_sorted, interpolate_threshold): 
    nan_regions_sorted_to_interpolate = [
        region for region in nan_regions_sorted if region[4] <= interpolate_threshold
    ]

    df_filled = df.copy()

    for data in nan_regions_sorted_to_interpolate:
        print(f"Region to fill: \n   Length: {data[4]} steps\n   Start Time: {data[2]}\n   End Time: {data[3]}")
    
        start_index = data[0] - 1  # Find the last real value to interpolate from
        print(start_index)
        end_index = data[1] + 1    # Find the next real value to interpolate to

        # Check if both start_index and end_index are in df.index
        if start_index in df_filled.index and end_index in df_filled.index:
            print(f'Both indices ({start_index}, {end_index}) are valid -> interpolate {data[4]} steps')
            start_value = df_filled.loc[start_index, 'Strain']
            end_value = df_filled.loc[end_index, 'Strain']
            # Interpolate between the values
            for target_index in range(data[0], data[1]+1):
                print(f'target: {target_index}')
                interpolated_value = start_value + (end_value - start_value) * (target_index - start_index) / (end_index - start_index)
                df_filled.loc[target_index, 'Strain'] = interpolated_value  # Assign the interpolated value to df
                print(f"Interpolated value at index {target_index}: {interpolated_value}")

        # Handle case where only start_index is valid (extrapolation)
        elif start_index in df_filled.index and end_index not in df_filled.index:
            print(f'End index {end_index} is not a valid index -> extrapolating {data[4]} steps')
            start_index = start_index-1
            end_index = start_index
            start_value = df_filled.loc[start_index, 'Strain']
            end_value = df_filled.loc[end_index, 'Strain']
            # Extrapolate from the start_value towards the invalid end_index
            for target_index in range(data[0], data[1]):
                extrapolated_value = start_value + (end_value - start_value) * (target_index - start_index) / (end_index - start_index)
                df_filled.loc[target_index, 'Strain'] = extrapolated_value  # Assign the extrapolated value to df
                print(f"Extrapolated value at index {target_index}: {extrapolated_value}")

        # Handle case where only end_index is valid (extrapolation)
        elif start_index not in df_filled.index and end_index in df_filled.index:
            print(f'Start index {start_index} is not a valid index -> extrapolating {data[4]} steps')
            start_index = end_index
            end_index = end_index + 1
            start_value = df_filled.loc[start_index, 'Strain']
            end_value = df_filled.loc[end_index, 'Strain']
            # Extrapolate from the end_value towards the invalid start_index
            for target_index in range(data[1], data[0] - 1, -1):
                extrapolated_value = end_value - (end_value - start_value) * (end_index - target_index) / (end_index - start_index)
                df_filled.loc[target_index, 'Strain'] = extrapolated_value  # Assign the extrapolated value to df
                print(f"Extrapolated value at index {target_index}: {extrapolated_value}")

        else:
            print(f"Neither start_index {start_index} nor end_index {end_index} are valid. Skipping region.")
            continue
        
        print(f'Start value: {start_value}')
        print(f'End value: {end_value} \n')

    print(f'Number of missing values in df: {df['Strain'].isna().sum()}\nNumber of missing values in df_filled:{df_filled['Strain'].isna().sum()}')
    print(f'Number of missing values filled: {df['Strain'].isna().sum() - df_filled['Strain'].isna().sum()}')
    return df_filled