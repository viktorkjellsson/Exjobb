import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

def extract_strains(start_time, end_time, loop, pos, col):

    main_path = 'GÄB'

    # List all subfolders in the main folder
    subfolders = [
        name for name in os.listdir(main_path)
        if os.path.isdir(os.path.join(main_path, name)) and name.isdigit()  # Check if folder name is a number
    ]

    # Find the indexes of the start and end subfolders
    try:
        start_index = subfolders.index(start_time)
        end_index = subfolders.index(end_time)

        # Extract subfolders between the start and end subfolder (exclusive)
        subfolders_between = subfolders[start_index:end_index + 1]
        
        # Generate relative paths from the main folder
        subfolder_relative_paths = [os.path.relpath(os.path.join(main_path, folder), main_path) for folder in subfolders_between]

        print(len(subfolder_relative_paths), "subfolders found for the specified start and end times.")

    except ValueError:
        print("One or both of the specified subfolders were not found.")


    columns = ['Time_index', 'Time', 'Strain']
    df_strains = pd.DataFrame(columns=columns)

    for subfolder in subfolder_relative_paths:
        path = os.path.join(main_path, subfolder, loop)
        try:
            df = pd.read_csv(path, delimiter='\t', header=None) 

            # Ensure that column_number is within the valid range
            if col < len(df.columns):
                # Find the row based on a matching condition (adjust as necessary)
                matching_row = df[df[0] == pos]  # Assuming row_name is in column 0, adjust if needed

                # Check if the row exists
                if not matching_row.empty:
                    strain = matching_row.iloc[0, col]
                else:
                    print(f"No matching row found in {subfolder}.")
                    strain = np.nan
            else:
                print(f"Invalid column number in {subfolder}.")
                strain = np.nan
        except FileNotFoundError:
            print(f"File not found: {path}")
            strain = np.nan

        time = pd.to_datetime(str(subfolder), format="%Y%m%d%H%M%S") #convert the subfolder name to a datetime object
        row = pd.DataFrame([[subfolder, time, strain]], columns=columns) #create a new row with the data
        df_strains = pd.concat([df_strains, row], ignore_index=True) #add the new row to the dataframe

    # Define the relative path for saving the CSV file
    output_folder = "strain_csv"
    os.makedirs(output_folder, exist_ok=True)  # Create the folder if it doesn't exist
    output_csv = os.path.join(output_folder, f"{loop}_{pos}_{start_time}-{end_time}.csv")

    # Save the DataFrame to a CSV file
    df_strains.to_csv(output_csv, index=False)

    return df_strains

start_time = '20090605000000' #First timestamp
end_time = '20210611160000' #Last timestamp

# df_loops = pd.read_csv('file_counts.txt', delimiter='\t', header=None)
# # Filter the rows where column 0 starts with 'EI'
# df_filtered = df_loops[df_loops[0].str.startswith('EI')]
# loops = df_filtered[0].tolist()

col = 4  # Column number containing the strain data

# Input Section
loop = input("Enter the name of the loop (e.g., EI_N-B-Close_Comp.txt): ")
print(f"Loop: {loop}")
pos = input("Enter the position (e.g., N13, B, 12.64): ")
print(f"Position: {pos}")

extract_strains(start_time, end_time, loop, pos, col)