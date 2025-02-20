import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

def extract_strains(start_time, end_time, loop, pos, col):

    main_path = '../0_GAB'

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
                matching_row = df[df[0] == pos]

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
    output_folder = "../timeseries_csv"
    os.makedirs(output_folder, exist_ok=True)  # Create the folder if it doesn't exist
    output_csv = os.path.join(output_folder, f"{loop}_{pos}_{start_time}-{end_time}.csv")

    # Save the DataFrame to a CSV file
    df_strains.to_csv(output_csv, index=False)

    return df_strains

default_start_time = '20090605000000' #First timestamp
default_end_time = '20210611160000' #Last timestampmp

    # Function to get the start date and time input with 'd' for default and 'c' for custom
def get_start_input(prompt):
    user_input = input(f"{prompt} (Press 'd' for default, 'c' for custom): ").strip().lower()
    
    if user_input == 'd':
        print(f"Using default date-time: {default_start_time}")
        return default_start_time
    elif user_input == 'c':
        # Custom date-time input from user
        custom_start_time = input("Enter date and time in format 'YYYYMMDDHH0000': ").strip()
        return custom_start_time
    else:
        print("Invalid input! Please press 'd' or 'c'.")
        return get_start_input(prompt)  # Recurse if invalid input
    
    # Function to get the start date and time input with 'd' for default and 'c' for custom
def get_end_input(prompt):
    user_input = input(f"{prompt} (Press 'd' for default, 'c' for custom): ").strip().lower()
    
    if user_input == 'd':
        print(f"Using default date-time: {default_end_time}")
        return default_end_time
    elif user_input == 'c':
        # Custom date-time input from user
        custom_end_time = input("Enter date and time in format 'YYYYMMDDHH0000': ").strip()
        return custom_end_time
    else:
        print("Invalid input! Please press 'd' or 'c'.")
        return get_start_input(prompt)  # Recurse if invalid input
    
    
# Input Section
print("\nStart Date-Time Input:")
start_time = get_start_input("Enter the start date and time")
print(f"\nStart date-time: {start_time}")

print("\nEnd Date-Time Input:")
end_time = get_end_input("Enter the end date and time")
print(f"End date-time: {end_time}")

loop = input("Enter the name of the loop (e.g., EI_N-B-Close_Comp.txt): ")
print(f"Loop: {loop}")

pos = input("Enter the position (e.g., N13, B, 12.64): ")
print(f"Position: {pos}")

col = input("Enter the column: ")
col = int(col)
print(f"Column: {col}")

extract_strains(start_time, end_time, loop, pos, col)

#kan lägga till så att den accepterar input av en textfil med alla loops och positioner som ska köras