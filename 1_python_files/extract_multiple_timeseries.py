import pandas as pd
import os
import numpy as np
from datetime import datetime
from tkinter import Tk, filedialog

def extract_strains(start_time, end_time, loop, pos, col):
    """
    Function that extracts a timeseries and saves it as a .csv.

    Args:
        start_time: timestamp of the first value of the timeseries (YYYYMMDDHH0000)
        end_time: timestamp of the last value of the timeseries (YYYYMMDDHH0000)
        loop: The name of the fiber optic loop that is being measured (e.g. EI_N-B-Close_Comp.txt)
        pos: The position along the fiber optic cable (e.g. IX, B, 34.53)
        col: The column of the .txt file with the searched value

    Returns:
        df_strains: A dataframe that contains timeseries of the strains for the given timespan, loop, and position. Saves the dataframe as a .csv. 
    """


    main_path = '../0_GAB'

    # List all subfolders in the main folder
    subfolders = [
        name for name in os.listdir(main_path)
        if os.path.isdir(os.path.join(main_path, name)) and name.isdigit()  # Check if folder name is a number
    ]

    # Initialize variable to avoid UnboundLocalError
    subfolder_relative_paths = []

    # Find the indexes of the start and end subfolders
    try:
        start_index = subfolders.index(start_time)
        end_index = subfolders.index(end_time)

        # Extract subfolders between the start and end subfolder (inclusive)
        subfolders_between = subfolders[start_index:end_index + 1]
        
        # Generate relative paths from the main folder
        subfolder_relative_paths = [os.path.relpath(os.path.join(main_path, folder), main_path) for folder in subfolders_between]

        print(len(subfolder_relative_paths), "subfolders found for the specified start and end times.")

    except ValueError:
        print("One or both of the specified subfolders were not found.")

    # Exit early if no valid subfolders are found
    if not subfolder_relative_paths:
        print("No valid subfolders to process. Skipping extraction.")
        return None

    columns = ['Time_index', 'Time', 'Strain']
    df_strains = pd.DataFrame(columns=columns)

    for subfolder in subfolder_relative_paths:
        path = os.path.join(main_path, subfolder, loop)
        try:
            df = pd.read_csv(path, delimiter='\t', header=None) 

            # Ensure that col is within the valid range
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

        time = pd.to_datetime(str(subfolder), format="%Y%m%d%H%M%S")  # Convert folder name to datetime
        row = pd.DataFrame([[subfolder, time, strain]], columns=columns)  # Create new row with data
        df_strains = pd.concat([df_strains, row], ignore_index=True)  # Append new row to dataframe

    # Define the relative path for saving the CSV file
    output_folder = "../timeseries_csv"

    # Prompt the user for input to select between default or custom subfolder
    folder_choice = input("Type 'd' for default subfolder or 'c' for custom subfolder: ").strip().lower()

    # Check user input and determine the subfolder
    if folder_choice == 'd':
        subfolder = ""  # Default: no subfolder
    elif folder_choice == 'c':
        subfolder = input("Enter the name of the custom subfolder: ").strip()
        os.makedirs(os.path.join(output_folder, subfolder), exist_ok=True)  # Create subfolder if it doesn't exist
    else:
        print("Invalid choice. Defaulting to the main output folder.")
        subfolder = ""  # Default to no subfolder if invalid input

    # Create the full output path including the subfolder
    output_csv = os.path.join(output_folder, subfolder, f"{loop}_{pos}_{start_time}-{end_time}.csv")

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    # Save the DataFrame to a CSV file
    df_strains.to_csv(output_csv, index=False)

    return df_strains
    
def get_file_input(prompt):
    """Opens a file dialog to select a .txt or .csv file."""
    print(prompt)  # Print the prompt
    Tk().withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename(title="Select a file", 
                                           filetypes=[("Text files", "*.txt"), 
                                                     ("CSV files", "*.csv")])
    return file_path

print("\nFile input:")
file = get_file_input("Enter the name of the file (or select from explorer)")
print(f"\nFile: {file}" if file else "\nNo file selected.")

if file:
    df_args = pd.read_csv(file, header=None, sep=' ')
    print(df_args)
    df_args[3] = df_args[3].astype(str) + ' ' + df_args[4].astype(str) + ' ' + df_args[5].astype(str)
    df_args = df_args.drop([4, 5], axis=1)
    df_args[0] = df_args[0].astype(str)
    df_args[1] = df_args[1].astype(str)
    df_args.columns = ['start_time', 'end_time', 'loop', 'pos', 'col']

    for i in range(len(df_args)):
        start_time = df_args.iloc[i, 0]
        end_time = df_args.iloc[i, 1]
        loop = df_args.iloc[i, 2]
        pos = df_args.iloc[i, 3]
        col = df_args.iloc[i, 4]

        extract_strains(start_time, end_time, loop, pos, col)
