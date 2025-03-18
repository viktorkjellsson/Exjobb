from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parents[2]))  # Go up two levels to reach the project root
from configs.path_config import BASE_DIR, EXTRACTED_DATA_DIR, RAW_DATA_DIR


def extract_strains(start_time, end_time, loop, pos, col):
    base_dir = BASE_DIR  # Assuming BASE_DIR is a Path object

    # Define the file path for timestamps
    file_path = base_dir / "configs" / "txt_configs" / "timestamps.txt"

    # Read the folder names from the text file
    with file_path.open("r") as file:
        subfolders = [line.strip() for line in file.readlines() if line.strip().isdigit()]

    try:
        # Find the indexes of the start and end subfolders
        start_index = subfolders.index(start_time)
        end_index = subfolders.index(end_time)

        # Extract subfolders between start and end timestamps (inclusive)
        subfolders_between = subfolders[start_index : end_index + 1]

        # Generate relative paths
        subfolder_relative_paths = [Path(folder) for folder in subfolders_between]

        print(len(subfolder_relative_paths), "subfolders found for the specified start and end times.")

    except ValueError:
        print("One or both of the specified subfolders were not found.")
        return None  # Return None to indicate failure

    # Create an empty DataFrame
    columns = ["Time_index", "Time", "Strain"]
    df_strains = pd.DataFrame(columns=columns)

    for subfolder in subfolder_relative_paths:
        # Construct the file path using pathlib
        path = RAW_DATA_DIR / subfolder / loop

        try:
            df = pd.read_csv(path, delimiter="\t", header=None)

            # Ensure that column_number is within the valid range
            if col < len(df.columns):
                # Find the row based on a matching condition
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

        # Convert the subfolder name to a datetime object
        time = pd.to_datetime(str(subfolder), format="%Y%m%d%H%M%S")

        # Create a new row with the extracted data
        row = pd.DataFrame([[subfolder, time, strain]], columns=columns)

        # Append to DataFrame
        df_strains = pd.concat([df_strains, row], ignore_index=True)

    # Define the output folder and create it if necessary
    output_dir = EXTRACTED_DATA_DIR
    # output_folder.mkdir(parents=True, exist_ok=True)

    # Define the output CSV path
    output_folder = output_dir / f"{loop}_{pos}_{start_time}-{end_time}.csv"

    # Save the DataFrame to a CSV file
    df_strains.to_csv(output_folder, index=False)

    return df_strains

# Default timestamps
default_start_time = "20090605000000"  # First timestamp
default_end_time = "20210611160000"  # Last timestamp

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
print(type(start_time))

print("\nEnd Date-Time Input:")
end_time = get_end_input("Enter the end date and time")
print(f"End date-time: {end_time}")
print(type(end_time))

loop = input("Enter the name of the loop (e.g., EI_N-B-Close_Comp.txt): ")
print(f"Loop: {loop}")
print(type(loop))

pos = input("Enter the position (e.g., N13, B, 12.64): ")
print(f"Position: {pos}")
print(type(pos))

col = input("Enter the column: ")
col = int(col)
print(f"Column: {col}")
print(type(col))

extract_strains(start_time, end_time, loop, pos, col)