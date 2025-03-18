import pandas as pd
import numpy as np
from datetime import datetime
from tkinter import Tk, filedialog
from pathlib import Path
import sys

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parents[2]))  # Go up two levels to reach the project root
from configs.path_config import BASE_DIR, EXTRACTED_DATA_DIR, RAW_DATA_DIR


def get_subfolder_list(start_time, end_time):
    """Retrieve a list of subfolders between the specified timestamps."""
    file_path = BASE_DIR / "configs" / "txt_configs" / "timestamps_04.txt"

    with file_path.open('r') as file:
        subfolder_list = [line.strip() for line in file.readlines() if line.strip().isdigit()]

    if not subfolder_list:
        print("No subfolders found.")
        return []

    # Identify valid indices
    start_index = subfolder_list.index(start_time) if start_time in subfolder_list else 0
    end_index = subfolder_list.index(end_time) if end_time in subfolder_list else len(subfolder_list) - 1

    return subfolder_list[start_index:end_index + 1]


def read_strain_value(file_path, pos, col):
    """Read strain value from the specified file, given the position and column index."""
    try:
        df = pd.read_csv(file_path, delimiter='\t', header=None)

        if col >= len(df.columns):
            print(f"Invalid column number in {file_path.name}.")
            return np.nan

        matching_row = df[df[0] == pos]
        return matching_row.iloc[0, col] if not matching_row.empty else np.nan

    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return np.nan


def extract_strains(start_time, end_time, loop, pos, col, folder):
    """
    Extracts strain data for the given time range, loop, and position and saves it as a CSV.
    """
    subfolders = get_subfolder_list(start_time, end_time)
    if not subfolders:
        return None

    df_strains = pd.DataFrame(columns=['Time_index', 'Time', 'Strain'])

    for subfolder in subfolders:
        file_path = RAW_DATA_DIR / subfolder / loop
        strain = read_strain_value(file_path, pos, col)
        time = pd.to_datetime(subfolder, format="%Y%m%d%H%M%S")
        df_strains = pd.concat([df_strains, pd.DataFrame([[subfolder, time, strain]], columns=df_strains.columns)])

    output_dir = EXTRACTED_DATA_DIR / folder
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv_path = output_dir / f"{loop}_{pos}_{start_time}-{end_time}.csv"

    df_strains.to_csv(output_csv_path, index=False)
    return df_strains


def get_file_input(prompt):
    """Opens a file dialog to select a .txt or .csv file."""
    print(prompt)
    Tk().withdraw()
    file_path = filedialog.askopenfilename(title="Select a file", filetypes=[("Text files", "*.txt"), ("CSV files", "*.csv")])
    return Path(file_path) if file_path else None


# Main Execution
file = get_file_input("Enter the name of the file (or select from explorer)")
print(f"\nFile: {file}" if file else "\nNo file selected.")

folder_choice = input("Type 'd' for default subfolder or 'c' for custom subfolder: ").strip().lower()
custom_folder = "" if folder_choice == 'd' else input("Enter the name of the custom subfolder: ").strip()

if file:
    df_args = pd.read_csv(file, header=None, sep=' ')
    df_args[3] = df_args[3].astype(str) + ' ' + df_args[4].astype(str) + ' ' + df_args[5].astype(str)
    df_args = df_args.drop(columns=[4, 5])
    df_args.columns = ['start_time', 'end_time', 'loop', 'pos', 'col']

    for _, row in df_args.iterrows():
        extract_strains(row['start_time'], row['end_time'], row['loop'], row['pos'], row['col'], custom_folder)