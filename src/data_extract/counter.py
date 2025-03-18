from collections import Counter
import sys
from pathlib import Path

# Add the root project directory to the Python path
project_root = Path.cwd().parent.parent  # This will get the project root since the notebook is in 'notebooks/'
sys.path.append(str(project_root))

from configs.path_config import RAW_DATA_DIR, TXT_OUTPUT_DIR

def count_files_in_subfolders(base_dir):
    # Initialize a Counter to store file names and their occurrences
    file_counter = Counter()

    # List all subfolders in the main folder
    subfolders = [
        subfolder for subfolder in base_dir.iterdir()
        if subfolder.is_dir()  # Check if it's a directory
    ]

    # Iterate through each subfolder
    for subfolder in subfolders:
        print(f"Processing subfolder: {subfolder}")

        # List all files in the subfolder
        files_in_subfolder = [
            file.name for file in subfolder.iterdir()
            if file.is_file()  # Check if it's a file
        ]

        # Update the counter with the files from the current subfolder
        file_counter.update(files_in_subfolder)

    # Return the counter with file counts
    return file_counter

# Example usage:
base_dir = Path(RAW_DATA_DIR)  # Make sure RAW_DATA_DIR is converted to a Path object
file_counts = count_files_in_subfolders(base_dir)

# Save the results to a .txt file
output_file_path = Path(TXT_OUTPUT_DIR) / 'file_counts.txt'
with open(output_file_path, 'w') as file:
    for filename, count in file_counts.items():
        file.write(f"{filename},{count}\n")

print(f"File counts saved to {output_file_path}")
