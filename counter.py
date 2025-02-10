from collections import Counter
import os
import time

start_time = time.time()

def count_files_in_subfolders(main_folder):
    # Initialize a Counter to store file names and their occurrences
    file_counter = Counter()

    # List all subfolders in the main folder
    subfolders = [
        name for name in os.listdir(main_folder)
        if os.path.isdir(os.path.join(main_folder, name))  # Check if folder
    ]

    # Iterate through each subfolder
    for subfolder in subfolders:
        subfolder_path = os.path.join(main_folder, subfolder)
        print(f"Processing subfolder: {subfolder_path}")

        # List all files in the subfolder
        files_in_subfolder = [
            name for name in os.listdir(subfolder_path)
            if os.path.isfile(os.path.join(subfolder_path, name))  # Check if file
        ]

        # Update the counter with the files from the current subfolder
        file_counter.update(files_in_subfolder)

    # Return the counter with file counts
    return file_counter

# Example usage:
main_folder = 'GÄB'
file_counts = count_files_in_subfolders(main_folder)

# Save the results to a .txt file
output_file_path = 'file_counts.txt'
with open(output_file_path, 'w') as file:
    for filename, count in file_counts.items():
        file.write(f"{filename}\t{count}\n")

print(f"File counts saved to {output_file_path}")

end_time = time.time()

# Calculate and print the execution time
execution_time = end_time - start_time
print(f"Execution time: {execution_time:.4f} seconds")
