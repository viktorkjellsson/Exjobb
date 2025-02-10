import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def create_df(start_time, end_time, column, position):

    main_path = 'Exjobb\GÄB'

    columns = ['Time_index', 'Time', 'Strain']
    df = pd.DataFrame(columns=columns)

    # List all subfolders in the main folder (relative paths)
    subfolders = [name for name in os.listdir(main_path) if os.path.isdir(os.path.join(main_path, name))]
    # Sort the subfolders numerically or alphabetically if necessary
    subfolders = sorted(subfolders, key=lambda x: int(x))  # Assuming numeric sorting is required

    # Find the indexes of the start and end subfolders
    try:
        start_index = subfolders.index(start_time)
        end_index = subfolders.index(end_time)

        # Extract subfolders between the start and end subfolder (exclusive)
        subfolders_between = subfolders[start_index + 1:end_index]
        
        # Generate relative paths from the main folder
        subfolder_relative_paths = [os.path.relpath(os.path.join(main_path, folder), main_path) for folder in subfolders_between]

        print("Relative paths of subfolders between the two specified subfolders:", subfolder_relative_paths)

    except ValueError:
        print("One or both of the specified subfolders were not found.")

    # time_index = 20090330120000
    # time = pd.to_datetime(str(time_index), format="%Y%m%d%H%M%S")
    # strain = 0.1

    row = pd.DataFrame([[time_index, time, strain]], columns=columns)
    df = pd.concat([df, row], ignore_index=True)


start_time = '20100928120000'
end_time = '20101028200000'

