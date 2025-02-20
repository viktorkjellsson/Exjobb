0_GAB: The data containing subfolders corresponding to a timestamp YYYYMMDDHH0000. Each subfolder contains .txt files that holds the data for a specific fiber optic loop for that given timestamp.

1_python_files: contains all the python files (.py and .ipynb) 
    counter.py: Scans through all the subfolders in 0_GAB (timestamps) and counts number of times a certain loop occurs as a .txt file
    extract_timeseries.py: extract the data from the subdirectories and combines it into a timeseries
    data_preprocessing: The main data preprocessing pipeline 
    labbet.ipynb: just a bunch of crap that is being tested

timeseries_csv: folder with the extracted timeseries created by extract_timeseries.py 