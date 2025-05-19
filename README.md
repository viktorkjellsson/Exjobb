###########################################
    REPOSITORY TREE
###########################################

Exjobb/
├── configs/
│   ├── txt_configs
│   ├── model_config.py
│   ├── path_config.py
│   └── support_coords.py
├── data/
│   ├── extracted/
│   └── raw/
├── models/
│   ├── models/
│   └── lstm_model.py
├── notebooks/
│   ├── clustering.ipynb
│   ├── eda.ipynb
│   ├── plot_loop_at_set_time.ipynb
│   ├── test.ipynb
│   └── train.ipynb
├── output/
└── src/
    ├── clustering/
    |   ├── clustering_models.py
    |   ├── clustering_preprocessing.py
    |   ├── clustering_visualization.py
    |   └── sankey_diagram.py
    └── data_extract/
    |   ├── counter.py
    |   ├── extract_multiple_timeseries.py
    |   └── extract_timeseries.py
    └── processing/
    |   ├── dataset.py
    |   ├── interpolate_nan.py
    |   ├── nan_regions.py
    |   ├── preprocessing.py
    |   ├── remove_outliers.py
    |   ├── shift_region.py
    |   ├── shift_segments.py
    |   └── utils.py

###########################################
    DESCRIPTION OF THE FILES
###########################################

configs/
    txt_configs: Text files for model and data configurations. 
    model_config.py: LSTM model parameters.
    path_config.py: Configuration of relative paths in the repository.
    support_coords.py: Coordinates for the supports.

data/
    extracted/: The extracted and reformatted data. Contains time series and strain distributions.
    raw/: The raw data.

models/
    models/: Contains all the trained models their wheights, error thresholds and training logs.
    lstm_model.py

notebooks/
    clustering.ipynb: The workspace for the clustering model.
    eda.ipynb: Exploratory data analysis.
    plot_loop_at_set_time.ipynb: Extracts the strain distributions from the raw data and plots strain distributions for a specific time stamp.
    test.ipynb: The notebook for test/evaluation of the pretrained LSTM-models.
    train.ipynb: The notebook for training the LSTM-models.

output/

src/
    clustering/
        clustering_models.py: Contains functions that are needed for the different clustering models.
        clustering_preprocessing.py: Preprocessing pipeline for the clustering model.
        clustering_visualization.py: Contains plotting tools for visualizing the resulst from the clustering model.
        sankey_diagram.py: Produces a Sankey diagram that maps the cluster reassignment of datapoints over time.
    data_extract/
        counter.py
        extract_multiple_timeseries.py: Extract multiple time series at once from a .txt file configuration.
        extract_timeseries.py: Extracts one time series at a times.
    processing/
        dataset.py: Constructs the dataset and PyTorch DataLoaders for training and test.
        interpolate_nan.py: Interpolates NaN regions. 
        nan_regions.py: Identifies consequetive NaN regions. 
        preprocessing.py: The pre processing pipline. 
        remove_outliers.py: Identifies and removes outliers.
        shift_region.py
        shift_segments.py
        utils.py: Contains a set of tools for test/evaluation