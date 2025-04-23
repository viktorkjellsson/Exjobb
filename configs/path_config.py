from pathlib import Path


# =============================
# Model configurations
# =============================

# Model features
FEATURES = ['Strain', 
            'Temperature',
            'Rolling_mean', 
            'Rolling_std']
# FEATURES = ['Strain']


# Model parameters

PARAMS = {
        'hidden_dim' : 128,
        'num_layers' : 2,
        'num_epochs' : 5,
        'learning_rate' : 0.01,
        'dropout' : 0.3}

# =============================
# Loop configurations
# =============================

# GROUP1 - one of the subsets corresponting to loops with similarity in available data
GROUP1 = ['N-B_Far_Comp.txt_N13, B, 0.03.csv',
        'N-B_Mid1_Comp.txt_N5, B, 0.02.csv',
        'N-B_Mid2_Comp.txt_N10, B, 0.11.csv',
        'N-B-Close_Comp.txt_IX, B, 0.04.csv',
        'N-C_Close2_Comp.txt_IX, C, 0.09.csv',
        'N-C_Far_Comp.txt_N13, C, 0.03.csv',
        'N-C_Mid_Comp.txt_N10, C, 0.04.csv',
        'N-D_Far_Comp.txt_N13, D, 0.02.csv',
        'N-E_Close_Comp.txt_IX, E, 0.04.csv',
        'N-E_Far_Comp.txt_N13, E, 0.03.csv',
        'N-E_Mid1_Comp.txt_N5, E, 0.03.csv',
        'N-F_Close_Comp.txt_IX, F, 0.04.csv',
        'N-F_Mid_Comp.txt_N13, F, 0.02.csv',
        'N-Klaff_Comp.txt_V, B, 1.65.csv'
        ]

# GROUP1A - subset of GROUP1
GROUP1A = ['N-B_Far_Comp.txt_N13, B, 0.03.csv',
        'N-B_Mid1_Comp.txt_N5, B, 0.02.csv',
        'N-B_Mid2_Comp.txt_N10, B, 0.11.csv',
        'N-B-Close_Comp.txt_IX, B, 0.04.csv',
        'N-C_Far_Comp.txt_N13, C, 0.03.csv',
        'N-C_Mid_Comp.txt_N10, C, 0.04.csv',
        'N-E_Far_Comp.txt_N13, E, 0.03.csv',
        'N-F_Close_Comp.txt_IX, F, 0.04.csv',
        'N-F_Mid_Comp.txt_N13, F, 0.02.csv',
        ]

GROUP2S = ['S-B_Mid1_Comp.txt S14, B, 0.08',
        'S-B_Mid2_Comp.txt S10, B, 0.8',
        'S-B_Parking_Comp.txt S25, B, 0.07',
        'S-D_Parking_Comp.txt S25, D, 0.09',
        'S-F_Mid_Comp.txt S11, F, 0.02',
        'S-F_Parking_Comp.txt S25, F, 0.09',
        'S-F_Tunnel_Comp.txt S19, F, 0.06']

COMBINED = ['N-B-Close_Comp.txt_IX, B, 0.04.csv',
        'N-B_Far_Comp.txt_N13, B, 0.03.csv',
        'N-B_Mid1_Comp.txt_N5, B, 0.02.csv',
        'N-B_Mid2_Comp.txt_N10, B, 0.11.csv',
        'N-C_Far_Comp.txt_N13, C, 0.03.csv',
        'N-C_Mid_Comp.txt_N10, C, 0.04.csv',
        'N-E_Far_Comp.txt_N13, E, 0.03.csv',
        'N-F_Close_Comp.txt_IX, F, 0.04.csv',
        'N-F_Mid_Comp.txt_N13, F, 0.02.csv',
        'S-B_Close_Comp.txt_II,B,0.06_20090605000000-20210611160000.csv',
        'S-B_Tunnel_Comp.txt_S19,B,0.06_20090605000000-20210611160000.csv',
        'S-C_Close_Comp.txt_II,C,0.06_20090605000000-20210611160000.csv',
        'S-C_Far_D_Mid_Comp.txt_S6,C,0.07_20090605000000-20210611160000.csv',
        'S-D_Close_Comp.txt_II,D,0.06_20090605000000-20210611160000.csv',
        'S-E_Close_Comp.txt_I,E,0.04_20090605000000-20210611160000.csv',
        'S-E_Mid_Comp.txt_S12,E,0.09_20090605000000-20210611160000.csv',
        'S-E_Tunnel_Comp.txt_S19,E,0.06_20090605000000-20210611160000.csv',
        'S-F_Close_Comp.txt_II,F,0.06_20090605000000-20210611160000.csv']

# =============================
# Directories and Paths
# =============================

# Define base project directory using Path
BASE_DIR = Path(__file__).resolve().parent.parent
print(BASE_DIR)

# Data directories
RAW_DATA_DIR = BASE_DIR / 'data' / 'raw'                # Path to raw data
PROCESSED_DATA_DIR = BASE_DIR / 'data' / 'processed'    # Path to processed data
EXTRACTED_DATA_DIR = BASE_DIR / 'data' / 'extracted'    # Path to processed data
OUTPUT_DIR = BASE_DIR / 'output'                        # Path to output (e.g., model, results)

# Source directories
UTILS_DIR = BASE_DIR / 'src' / 'utils'
DATA_EXTRACT_DIR = BASE_DIR / 'src' / 'data_extract'
PROCESSING_DIR = BASE_DIR / 'src' / 'processing'

# Output directories
TXT_OUTPUT_DIR = BASE_DIR / 'output' / 'txt'

# Configuration directories
CONFIG_DIR = BASE_DIR / 'configs'

# Model directories
MODEL_DIR = BASE_DIR / 'models'
LOGS_DIR = MODEL_DIR / 'logs'
WEIGHTS_DIR = MODEL_DIR / 'weights'