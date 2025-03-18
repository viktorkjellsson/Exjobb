from pathlib import Path

# =============================
# Directories and Paths
# =============================

# Define base project directory using Path
BASE_DIR = Path(__file__).resolve().parent
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