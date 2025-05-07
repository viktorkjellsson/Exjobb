"""
This script includes the model parameters, input and output features, and the directories for data, models, and logs.

Usage in other scripts: from configs.model_config import * 
"""

# =============================
# Model configurations
# =============================

# # Model features

INPUT_FEATURES = ['Strain', 
                'Temperature',
                'Rolling_mean', 
                'Rolling_std']
# INPUT_FEATURES = ['Strain', 
#                 'Temperature']
# INPUT_FEATURES = ['Strain']

OUTPUT_FEATURES = ['Strain']


# Model parameters

PARAMS = {
        'hidden_dim' : 512,
        'num_layers' : 2,
        'num_epochs' : 25,
        'learning_rate' : 0.001,
        'dropout' : 0.4,
        'sequence_length' : 64,
        'batch_size' : 16,
        'test_size' : 0.3
        }