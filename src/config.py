"""
Configuration file for the ML project.
Define default hyperparameters, file paths, and other constants here.
"""

# Training parameters
BATCH_SIZE_FF = 25
BATCH_SIZE_LSTM = 64
EPOCHS_FF = 5
EPOCHS_LSTM = 5
LEARNING_RATE_FF = 0.0005
LEARNING_RATE_LSTM = 0.001

# TF-IDF settings
NUM_FEATURES = 5000

# Dummy data parameters
NUM_SAMPLES = 500
NUM_CLASSES = 3

# Paths (if needed)
RESULTS_EXCEL_PATH = "results.xlsx"

# Path to your input CSV data
CSV_DATA_PATH = "data/text_data.csv" 


