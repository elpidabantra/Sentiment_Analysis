# My ML Project

This repository implements two text classification models:

- Feedforward Neural Network (using TF-IDF features)
- LSTM Model (using Word2Vec embeddings)

## Repository Structure

- **src/**: Contains the source code (configuration, data preprocessing, models, training routines, and utilities).
- **tests/**: Contains unit tests for various modules.
- **deployment/**: Contains deployment artifacts (e.g., Dockerfile).
- **.github/workflows/**: Contains CI/CD pipeline definitions for GitHub Actions.

## Getting Started

### 1. Installation

Create a virtual environment and install the required dependencies:

python -m venv venv
source venv/bin/activate      # For Linux/MacOS
venv\Scripts\activate         # For Windows
pip install --upgrade pip
pip install -r requirements.txt

### 2. Running the Training Script (training.py)
Ensure you have activated your virtual environment (venv). Then, execute the main training script:

bash
Copy
python src/training.py

### Notes
If you use real data (defined in config.CSV_DATA_PATH), ensure the CSV file exists at the specified path.
If the CSV file is not found, dummy text data will be used for demonstration purposes.
