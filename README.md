# My ML Project

This repository implements two text classification models:
1. **Feedforward Neural Network** (using TF-IDF features)
2. **LSTM Model** (using Word2Vec embeddings)

## Repository Structure

- **src/**: Contains the source code (configuration, data preprocessing, models, training routines, and utilities).
- **tests/**: Contains unit tests for various modules.
- **deployment/**: Contains deployment artifacts (e.g., Dockerfile).
- **.github/workflows/**: Contains CI/CD pipeline definitions for GitHub Actions.

## Getting Started

### 1. Installation

Create a virtual environment and install the requirements:

```bash
python -m venv venv
source venv/bin/activate      # Linux/MacOS
venv\Scripts\activate         # Windows
pip install --upgrade pip
pip install -r requirements.txt
