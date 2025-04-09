# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set work directory
WORKDIR /app

# Copy requirements and install them first to leverage cache
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the rest of the repository into the container
COPY . .

# Expose a port if needed (e.g., for a web service)
# EXPOSE 5000

# Define default command (this runs the training script)
CMD ["python", "src/training.py"]
