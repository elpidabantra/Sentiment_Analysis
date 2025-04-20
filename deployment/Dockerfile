# Use an official Python runtime as a parent image (upgraded Python version)
FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /app

# Copy requirements.txt first and install dependencies (leverages Docker caching)
COPY requirements.txt .

RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install gunicorn  # for web server deployment if needed

# Copy your application source code into the container
COPY . .

# Clearly set container's default listening port to 8000. Useful for deploying as a service or API.
EXPOSE 8000

# Provide flexibility for running different scripts or commands - Without a CMD (or ENTRYPOINT), the container doesn't know what to execute and will immediately exit.
# For training:
CMD ["python", "src/training.py"]
