# Use a base image for Anaconda
FROM continuumio/miniconda3

# Update and install any required system packages
RUN apt-get update && apt-get install -y build-essential libssl-dev libffi-dev python3-dev

# Set working directory
WORKDIR /workspace

# Copy requirements.txt to the container
COPY requirements.txt .

# Create and activate a new conda environment for Anomalib with Python 3.10
RUN conda create -n anomalib_env python=3.10 -y
RUN echo "conda activate anomalib_env" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

# Install dependencies from requirements.txt in the conda environment
RUN conda activate anomalib_env && pip install -r requirements.txt

# Expose any required ports
EXPOSE 8888

# Default command to run when the container starts
CMD ["bash"]
