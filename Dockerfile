# Use the official Miniconda3 base image
FROM continuumio/miniconda3:latest

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt first to take advantage of Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install OpenCV
RUN apt-get update && \
    apt-get install -y python3-opencv && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy the rest of the application code to the working directory
COPY . .

# Set the default command to start a shell (bash)
CMD ["/bin/bash"]
