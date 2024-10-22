import os
import subprocess

# Define the output directory and filename
output_dir = '/home/pi/anomalyDetectionThesis/modelAndImages/capturedImages'
output_file = os.path.join(output_dir, 'captured_image.jpg')

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Capture the image using libcamera-still
try:
    subprocess.run(['libcamera-still', '-o', output_file, '-t', '2000'], check=True)
    print(f'Image captured and saved to {output_file}')
except subprocess.CalledProcessError as e:
    print(f'Error capturing image: {e}')
