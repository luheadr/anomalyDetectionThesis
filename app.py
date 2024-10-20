import cv2

# Initialize the camera
camera = cv2.VideoCapture(0)  # 0 refers to the default camera

# Check if the camera opened successfully
if not camera.isOpened():
    print("Error: Could not open camera.")
else:
    # Capture a single frame
    ret, frame = camera.read()

    if ret:
        # Save the image to the pictures folder
        cv2.imwrite("/home/pi/Pictures/captured_image.jpg", frame)
        print("Image saved successfully.")
    else:
        print("Error: Failed to capture image.")

# Release the camera
camera.release()
