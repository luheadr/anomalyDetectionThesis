import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image

# Load your model (update the path to the model files)
model = torch.load('/app/modelAndImages/model/model.bin')
model.eval()

# Function to run inference on a specified image
def run_inference(image_path):
    image = Image.open(image_path)
    # Preprocess image as required by your model
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),  # Adjust size according to your model
        transforms.ToTensor(),
    ])
    image_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        output = model(image_tensor)
        # Process output as necessary
        print("Inference output:", output)

if __name__ == "__main__":
    # Set the image path to the one you want to test
    test_image_path = '/app/modelAndImages/testImages/000.png'
    run_inference(test_image_path)  # Run inference on the specified image
