"""Simple inference script without anomalib dependencies."""

import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from pathlib import Path
import cv2
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("inference")

class PadimModel(nn.Module):
    """Simplified Padim model."""
    
    def __init__(self, input_size=(256, 256)):
        super().__init__()
        self.input_size = input_size
        
        # Load backbone
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone.eval()
        
        # We'll use multiple layers for better feature representation
        self.layers = ['layer1', 'layer2', 'layer3']
        self.feature_maps = []
        
        # Register hooks to get intermediate layer outputs
        def hook_fn(module, input, output):
            self.feature_maps.append(output)
            
        for name, module in self.backbone.named_children():
            if name in self.layers:
                module.register_forward_hook(hook_fn)
        
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        self.feature_maps = []  # Clear previous feature maps
        _ = self.backbone(x)  # This will trigger the hooks
        
        # Process and combine feature maps
        resized_maps = []
        for feat_map in self.feature_maps:
            # Resize each feature map to input size
            resized = nn.functional.interpolate(
                feat_map,
                size=self.input_size,
                mode='bilinear',
                align_corners=False
            )
            resized_maps.append(resized)
        
        # Concatenate all feature maps
        combined = torch.cat(resized_maps, dim=1)
        
        # Create anomaly map from combined features
        anomaly_map = torch.mean(combined, dim=1, keepdim=True)
        
        # Normalize the anomaly map
        min_val = torch.min(anomaly_map)
        max_val = torch.max(anomaly_map)
        anomaly_map = (anomaly_map - min_val) / (max_val - min_val + 1e-8)
        
        return {"anomaly_map": anomaly_map}

def load_image(image_path: str, image_size: tuple = (256, 256)) -> np.ndarray:
    """Load and preprocess image."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, image_size, interpolation=cv2.INTER_AREA)
    return image

def infer(image_path: str, weights_path: str, output_path: str) -> None:
    """Run inference on a single image."""
    # Disable gradient computation for inference
    torch.set_grad_enabled(False)
    
    # Model parameters
    image_size = (256, 256)
    
    # Initialize model
    model = PadimModel(input_size=image_size)
    
    # Load weights if they exist
    if Path(weights_path).exists():
        try:
            checkpoint = torch.load(weights_path, map_location="cpu")
            if "state_dict" in checkpoint:
                state_dict = {k.replace("model.", ""): v for k, v in checkpoint["state_dict"].items()}
            else:
                state_dict = checkpoint
            model.load_state_dict(state_dict)
            logger.info("Loaded weights from checkpoint")
        except Exception as e:
            logger.warning(f"Failed to load weights: {e}")
    
    model.eval()
    
    # Load and preprocess image
    image = load_image(image_path, image_size)
    
    # Convert to tensor
    image_tensor = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    
    # Run inference
    with torch.no_grad():
        predictions = model(image_tensor)
    
    # Save results
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save anomaly map
    anomaly_map = predictions["anomaly_map"].squeeze().cpu().numpy()
    
    # Apply color map for better visualization
    anomaly_map_colored = cv2.applyColorMap(
        (anomaly_map * 255).astype(np.uint8),
        cv2.COLORMAP_JET
    )
    
    # Save original image
    cv2.imwrite(str(output_path / "input_image.png"),
                cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    
    # Save anomaly map
    cv2.imwrite(str(output_path / "anomaly_map.png"), 
                anomaly_map_colored)
    
    # Create and save overlay
    overlay = cv2.addWeighted(
        cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
        0.7,  # Original image weight
        anomaly_map_colored,
        0.3,  # Anomaly map weight
        0
    )
    cv2.imwrite(str(output_path / "overlay.png"), overlay)
    
    logger.info(f"Results saved to {output_path}")

if __name__ == "__main__":
    # These paths will be mounted from the host system
    IMAGE_PATH = "/app/input/000.png"
    WEIGHTS_PATH = "/app/weights/model.ckpt"
    OUTPUT_PATH = "/app/output"
    
    infer(IMAGE_PATH, WEIGHTS_PATH, OUTPUT_PATH)