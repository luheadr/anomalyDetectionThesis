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

def load_image(image_path: str, image_size: tuple = (256, 256)) -> np.ndarray:
    """Load and preprocess image."""
    logger.info(f"Attempting to load image from: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"Failed to load image from {image_path}")
        raise ValueError(f"Could not load image from {image_path}")
    logger.info(f"Successfully loaded image with shape: {image.shape}")
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, image_size, interpolation=cv2.INTER_AREA)
    return image

class PadimModel(nn.Module):
    """PaDiM model matching the checkpoint structure."""
    
    def __init__(self, input_size=(256, 256)):
        super().__init__()
        self.input_size = input_size
        
        # Load backbone without pretrained weights
        self.feature_extractor = models.resnet18(weights=None)
        self.feature_extractor.eval()
        
        # We'll use multiple layers for better feature representation
        self.layers = ['layer1', 'layer2', 'layer3']
        self.feature_maps = []
        
        # Register hooks to get intermediate layer outputs
        def hook_fn(module, input, output):
            self.feature_maps.append(output)
            
        for name, module in self.feature_extractor.named_children():
            if name in self.layers:
                module.register_forward_hook(hook_fn)
        
        # Initialize additional components from checkpoint with correct dimensions
        self.gaussian = nn.Module()
        self.gaussian.mean = nn.Parameter(torch.zeros(100, 4096))
        self.gaussian.inv_covariance = nn.Parameter(torch.zeros(4096, 100, 100))
        
        self.normalization_metrics = nn.Module()
        self.normalization_metrics.min = nn.Parameter(torch.tensor(0.0))
        self.normalization_metrics.max = nn.Parameter(torch.tensor(1.0))
        
        self.image_threshold = nn.Module()
        self.image_threshold.value = nn.Parameter(torch.tensor(0.5))
        
        self.pixel_threshold = nn.Module()
        self.pixel_threshold.value = nn.Parameter(torch.tensor(0.5))
        
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        self.feature_maps = []  # Clear previous feature maps
        _ = self.feature_extractor(x)  # This will trigger the hooks
        
        # Process and combine feature maps
        resized_maps = []
        for feat_map in self.feature_maps:
            resized = nn.functional.interpolate(
                feat_map,
                size=(64, 64),  # Fixed size for all feature maps
                mode='bilinear',
                align_corners=False
            )
            resized_maps.append(resized)
        
        # Concatenate all feature maps
        features = torch.cat(resized_maps, dim=1)  # [B, C, H, W]
        B, C, H, W = features.shape
        logger.info(f"Feature shape before reshape: {features.shape}")
        
        # Project features to match the expected dimension
        projection = nn.Linear(C, 100).to(x.device)
        
        # Reshape features for processing
        features = features.permute(0, 2, 3, 1)  # [B, H, W, C]
        features = features.reshape(-1, C)  # [BHW, C]
        features = projection(features)  # [BHW, 100]
        
        logger.info(f"Feature shape after projection: {features.shape}")
        
        # Get pre-computed statistics from the model
        mean = self.gaussian.mean  # [100, 4096]
        inv_cov = self.gaussian.inv_covariance  # [4096, 100, 100]
        
        # Compute distances for each location
        distances = []
        for i in range(4096):
            loc_mean = mean[:, i]  # [100]
            loc_cov = inv_cov[i]  # [100, 100]
            
            # Compute difference from mean
            diff = features - loc_mean.unsqueeze(0)  # [BHW, 100]
            
            # Compute Mahalanobis distance
            dist = torch.sum((diff @ loc_cov) * diff, dim=1)  # [BHW]
            distances.append(dist)
        
        # Stack distances
        distances = torch.stack(distances, dim=1)  # [BHW, 4096]
        
        # Take minimum distance for each location
        min_distances = torch.min(distances, dim=1)[0]  # [BHW]
        
        # Reshape back to spatial dimensions
        anomaly_map = min_distances.reshape(B, H, W)  # [B, H, W]
        
        # Add channel dimension
        anomaly_map = anomaly_map.unsqueeze(1)  # [B, 1, H, W]
        
        # Normalize to [0, 1]
        anomaly_map = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-8)
        
        # Upscale to input size
        anomaly_map = nn.functional.interpolate(
            anomaly_map,
            size=self.input_size,
            mode='bilinear',
            align_corners=False
        )
        
        return {"anomaly_map": anomaly_map}

def infer(image_path: str, weights_path: str, output_path: str) -> None:
    """Run inference on a single image."""
    # Disable gradient computation for inference
    torch.set_grad_enabled(False)
    
    # Model parameters
    image_size = (256, 256)
    
    # Initialize model
    model = PadimModel(input_size=image_size)
    
    # Load weights and print debug info
    weights_path = Path(weights_path)
    logger.info(f"Looking for weights at: {weights_path.absolute()}")
    
    if weights_path.exists():
        logger.info(f"Found weights file of size: {weights_path.stat().st_size / 1024 / 1024:.2f} MB")
        try:
            logger.info("Attempting to load checkpoint...")
            checkpoint = torch.load(weights_path, map_location="cpu")
            
            logger.info("Successfully loaded checkpoint")
            logger.info(f"Checkpoint type: {type(checkpoint)}")
            logger.info("Checkpoint keys:")
            logger.info(checkpoint.keys())
            
            # Process state dict
            if "state_dict" in checkpoint:
                logger.info("Processing state dict from checkpoint...")
                state_dict = {k.replace("model.", ""): v for k, v in checkpoint["state_dict"].items()}
            else:
                logger.info("Using checkpoint directly as state dict")
                state_dict = checkpoint
                
            # Print shapes of tensors in state dict
            logger.info("State dict tensor shapes:")
            for k, v in state_dict.items():
                logger.info(f"{k}: {v.shape}")
                
            model.load_state_dict(state_dict, strict=False)
            logger.info("Successfully loaded weights into model")
        except Exception as e:
            logger.error(f"Failed to load weights: {str(e)}")
            logger.error("Stack trace:", exc_info=True)
    else:
        logger.warning(f"No weights file found at {weights_path}")
    
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
    if not output_path.exists():
        output_path.mkdir(parents=True, mode=0o777, exist_ok=True)
        output_path.parent.chmod(0o777)
    
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
    INPUT_PATH = "/app/input/000.png"
    WEIGHTS_PATH = "/app/weights/model.ckpt"
    OUTPUT_DIR = "/app/output/000"  # Create a subdirectory for this image
    
    infer(INPUT_PATH, WEIGHTS_PATH, OUTPUT_DIR)