"""Inference Entrypoint script."""

import argparse
import torch
from pathlib import Path
from torch.utils.data import DataLoader

from anomalib.data import PredictDataset
from anomalib.engine import Engine
from anomalib.models.padim.lightning_model import PadimLightning


def get_parser() -> argparse.ArgumentParser:
    """Get parser.

    Returns:
        ArgumentParser: The parser object.
    """
    parser = argparse.ArgumentParser(description="Inference on Anomaly models in Lightning format.")
    
    # Add model arguments directly
    parser.add_argument("--model.name", type=str, default="padim")
    parser.add_argument("--model.backbone", type=str, default="resnet18")
    parser.add_argument("--model.layers", nargs="+", default=["layer1", "layer2", "layer3"])
    parser.add_argument("--model.pre_trained", type=bool, default=True)
    parser.add_argument("--model.normalization_method", type=str, default="min_max")
    
    # Add other arguments
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to model weights")
    parser.add_argument("--data.path", type=str, required=True, help="Path to input image")
    parser.add_argument("--data.image_size", nargs=2, type=int, default=[256, 256], help="Image size [height, width]")
    parser.add_argument("--output", type=str, required=False, help="Path to save the output image(s).")
    parser.add_argument(
        "--show",
        action="store_true",
        required=False,
        help="Show the visualized predictions on the screen.",
    )

    return parser


def infer(args) -> None:
    """Run inference."""
    # Disable gradient computation
    torch.set_grad_enabled(False)
    
    # Initialize engine
    engine = Engine(
        default_root_dir=args.output,
        devices=1,
    )

    # Create model directly
    model = PadimLightning.load_from_checkpoint(
        args.ckpt_path,
        backbone=args.model_backbone,
        layers=args.model_layers,
        pre_trained=args.model_pre_trained,
        normalization_method=args.model_normalization_method
    )
    model.eval()

    # Create dataset
    dataset = PredictDataset(
        path=args.data_path,
        image_size=args.data_image_size,
    )
    dataloader = DataLoader(dataset)

    # Run prediction
    engine.predict(model=model, dataloaders=[dataloader])


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    
    # Convert dot notation to underscores for easier access
    args.model_name = getattr(args, "model.name")
    args.model_backbone = getattr(args, "model.backbone")
    args.model_layers = getattr(args, "model.layers")
    args.model_pre_trained = getattr(args, "model.pre_trained")
    args.model_normalization_method = getattr(args, "model.normalization_method")
    args.data_path = getattr(args, "data.path")
    args.data_image_size = getattr(args, "data.image_size")
    
    infer(args)