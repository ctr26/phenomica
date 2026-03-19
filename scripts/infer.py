"""Standalone inference script for phenomica models.

Usage:
    uv run python scripts/infer.py \\
        --checkpoint path/to/best_model.pt \\
        --input_dir path/to/images \\
        --output features.pt \\
        --batch_size 256
"""

import argparse
import pathlib

import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from phenomica.data import get_transforms
from phenomica.models import build_model


def main():
    parser = argparse.ArgumentParser(description="Extract features with a trained phenomica model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory of images")
    parser.add_argument("--output", type=str, default="features.pt", help="Output file path")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=224)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_cfg = checkpoint["model_cfg"]
    model = build_model(model_cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    transform = get_transforms(args.image_size, is_train=False)
    dataset = datasets.ImageFolder(args.input_dir, transform=transform)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    all_features = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            features = model.extract_features(images)
            all_features.append(features.cpu())
            all_labels.append(labels)

    features = torch.cat(all_features, dim=0)
    labels = torch.cat(all_labels, dim=0)

    output_path = pathlib.Path(args.output)
    torch.save({"features": features, "labels": labels}, output_path)
    print(f"Saved {features.shape[0]} features ({features.shape[1]}-dim) to {output_path}")


if __name__ == "__main__":
    main()
