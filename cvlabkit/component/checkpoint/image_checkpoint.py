"""Image checkpoint component for saving visualization artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from einops import rearrange
from torchvision.utils import make_grid, save_image

from cvlabkit.component.base.checkpoint import Checkpoint


class ImageCheckpoint(Checkpoint):
    """Checkpoint component for saving image artifacts.

    Handles saving of:
    - Training visualizations
    - Generated samples
    - Comparison grids
    - Per-epoch snapshots

    Usage in Agent:
        self.image_checkpoint = self.create.checkpoint.image(
            save_dir="results/my_experiment"
        )

        # Save single image
        self.image_checkpoint.save_image(
            generated_image,
            filename="sample_epoch_10.png"
        )

        # Save comparison grid
        self.image_checkpoint.save_comparison(
            [original, weak_aug, generated, strong_aug],
            labels=["Original", "Weak", "Generated", "Strong"],
            filename="comparison_step_1000.png"
        )
    """

    def __init__(self, cfg):
        """Initialize ImageCheckpoint.

        Args:
            cfg: Config object with:
                - save_dir: Directory to save images (default: "results")
                - format: Image format (default: "png", options: "png", "jpg")
                - quality: JPEG quality 1-100 (default: 95)
                - normalize: Auto-normalize to [0,1] (default: True)
                - nrow: Images per row in grid (default: 4)
                - padding: Pixels between images in grid (default: 2)
                - verbose: Print save messages (default: True)
        """
        self.save_dir = Path(cfg.get("save_dir", "results"))
        self.format = cfg.get("format", "png")
        self.quality = cfg.get("quality", 95)
        self.normalize = cfg.get("normalize", True)
        self.nrow = cfg.get("nrow", 4)
        self.padding = cfg.get("padding", 2)
        self.verbose = cfg.get("verbose", True)

        # Create directory
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def save(self, state: Dict[str, Any], filename: str) -> Path:
        """Save image state to file (compatibility with Checkpoint interface).

        Args:
            state: Dictionary with:
                - "image": torch.Tensor or list of tensors
                - "metadata": Optional dict with additional info
            filename: Output filename

        Returns:
            Path: Full path to saved image
        """
        image = state.get("image")
        if image is None:
            raise ValueError("state must contain 'image' key")

        return self.save_image(image, filename)

    def load(self, filename: str) -> Dict[str, Any]:
        """Load image from file (compatibility with Checkpoint interface).

        Note: This loads image as PIL Image, not torch.Tensor.

        Args:
            filename: Image filename

        Returns:
            Dict containing loaded image and metadata
        """
        from PIL import Image

        load_path = self.save_dir / filename

        if not load_path.exists():
            raise FileNotFoundError(f"Image not found: {load_path}")

        image = Image.open(load_path)

        if self.verbose:
            print(f"[ImageCheckpoint] Loaded: {load_path}")

        return {
            "image": image,
            "path": load_path,
        }

    def save_image(
        self,
        image: Union[torch.Tensor, list[torch.Tensor]],
        filename: str,
        normalize: Optional[bool] = None,
    ) -> Path:
        """Save image tensor(s) to file.

        Args:
            image: Image tensor [C, H, W] or [B, C, H, W], or list of tensors
            filename: Output filename (auto-adds extension if missing)
            normalize: Override normalize setting

        Returns:
            Path: Full path to saved image
        """
        # Handle filename extension
        if not any(filename.endswith(ext) for ext in [".png", ".jpg", ".jpeg"]):
            filename = f"{filename}.{self.format}"

        save_path = self.save_dir / filename

        # Prepare image
        if isinstance(image, list):
            # List of images → grid
            image = torch.stack(image)

        # Ensure 4D: [B, C, H, W]
        if image.ndim == 3:
            image = image.unsqueeze(0)

        # Normalize if requested
        normalize = normalize if normalize is not None else self.normalize

        # Save
        save_image(
            image,
            save_path,
            normalize=normalize,
            nrow=self.nrow,
            padding=self.padding,
            quality=self.quality if self.format == "jpg" else None,
        )

        if self.verbose:
            print(f"[ImageCheckpoint] Saved: {save_path}")

        return save_path

    def save_grid(
        self,
        images: list[torch.Tensor],
        filename: str,
        nrow: Optional[int] = None,
        labels: Optional[list[str]] = None,
    ) -> Path:
        """Save multiple images as a grid.

        Args:
            images: List of image tensors
            filename: Output filename
            nrow: Images per row (overrides default)
            labels: Optional labels for each image (not yet supported)

        Returns:
            Path: Full path to saved image
        """
        # Stack images
        grid = torch.stack(images)

        # Make grid
        nrow = nrow if nrow is not None else self.nrow
        grid_image = make_grid(
            grid,
            nrow=nrow,
            padding=self.padding,
            normalize=self.normalize,
        )

        # Save
        if not any(filename.endswith(ext) for ext in [".png", ".jpg", ".jpeg"]):
            filename = f"{filename}.{self.format}"

        save_path = self.save_dir / filename
        save_image(
            grid_image,
            save_path,
            quality=self.quality if self.format == "jpg" else None,
        )

        if self.verbose:
            print(f"[ImageCheckpoint] Saved grid: {save_path}")

        return save_path

    def save_comparison(
        self,
        images: list[torch.Tensor],
        filename: str,
        labels: Optional[list[str]] = None,
        layout: str = "horizontal",
    ) -> Path:
        """Save comparison of multiple image sets.

        Args:
            images: List of image tensors [B, C, H, W] or list of lists
            filename: Output filename
            labels: Optional labels for each image type
            layout: "horizontal" or "grid"

        Returns:
            Path: Full path to saved image

        Example:
            # Compare original, weak, generated, strong for 16 samples
            images = [original, weak_aug, generated, strong_aug]  # Each [16, 3, 32, 32]
            save_comparison(images, "comparison.png", labels=["Orig", "Weak", "Gen", "Strong"])
        """
        # Stack all images: [N, B, C, H, W] where N=num_types, B=batch_size
        if isinstance(images[0], list):
            # List of lists → stack each
            images = [torch.stack(img_list) for img_list in images]

        # Stack into [N, B, C, H, W]
        comparison = torch.stack(images)  # [N, B, C, H, W]

        if layout == "horizontal":
            # Rearrange: each row = one sample, each column = one type
            # [N, B, C, H, W] → [B, N, C, H, W] → [B*N, C, H, W]
            comparison = rearrange(comparison, "n b c h w -> (b n) c h w")
            nrow = len(images)  # One row per sample
        elif layout == "grid":
            # Flatten: [N*B, C, H, W]
            comparison = rearrange(comparison, "n b c h w -> (n b) c h w")
            nrow = self.nrow
        else:
            raise ValueError(f"Unknown layout: {layout}")

        # Save
        return self.save_image(comparison, filename)

    def save_timesteps(
        self,
        images: list[torch.Tensor],
        filename: str,
        num_cols: Optional[int] = None,
    ) -> Path:
        """Save progression of images over timesteps.

        Args:
            images: List of image tensors representing timesteps
            filename: Output filename
            num_cols: Number of columns (timesteps per row)

        Returns:
            Path: Full path to saved image
        """
        num_cols = num_cols if num_cols is not None else len(images)
        return self.save_grid(images, filename, nrow=num_cols)

    def list_images(self, pattern: str = "*.png") -> list[Path]:
        """List all saved images.

        Args:
            pattern: Glob pattern

        Returns:
            list[Path]: List of image paths
        """
        return sorted(self.save_dir.glob(pattern))

    def clear_old_images(self, keep_pattern: str = "latest_*"):
        """Clear old images, keeping only specified patterns.

        Args:
            keep_pattern: Glob pattern for images to keep
        """
        all_images = self.list_images()
        keep_images = set(self.save_dir.glob(keep_pattern))

        for img in all_images:
            if img not in keep_images:
                img.unlink()
                if self.verbose:
                    print(f"[ImageCheckpoint] Deleted: {img}")
