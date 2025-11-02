"""Periodic checkpoint component for saving artifacts at regular intervals."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

from cvlabkit.component.base.checkpoint import Checkpoint


class PeriodicCheckpoint(Checkpoint):
    """Checkpoint component that saves artifacts at regular intervals.

    Handles periodic saving based on:
    - Steps (every N training steps)
    - Epochs (every N epochs)
    - Custom conditions

    Can save any type of artifact (models, images, metrics, etc.)
    by using custom save callbacks.

    Usage in Agent:
        # Setup
        self.periodic_checkpoint = self.create.checkpoint.periodic(
            save_dir="periodic_saves",
            save_every_n_steps=1000,
            save_every_n_epochs=5,
            keep_last=10
        )

        # In train_step()
        self.periodic_checkpoint.on_step(
            current_step=self.current_step,
            save_fn=lambda path: self.save_checkpoint(path)
        )

        # In train_epoch() end
        self.periodic_checkpoint.on_epoch(
            current_epoch=self.current_epoch,
            save_fn=lambda path: self.save_visualization(path)
        )
    """

    def __init__(self, cfg):
        """Initialize PeriodicCheckpoint.

        Args:
            cfg: Config object with:
                - save_dir: Directory to save artifacts (default: "periodic_saves")
                - save_every_n_steps: Save every N steps (default: None, disabled)
                - save_every_n_epochs: Save every N epochs (default: None, disabled)
                - save_on_start: Save on first step/epoch (default: False)
                - keep_last: Keep only last N checkpoints (default: None, keep all)
                - name_format: Filename format (default: "{prefix}_{counter}.{ext}")
                - verbose: Print save messages (default: True)
        """
        self.save_dir = Path(cfg.get("save_dir", "periodic_saves"))
        self.save_every_n_steps = cfg.get("save_every_n_steps")
        self.save_every_n_epochs = cfg.get("save_every_n_epochs")
        self.save_on_start = cfg.get("save_on_start", False)
        self.keep_last = cfg.get("keep_last")
        self.name_format = cfg.get("name_format", "{prefix}_{counter}.{ext}")
        self.verbose = cfg.get("verbose", True)

        # Create directory
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Tracking
        self._last_saved_step = -1 if self.save_on_start else 0
        self._last_saved_epoch = -1 if self.save_on_start else 0
        self._saved_files = []

    def save(self, state: Dict[str, Any], filename: str) -> Path:
        """Save state to file (for Checkpoint interface compatibility).

        Args:
            state: State dictionary to save
            filename: Output filename

        Returns:
            Path: Full path to saved file
        """
        save_path = self.save_dir / filename

        # Use save_fn if provided in state
        if "save_fn" in state:
            state["save_fn"](save_path)
        else:
            # Default: save as torch checkpoint
            import torch
            torch.save(state, save_path)

        self._track_saved_file(save_path)

        if self.verbose:
            print(f"[PeriodicCheckpoint] Saved: {save_path}")

        return save_path

    def load(self, filename: str) -> Dict[str, Any]:
        """Load state from file (for Checkpoint interface compatibility).

        Args:
            filename: Filename to load

        Returns:
            Dict: Loaded state
        """
        import torch

        load_path = self.save_dir / filename

        if not load_path.exists():
            raise FileNotFoundError(f"File not found: {load_path}")

        state = torch.load(load_path)

        if self.verbose:
            print(f"[PeriodicCheckpoint] Loaded: {load_path}")

        return state

    def on_step(
        self,
        current_step: int,
        save_fn: Callable[[Path], None],
        prefix: str = "step",
        ext: str = "pt",
    ) -> Optional[Path]:
        """Check if should save on this step and execute save.

        Args:
            current_step: Current training step
            save_fn: Function to save artifact, receives save path
            prefix: Filename prefix (default: "step")
            ext: File extension (default: "pt")

        Returns:
            Optional[Path]: Path to saved file if saved, None otherwise
        """
        if self.save_every_n_steps is None:
            return None

        if current_step - self._last_saved_step >= self.save_every_n_steps:
            filename = self.name_format.format(
                prefix=prefix,
                counter=current_step,
                ext=ext
            )
            save_path = self.save_dir / filename

            save_fn(save_path)

            self._last_saved_step = current_step
            self._track_saved_file(save_path)

            if self.verbose:
                print(f"[PeriodicCheckpoint] Saved at step {current_step}: {save_path}")

            return save_path

        return None

    def on_epoch(
        self,
        current_epoch: int,
        save_fn: Callable[[Path], None],
        prefix: str = "epoch",
        ext: str = "pt",
    ) -> Optional[Path]:
        """Check if should save on this epoch and execute save.

        Args:
            current_epoch: Current training epoch
            save_fn: Function to save artifact, receives save path
            prefix: Filename prefix (default: "epoch")
            ext: File extension (default: "pt")

        Returns:
            Optional[Path]: Path to saved file if saved, None otherwise
        """
        if self.save_every_n_epochs is None:
            return None

        if current_epoch - self._last_saved_epoch >= self.save_every_n_epochs:
            filename = self.name_format.format(
                prefix=prefix,
                counter=current_epoch,
                ext=ext
            )
            save_path = self.save_dir / filename

            save_fn(save_path)

            self._last_saved_epoch = current_epoch
            self._track_saved_file(save_path)

            if self.verbose:
                print(f"[PeriodicCheckpoint] Saved at epoch {current_epoch}: {save_path}")

            return save_path

        return None

    def on_condition(
        self,
        condition: bool,
        save_fn: Callable[[Path], None],
        filename: str,
    ) -> Optional[Path]:
        """Save when custom condition is met.

        Args:
            condition: Whether to save
            save_fn: Function to save artifact, receives save path
            filename: Output filename

        Returns:
            Optional[Path]: Path to saved file if saved, None otherwise
        """
        if not condition:
            return None

        save_path = self.save_dir / filename
        save_fn(save_path)
        self._track_saved_file(save_path)

        if self.verbose:
            print(f"[PeriodicCheckpoint] Saved on condition: {save_path}")

        return save_path

    def _track_saved_file(self, path: Path):
        """Track saved file and clean up old ones if needed.

        Args:
            path: Path to saved file
        """
        self._saved_files.append(path)

        if self.keep_last is not None and len(self._saved_files) > self.keep_last:
            # Remove oldest file
            old_file = self._saved_files.pop(0)
            if old_file.exists():
                old_file.unlink()
                if self.verbose:
                    print(f"[PeriodicCheckpoint] Deleted old file: {old_file}")

    def list_saved_files(self) -> list[Path]:
        """List all saved files tracked by this checkpoint.

        Returns:
            list[Path]: List of saved file paths
        """
        return [f for f in self._saved_files if f.exists()]

    def clear_all(self):
        """Delete all tracked saved files."""
        for file in self._saved_files:
            if file.exists():
                file.unlink()
                if self.verbose:
                    print(f"[PeriodicCheckpoint] Deleted: {file}")

        self._saved_files = []

    def reset_counters(self):
        """Reset step/epoch counters (useful for resuming training)."""
        self._last_saved_step = 0
        self._last_saved_epoch = 0


class ImagePeriodicCheckpoint(PeriodicCheckpoint):
    """Specialized periodic checkpoint for images.

    Extends PeriodicCheckpoint with image-specific functionality.

    Usage in Agent:
        # Setup
        self.image_periodic = self.create.checkpoint.image_periodic(
            save_dir="periodic_images",
            save_every_n_epochs=5,
            format="png"
        )

        # At end of each epoch
        self.image_periodic.save_image_on_epoch(
            current_epoch=self.current_epoch,
            image=generated_samples,
            prefix="samples"
        )
    """

    def __init__(self, cfg):
        """Initialize ImagePeriodicCheckpoint.

        Args:
            cfg: Config object (includes PeriodicCheckpoint config plus):
                - format: Image format "png" or "jpg" (default: "png")
                - quality: JPEG quality 1-100 (default: 95)
                - normalize: Auto-normalize to [0,1] (default: True)
        """
        super().__init__(cfg)

        self.format = cfg.get("format", "png")
        self.quality = cfg.get("quality", 95)
        self.normalize = cfg.get("normalize", True)

    def save_image_on_step(
        self,
        current_step: int,
        image,
        prefix: str = "step",
    ) -> Optional[Path]:
        """Save image at regular step intervals.

        Args:
            current_step: Current training step
            image: Image tensor or list of tensors
            prefix: Filename prefix

        Returns:
            Optional[Path]: Path to saved image if saved
        """
        from torchvision.utils import save_image

        def save_fn(path):
            save_image(
                image,
                path,
                normalize=self.normalize,
                quality=self.quality if self.format == "jpg" else None,
            )

        return self.on_step(
            current_step=current_step,
            save_fn=save_fn,
            prefix=prefix,
            ext=self.format
        )

    def save_image_on_epoch(
        self,
        current_epoch: int,
        image,
        prefix: str = "epoch",
    ) -> Optional[Path]:
        """Save image at regular epoch intervals.

        Args:
            current_epoch: Current training epoch
            image: Image tensor or list of tensors
            prefix: Filename prefix

        Returns:
            Optional[Path]: Path to saved image if saved
        """
        from torchvision.utils import save_image

        def save_fn(path):
            save_image(
                image,
                path,
                normalize=self.normalize,
                quality=self.quality if self.format == "jpg" else None,
            )

        return self.on_epoch(
            current_epoch=current_epoch,
            save_fn=save_fn,
            prefix=prefix,
            ext=self.format
        )

    def save_comparison_on_epoch(
        self,
        current_epoch: int,
        images: list,
        prefix: str = "comparison",
        nrow: int = 4,
    ) -> Optional[Path]:
        """Save comparison grid at regular epoch intervals.

        Args:
            current_epoch: Current training epoch
            images: List of image tensors
            prefix: Filename prefix
            nrow: Images per row

        Returns:
            Optional[Path]: Path to saved image if saved
        """
        import torch
        from torchvision.utils import save_image

        def save_fn(path):
            grid = torch.stack(images) if isinstance(images[0], torch.Tensor) else images
            save_image(
                grid,
                path,
                nrow=nrow,
                normalize=self.normalize,
                quality=self.quality if self.format == "jpg" else None,
            )

        return self.on_epoch(
            current_epoch=current_epoch,
            save_fn=save_fn,
            prefix=prefix,
            ext=self.format
        )
