"""Model checkpoint component for saving/loading PyTorch models."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch

from cvlabkit.component.base.checkpoint import Checkpoint


class ModelCheckpoint(Checkpoint):
    """Checkpoint component for PyTorch models and training state.

    Handles saving/loading of:
    - Model state dict
    - Optimizer state dict
    - Training metadata (epoch, step, etc.)
    - Custom state (metrics, hyperparams, etc.)

    Usage in Agent:
        self.checkpoint = self.create.checkpoint.model(
            save_dir="checkpoints/my_experiment"
        )

        # Save
        self.checkpoint.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": self.current_epoch,
            "step": self.current_step,
        }, filename="checkpoint_epoch_10.pt")

        # Load
        state = self.checkpoint.load("checkpoint_epoch_10.pt")
        self.model.load_state_dict(state["model"])
    """

    def __init__(self, cfg):
        """Initialize ModelCheckpoint.

        Args:
            cfg: Config object with:
                - save_dir: Directory to save checkpoints (default: "checkpoints")
                - auto_save_latest: Auto-save as "latest.pt" (default: True)
                - save_optimizer: Include optimizer state (default: True)
                - verbose: Print save/load messages (default: True)
        """
        self.save_dir = Path(cfg.get("save_dir", "checkpoints"))
        self.auto_save_latest = cfg.get("auto_save_latest", True)
        self.save_optimizer = cfg.get("save_optimizer", True)
        self.verbose = cfg.get("verbose", True)

        # Create directory
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def save(self, state: Dict[str, Any], filename: str = "checkpoint.pt") -> Path:
        """Save checkpoint to file.

        Args:
            state: Dictionary containing state to save. Common keys:
                - "model": model.state_dict()
                - "optimizer": optimizer.state_dict()
                - "epoch": current epoch
                - "step": current step
                - "metrics": dict of metrics
                - ... (any custom state)
            filename: Checkpoint filename (relative to save_dir)

        Returns:
            Path: Full path to saved checkpoint
        """
        save_path = self.save_dir / filename

        # Save checkpoint
        torch.save(state, save_path)

        if self.verbose:
            print(f"[ModelCheckpoint] Saved: {save_path}")

        # Auto-save as latest
        if self.auto_save_latest:
            latest_path = self.save_dir / "latest.pt"
            torch.save(state, latest_path)
            if self.verbose:
                print(f"[ModelCheckpoint] Updated: {latest_path}")

        return save_path

    def load(self, filename: str, map_location: Optional[str] = None) -> Dict[str, Any]:
        """Load checkpoint from file.

        Args:
            filename: Checkpoint filename (relative to save_dir) or "latest"
            map_location: Device to load checkpoint (e.g., "cpu", "cuda:0")

        Returns:
            Dict[str, Any]: Loaded state dictionary

        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
        """
        # Handle "latest" shortcut
        if filename == "latest":
            filename = "latest.pt"

        load_path = self.save_dir / filename

        if not load_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {load_path}")

        # Load checkpoint
        state = torch.load(load_path, map_location=map_location)

        if self.verbose:
            print(f"[ModelCheckpoint] Loaded: {load_path}")
            if "epoch" in state:
                print(f"  Epoch: {state['epoch']}")
            if "step" in state:
                print(f"  Step: {state['step']}")

        return state

    def save_for_inference(
        self,
        model_state: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        filename: str = "model_inference.pt"
    ) -> Path:
        """Save lightweight checkpoint for inference (no optimizer).

        Args:
            model_state: Model state dict (from model.state_dict())
            metadata: Optional metadata (config, metrics, etc.)
            filename: Output filename

        Returns:
            Path: Full path to saved checkpoint
        """
        state = {
            "model_state_dict": model_state,
            "metadata": metadata or {},
        }

        save_path = self.save_dir / filename
        torch.save(state, save_path)

        if self.verbose:
            print(f"[ModelCheckpoint] Saved inference model: {save_path}")

        return save_path

    def list_checkpoints(self, pattern: str = "*.pt") -> list[Path]:
        """List all checkpoints in save_dir.

        Args:
            pattern: Glob pattern (default: "*.pt")

        Returns:
            list[Path]: List of checkpoint paths
        """
        return sorted(self.save_dir.glob(pattern))

    def delete_old_checkpoints(self, keep_last: int = 5, pattern: str = "checkpoint_*.pt"):
        """Delete old checkpoints, keeping only the most recent.

        Args:
            keep_last: Number of recent checkpoints to keep
            pattern: Glob pattern for checkpoints to manage
        """
        checkpoints = sorted(self.save_dir.glob(pattern))

        if len(checkpoints) > keep_last:
            to_delete = checkpoints[:-keep_last]
            for ckpt in to_delete:
                ckpt.unlink()
                if self.verbose:
                    print(f"[ModelCheckpoint] Deleted old checkpoint: {ckpt}")
