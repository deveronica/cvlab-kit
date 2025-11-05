# cvlabkit/component/transform/flow_added_rand_augment.py
"""RandAugment with Flow augmentation added to the operation pool."""

import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import get_dimensions
from torchvision.transforms.autoaugment import _apply_op

from cvlabkit.component.base import Transform
from cvlabkit.core.config import Config


class FlowAddedRandAugment(Transform):
    """RandAugment with Flow as an additional augmentation option.

    Adds pretrained flow-based augmentation to PyTorch's RandAugment pool.
    Flow becomes the 15th operation alongside Identity, ShearX, etc.

    Magnitude directly maps to flow timestep t:
        magnitude=0  → t=0.0 (no augmentation, original image)
        magnitude=15 → t=0.5 (medium augmentation)
        magnitude=30 → t=1.0 (maximum augmentation)

    Args:
        cfg (Config): Configuration with keys:
            - num_ops (int): Number of ops to apply sequentially (default: 2)
            - magnitude (int): Magnitude index 0-30 (default: 9)
            - num_magnitude_bins (int): Number of magnitude bins (default: 31)
            - flow_checkpoint (str): Path to pretrained flow generator
            - flow_steps (int): ODE solver steps (default: 4)
            - generator (str): Generator model type (default: "unet")
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.num_ops = cfg.get("num_ops", 2)
        self.magnitude = cfg.get("magnitude", 9)
        self.num_magnitude_bins = cfg.get("num_magnitude_bins", 31)
        self.interpolation = InterpolationMode.NEAREST
        self.fill = None

        # Flow-specific configs
        self.flow_checkpoint = cfg.get("flow_checkpoint")
        self.flow_steps = cfg.get("flow_steps", 4)
        self.generator_type = cfg.get("generator", "unet")

        # Load flow generator
        self.generator = None
        self.device = None
        if self.flow_checkpoint:
            self._load_generator(cfg)

    def _load_generator(self, cfg: Config):
        """Load pretrained flow generator."""
        checkpoint_path = Path(self.flow_checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Flow checkpoint not found: {checkpoint_path}")

        # Determine device
        device_id = cfg.get("device", 0)
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{device_id}")
        else:
            self.device = torch.device("cpu")

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Create generator using Creator pattern
        from cvlabkit.core.creator import Creator
        temp_cfg = Config({"model": self.generator_type})
        temp_creator = Creator(temp_cfg)
        self.generator = temp_creator.model().to(self.device)

        # Load weights
        if "model" in checkpoint:
            self.generator.load_state_dict(checkpoint["model"])
        else:
            self.generator.load_state_dict(checkpoint)

        self.generator.eval()

        # Freeze parameters
        for param in self.generator.parameters():
            param.requires_grad = False

        print(f"FlowAddedRandAugment: Loaded generator from {checkpoint_path}")
        print(f"  Device: {self.device}")
        print(f"  Flow steps: {self.flow_steps}")

    def _augmentation_space(self, num_bins: int, image_size: tuple[int, int]) -> dict:
        """Define augmentation space with Flow added.

        Returns:
            dict: Mapping of op_name → (magnitudes, signed)
        """
        ops = {
            # Standard RandAugment operations
            "Identity": (torch.tensor(0.0), False),
            "ShearX": (torch.linspace(0.0, 0.3, num_bins), True),
            "ShearY": (torch.linspace(0.0, 0.3, num_bins), True),
            "TranslateX": (torch.linspace(0.0, 150.0 / 331.0 * image_size[1], num_bins), True),
            "TranslateY": (torch.linspace(0.0, 150.0 / 331.0 * image_size[0], num_bins), True),
            "Rotate": (torch.linspace(0.0, 30.0, num_bins), True),
            "Brightness": (torch.linspace(0.0, 0.9, num_bins), True),
            "Color": (torch.linspace(0.0, 0.9, num_bins), True),
            "Contrast": (torch.linspace(0.0, 0.9, num_bins), True),
            "Sharpness": (torch.linspace(0.0, 0.9, num_bins), True),
            "Posterize": (8 - (torch.arange(num_bins) / ((num_bins - 1) / 4)).round().int(), False),
            "Solarize": (torch.linspace(255.0, 0.0, num_bins), False),
            "AutoContrast": (torch.tensor(0.0), False),
            "Equalize": (torch.tensor(0.0), False),
        }

        # Add Flow operation: magnitude maps directly to t
        if self.generator is not None:
            # magnitude 0 → t=0.0 (no aug), magnitude 30 → t=1.0 (max aug)
            ops["Flow"] = (torch.linspace(0.0, 1.0, num_bins), False)

        return ops

    def _apply_flow(self, img, t_target: float):
        """Apply flow-based augmentation.

        Args:
            img: PIL Image or Tensor
            t_target: Target timestep in [0, 1]
                     0.0 = no augmentation (return original)
                     1.0 = maximum augmentation

        Returns:
            Augmented image (same type as input)
        """
        # Special case: t=0 means no augmentation
        if t_target <= 0.0:
            return img

        is_pil = isinstance(img, Image.Image)

        # Convert to tensor [0, 1]
        if is_pil:
            to_tensor = transforms.ToTensor()
            img_tensor = to_tensor(img)
        else:
            img_tensor = img
            # Ensure [0, 1] range
            if img_tensor.dtype == torch.uint8:
                img_tensor = img_tensor.float() / 255.0

        # Add batch dimension and move to device
        x_0 = img_tensor.unsqueeze(0).to(self.device)
        t = torch.tensor([t_target], device=self.device)

        # Euler ODE solver: integrate from t=0 to t=t_target
        with torch.no_grad():
            x_t = x_0.clone()
            for i in range(self.flow_steps):
                t_current = t * (i / self.flow_steps)
                v_t = self.generator(x_t, t_current)
                dt = t / self.flow_steps
                x_t = x_t + v_t * dt.view(-1, 1, 1, 1)

        # Remove batch dimension
        result = x_t.squeeze(0).cpu()

        # Convert back to original type
        if is_pil:
            to_pil = transforms.ToPILImage()
            result = to_pil(result)
        elif img.dtype == torch.uint8:
            result = (result * 255.0).clamp(0, 255).byte()

        return result

    def __call__(self, img, **kwargs):
        """Apply RandAugment with Flow option.

        Args:
            img: PIL Image or Tensor
            **kwargs: Additional arguments (ignored, for compatibility)

        Returns:
            Augmented image (same type as input)
        """
        fill = self.fill
        channels, height, width = get_dimensions(img)

        if isinstance(img, torch.Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * channels
            elif fill is not None:
                fill = [float(f) for f in fill]

        # Get augmentation space
        op_meta = self._augmentation_space(self.num_magnitude_bins, (height, width))

        # Apply num_ops random operations
        for _ in range(self.num_ops):
            op_index = int(torch.randint(len(op_meta), (1,)).item())
            op_name = list(op_meta.keys())[op_index]
            magnitudes, signed = op_meta[op_name]
            magnitude = float(magnitudes[self.magnitude].item()) if magnitudes.ndim > 0 else 0.0

            if signed and torch.randint(2, (1,)):
                magnitude *= -1.0

            # Apply operation
            if op_name == "Flow":
                img = self._apply_flow(img, magnitude)
            else:
                img = _apply_op(img, op_name, magnitude,
                              interpolation=self.interpolation, fill=fill)

        return img

    def __repr__(self) -> str:
        flow_info = f", flow_checkpoint={self.flow_checkpoint}" if self.generator else ""
        return (
            f"{self.__class__.__name__}("
            f"num_ops={self.num_ops}"
            f", magnitude={self.magnitude}"
            f", num_magnitude_bins={self.num_magnitude_bins}"
            f", interpolation={self.interpolation}"
            f", fill={self.fill}"
            f"{flow_info}"
            f")"
        )
