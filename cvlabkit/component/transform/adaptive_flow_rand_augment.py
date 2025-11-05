# cvlabkit/component/transform/adaptive_flow_rand_augment.py
"""AdaptiveRandAugment with Flow added to the augmentation pool."""

from cvlabkit.component.base import Transform
from cvlabkit.component.transform.flow_added_rand_augment import FlowAddedRandAugment
from cvlabkit.core.config import Config


class AdaptiveFlowRandAugment(Transform):
    """AdaptiveRandAugment with Flow as 15th operation in augmentation pool.

    Combines adaptive difficulty-based augmentation with flow-based transformations.
    The augmentation pool contains 15 operations: 14 standard RandAugment ops + Flow.

    Args:
        cfg (Config): Configuration with keys:
            - magnitude_min (int): Minimum magnitude for hardest samples (score=1.0)
            - magnitude_max (int): Maximum magnitude for easiest samples (score=0.0)
            - num_ops (int): Number of ops to apply sequentially (default: 2)
            - flow_checkpoint (str): Path to pretrained flow generator
            - flow_steps (int): ODE solver steps (default: 4)
            - generator (str): Generator model type (default: "unet")
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.magnitude_min = cfg.get("magnitude_min")
        self.magnitude_max = cfg.get("magnitude_max")
        self.num_ops = cfg.get("num_ops", 2)

        if self.magnitude_min is None or self.magnitude_max is None:
            raise ValueError(
                "AdaptiveFlowRandAugment requires 'magnitude_min' and 'magnitude_max' parameters."
            )

        # Create FlowAddedRandAugment instance (will be updated per call)
        self.flow_checkpoint = cfg.get("flow_checkpoint")
        self.flow_steps = cfg.get("flow_steps", 4)
        self.generator_type = cfg.get("generator", "unet")

        # Create base config for FlowAddedRandAugment
        self.base_cfg = Config({
            "num_ops": self.num_ops,
            "magnitude": 0,  # Will be updated dynamically
            "num_magnitude_bins": 31,
            "flow_checkpoint": self.flow_checkpoint,
            "flow_steps": self.flow_steps,
            "generator": self.generator_type,
            "device": cfg.get("device", 0),
        })

        # Create FlowAddedRandAugment (loads generator once)
        self.augmenter = FlowAddedRandAugment(self.base_cfg)

    def __call__(self, sample, **kwargs):
        """Apply FlowAddedRandAugment with adaptive magnitude.

        Args:
            sample (PIL.Image): Input image
            **kwargs: Expected to contain 'difficulty_score' (float in [0, 1])

        Returns:
            PIL.Image: Augmented image
        """
        # Default to easiest case if no score provided
        difficulty_score = kwargs.get("difficulty_score", 0.0)

        # Linear interpolation for magnitude
        # High difficulty → High magnitude (strong augmentation)
        magnitude = self.magnitude_min + (self.magnitude_max - self.magnitude_min) * difficulty_score
        final_magnitude = int(round(magnitude))

        # Update magnitude dynamically
        self.augmenter.magnitude = final_magnitude

        # Apply FlowAddedRandAugment (15 ops including Flow)
        return self.augmenter(sample)

    def __repr__(self) -> str:
        flow_info = f", flow_checkpoint={self.flow_checkpoint}" if self.flow_checkpoint else ""
        return (
            f"{self.__class__.__name__}("
            f"magnitude_min={self.magnitude_min}"
            f", magnitude_max={self.magnitude_max}"
            f", num_ops={self.num_ops}"
            f"{flow_info}"
            f")"
        )
