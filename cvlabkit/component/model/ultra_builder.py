import contextlib
import re
from copy import deepcopy
from typing import List, Tuple

import torch
import torch.nn as nn

from cvlabkit.component.base import Model

# Placeholder for external Ultralytics components and utilities.
# In a real Ultralytics environment, these would be imported from their respective modules.
# For the purpose of documentation and basic understanding, we assume their existence.


# Dummy LOGGER for documentation purposes
class DummyLogger:
    def warning(self, msg):
        pass

    def info(self, msg):
        pass


LOGGER = DummyLogger()


# Dummy classes for model parsing (replace with actual Ultralytics classes if available)
class Detect(nn.Module):
    pass


class WorldDetect(nn.Module):
    pass


class YOLOEDetect(nn.Module):
    pass


class Segment(nn.Module):
    pass


class YOLOESegment(nn.Module):
    pass


class Pose(nn.Module):
    pass


class OBB(nn.Module):
    pass


class ImagePoolingAttn(nn.Module):
    pass


class v10Detect(nn.Module):
    pass


class RTDETRDecoder(nn.Module):
    pass


class CBLinear(nn.Module):
    pass


class CBFuse(nn.Module):
    pass


class TorchVision(nn.Module):
    pass


class Index(nn.Module):
    pass


class HGStem(nn.Module):
    pass


class HGBlock(nn.Module):
    pass


class ResNetLayer(nn.Module):
    pass


class Classify(nn.Module):
    pass


class Conv(nn.Module):
    pass


class ConvTranspose(nn.Module):
    pass


class GhostConv(nn.Module):
    pass


class Bottleneck(nn.Module):
    pass


class GhostBottleneck(nn.Module):
    pass


class SPP(nn.Module):
    pass


class SPPF(nn.Module):
    pass


class C2fPSA(nn.Module):
    pass


class C2PSA(nn.Module):
    pass


class DWConv(nn.Module):
    pass


class Focus(nn.Module):
    pass


class BottleneckCSP(nn.Module):
    pass


class C1(nn.Module):
    pass


class C2(nn.Module):
    pass


class C2f(nn.Module):
    pass


class C3k2(nn.Module):
    pass


class RepNCSPELAN4(nn.Module):
    pass


class ELAN1(nn.Module):
    pass


class ADown(nn.Module):
    pass


class AConv(nn.Module):
    pass


class SPPELAN(nn.Module):
    pass


class C2fAttn(nn.Module):
    pass


class C3(nn.Module):
    pass


class C3TR(nn.Module):
    pass


class C3Ghost(nn.Module):
    pass


class C3x(nn.Module):
    pass


class RepC3(nn.Module):
    pass


class PSA(nn.Module):
    pass


class SCDown(nn.Module):
    pass


class C2fCIB(nn.Module):
    pass


class A2C2f(nn.Module):
    pass


# Dummy functions for model parsing (replace with actual Ultralytics functions)
def initialize_weights(model):
    pass


def make_divisible(x, divisor):
    return x


def check_yaml(path, hard=False):
    return path


def guess_model_scale(path):
    return "n"  # default to nano


class YAML:
    @staticmethod
    def load(path):
        return {"backbone": [], "head": [], "nc": 80, "activation": "SiLU"}


def colorstr(s):
    return s


class UltraBuilder(Model, nn.Module):
    """Ultralytics-style model builder that loads a YAML config,
    constructs the network, and provides a unified API.

    This component is designed to dynamically build a neural network model
    based on a YAML configuration file, typically used by Ultralytics YOLO
    models. It parses the YAML to construct the backbone and head of the model
    using various predefined modules.

    It inherits from both `cvlabkit.component.base.Model` (for framework compatibility)
    and `torch.nn.Module` (for PyTorch model functionality).

    Attributes:
        yaml (dict): The parsed model configuration dictionary from the YAML file.
        model (torch.nn.Sequential): The constructed neural network model.
        save (list): A list of layer indices whose outputs are saved for later use
            (e.g., feature maps for FPN).
        names (dict): A dictionary mapping class IDs to class names.
        inplace (bool): Whether to use inplace operations.
        end2end (bool): Flag indicating if the model has an end-to-end detection head.
        stride (torch.Tensor): The stride of the model's output features.

    Examples:
        Initialize a detection model:
        ```python
        # Assuming cfg.model = "ultra_builder" and cfg.model_path = "models/yolov5.yaml"
        model = create.model()
        results = model(image_tensor)
        ```
    """

    def __init__(self, cfg):
        """Initializes the YOLO detection model with the given config and parameters.

        Args:
            cfg (Config): The configuration object. Expected parameters:
                - `model_path` (str): Path to the YAML model configuration file.
                - `channel` (int): Input channels for the model.
                - `num_class` (int, optional): Number of output classes. Overrides
                  `nc` in the YAML if provided.
        """
        super().__init__()
        # Load model configuration from YAML file.
        self.yaml = yaml_model_load(cfg.model_path)

        # Handle deprecated 'Silence' module in YOLOv9.
        # This ensures compatibility with newer PyTorch versions.
        if self.yaml["backbone"][0][2] == "Silence":
            LOGGER.warning(
                "YOLOv9 `Silence` module is deprecated in favor of torch.nn.Identity. "
                "Please delete local *.pt file and re-download the latest model checkpoint."
            )
            self.yaml["backbone"][0][2] = "nn.Identity"

        # Set input channels and potentially override number of classes.
        self.yaml["channels"] = cfg.channel  # save channels
        nc = self.yaml.get("nc", None)
        if cfg.num_class and nc != cfg.num_class:
            LOGGER.info(f"Overriding model.yaml nc={nc} with nc={cfg.num_class}")
            self.yaml["nc"] = cfg.num_class  # override YAML value

        # Parse the model definition from the YAML to build the PyTorch model.
        # `parse_model` is an external helper function from Ultralytics.
        self.model, self.save = parse_model(
            deepcopy(self.yaml), cfg.channel
        )  # model, savelist
        self.names = {i: f"{i}" for i in range(self.yaml["nc"])}  # default names dict
        self.inplace = self.yaml.get("inplace", True)
        self.end2end = getattr(self.model[-1], "end2end", False)

        # Build strides for detection models.
        m = self.model[-1]  # Detect() or similar head module
        if isinstance(
            m,
            (
                Detect,
                WorldDetect,
                YOLOEDetect,
                Segment,
                YOLOESegment,
                Pose,
                OBB,
                ImagePoolingAttn,
                v10Detect,
            ),
        ):  # includes all Detect subclasses
            s = 256  # 2x min stride
            m.inplace = self.inplace

            # Define a nested forward function for stride calculation.
            def _forward(x):
                """Perform a forward pass through the model, handling different Detect subclass types accordingly."""
                if self.end2end:
                    return self.forward(x)["one2many"]
                # For segmentation, pose, OBB, etc., the first element of the tuple is usually the main output.
                return (
                    self.forward(x)[0]
                    if isinstance(m, (Segment, YOLOESegment, Pose, OBB))
                    else self.forward(x)
                )

            self.model.eval()  # Avoid changing batch statistics until training begins
            m.training = True  # Setting it to True to properly return strides
            # Calculate strides by passing a dummy input through the model.
            self.stride = torch.tensor(
                [s / x.shape[-2] for x in _forward(torch.zeros(1, cfg.channel, s, s))]
            )  # forward
            self.model.train()  # Set model back to training(default) mode
            if hasattr(m, "bias_init"):
                m.bias_init()  # only run once
        else:
            self.stride = torch.Tensor([32])  # default stride for i.e. RTDETR

        # Initialize model weights and biases.
        # `initialize_weights` is an external helper function.
        initialize_weights(self)
        # TODO: Add a configurable verbose option for model info.
        # if verbose:
        #     self.info()
        #     LOGGER.info("")

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Defines the forward pass of the model.

        This method delegates the call directly to the constructed `self.model`.

        Args:
            x (torch.Tensor): The input tensor (e.g., a batch of images).
            *args: Positional arguments to pass to the underlying model.
            **kwargs: Keyword arguments to pass to the underlying model.

        Returns:
            torch.Tensor: The output tensor from the model.
        """
        return self.model(x, *args, **kwargs)

    def parameters(self, recurse: bool = True):
        """Returns an iterator over the model parameters.

        This method delegates to the underlying `self.model.parameters()`
        to provide access to the trainable parameters of the constructed model.

        Args:
            recurse (bool): If True, will return parameters of all submodules.
                Defaults to True.

        Returns:
            An iterator over the model parameters.
        """
        return self.model.parameters(recurse)


def parse_model(
    d: dict, ch: int, verbose: bool = True
) -> Tuple[nn.Sequential, List[int]]:
    """Parses a YOLO model.yaml dictionary into a PyTorch model.

    This is a highly complex function that dynamically constructs a PyTorch
    model based on a dictionary representation of the model architecture.
    It relies heavily on internal Ultralytics module definitions and logic.

    Args:
        d (dict): Model dictionary, typically loaded from a YOLO YAML file.
        ch (int): Input channels for the model.
        verbose (bool): Whether to print model details during parsing.

    Returns:
        Tuple[torch.nn.Sequential, List[int]]: A tuple containing:
            - model (torch.nn.Sequential): The constructed PyTorch model.
            - save (list): A sorted list of layer indices whose outputs are saved.
    """
    import ast

    # Args
    legacy = True  # backward compatibility for v3/v5/v8/v9 models
    max_channels = float("inf")
    nc, act, scales = (d.get(x) for x in ("nc", "activation", "scales"))
    depth, width, kpt_shape = (
        d.get(x, 1.0) for x in ("depth_multiple", "width_multiple", "kpt_shape")
    )
    if scales:
        scale = d.get("scale")
        if not scale:
            scale = tuple(scales.keys())[0]
            LOGGER.warning(f"no model scale passed. Assuming scale='{scale}'.")
        depth, width, max_channels = scales[scale]

    if act:
        # Set default activation function for Conv layers.
        Conv.default_act = eval(
            act
        )  # redefine default activation, i.e. Conv.default_act = torch.nn.SiLU()
        if verbose:
            LOGGER.info(f"{colorstr('activation:')} {act}")  # print

    if verbose:
        LOGGER.info(
            f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}"
        )
    ch = [ch]
    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out

    # Define sets of modules for special handling during parsing.
    # These are typically Ultralytics-specific custom modules.
    base_modules = frozenset(
        {
            Classify,
            Conv,
            ConvTranspose,
            GhostConv,
            Bottleneck,
            GhostBottleneck,
            SPP,
            SPPF,
            C2fPSA,
            C2PSA,
            DWConv,
            Focus,
            BottleneckCSP,
            C1,
            C2,
            C2f,
            C3k2,
            RepNCSPELAN4,
            ELAN1,
            ADown,
            AConv,
            SPPELAN,
            C2fAttn,
            C3,
            C3TR,
            C3Ghost,
            torch.nn.ConvTranspose2d,
            DWConvTranspose2d,
            C3x,
            RepC3,
            PSA,
            SCDown,
            C2fCIB,
            A2C2f,
        }
    )
    repeat_modules = frozenset(  # modules with 'repeat' arguments
        {
            BottleneckCSP,
            C1,
            C2,
            C2f,
            C3k2,
            C2fAttn,
            C3,
            C3TR,
            C3Ghost,
            C3x,
            RepC3,
            C2fPSA,
            C2fCIB,
            C2PSA,
            A2C2f,
        }
    )

    # Iterate through backbone and head definitions in the YAML.
    for i, (f, n, m, args) in enumerate(
        d["backbone"] + d["head"]
    ):  # from, number, module, args
        # Dynamically get the module class.
        m = (
            getattr(torch.nn, m[3:])
            if "nn." in m
            else getattr(__import__("torchvision").ops, m[16:])
            if "torchvision.ops." in m
            else globals()[m]  # Custom Ultralytics modules are in globals
        )  # get module

        # Evaluate string arguments if they are valid Python literals.
        for j, a in enumerate(args):
            if isinstance(a, str):
                with contextlib.suppress(ValueError):
                    args[j] = locals()[a] if a in locals() else ast.literal_eval(a)

        # Apply depth gain (for scaling models like YOLOv5/v8/v9).
        n = n_ = max(round(n * depth), 1) if n > 1 else n  # depth gain

        # Handle module-specific arguments and channel calculations.
        if m in base_modules:
            c1, c2 = ch[f], args[0]
            if (
                c2 != nc
            ):  # if c2 not equal to number of classes (i.e. for Classify() output)
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            if m is C2fAttn:  # set 1) embed channels and 2) num heads
                args[1] = make_divisible(min(args[1], max_channels // 2) * width, 8)
                args[2] = int(
                    max(round(min(args[2], max_channels // 2 // 32)) * width, 1)
                    if args[2] > 1
                    else args[2]
                )

            args = [c1, c2, *args[1:]]
            if m in repeat_modules:
                args.insert(2, n)  # number of repeats
                n = 1
            if m is C3k2:  # for M/L/X sizes
                legacy = False
                if scale in "mlx":
                    args[3] = True
            if m is A2C2f:
                legacy = False
                if scale in "lx":  # for L/X sizes
                    args.extend((True, 1.2))
            if m is C2fCIB:
                legacy = False
        elif m is AIFI:
            args = [ch[f], *args]
        elif m in frozenset({HGStem, HGBlock}):
            c1, cm, c2 = ch[f], args[0], args[1]
            args = [c1, cm, c2, *args[2:]]
            if m is HGBlock:
                args.insert(4, n)  # number of repeats
                n = 1
        elif m is ResNetLayer:
            c2 = args[1] if args[3] else args[1] * 4
        elif m is torch.nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m in frozenset(
            {
                Detect,
                WorldDetect,
                YOLOEDetect,
                Segment,
                YOLOESegment,
                Pose,
                OBB,
                ImagePoolingAttn,
                v10Detect,
            }
        ):
            args.append([ch[x] for x in f])
            if m is Segment or m is YOLOESegment:
                args[2] = make_divisible(min(args[2], max_channels) * width, 8)
            if m in {Detect, YOLOEDetect, Segment, YOLOESegment, Pose, OBB}:
                m.legacy = legacy
        elif m is RTDETRDecoder:  # special case, channels arg must be passed in index 1
            args.insert(1, [ch[x] for x in f])
        elif m is CBLinear:
            c2 = args[0]
            c1 = ch[f]
            args = [c1, c2, *args[1:]]
        elif m is CBFuse:
            c2 = ch[f[-1]]
        elif m in frozenset({TorchVision, Index}):
            c2 = args[0]
            c1 = ch[f]
            args = [*args[1:]]
        else:
            c2 = ch[f]

        # Create the module instance(s).
        m_ = (
            torch.nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)
        )  # module
        t = str(m)[8:-2].replace("__main__.", "")  # module type string
        m_.np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type = i, f, t  # attach index, 'from' index, type
        if verbose:
            LOGGER.info(
                f"{i:>3}{str(f):>20}{n_:>3}{m_.np:10.0f}  {t:<45}{str(args):<30}"
            )  # print
        # Add layer index to save list if its output is needed by subsequent layers.
        save.extend(
            x % i for x in ([f] if isinstance(f, int) else f) if x != -1
        )  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []  # Clear channels after first layer (input layer)
        ch.append(c2)  # Add output channels of current layer
    return torch.nn.Sequential(*layers), sorted(save)


def yaml_model_load(path: str) -> dict:
    """Loads a YOLO model configuration from a YAML file.

    This function handles loading the YAML file and performs some compatibility
    adjustments for Ultralytics YOLO models, such as handling P6 suffix and
    guessing model scale.

    Args:
        path (str): Path to the YAML file.

    Returns:
        dict: The loaded model configuration dictionary.
    """
    from pathlib import Path

    # TODO: Replace direct regex with more robust path handling if needed.
    path = Path(path)
    if path.stem in (f"yolov{d}{x}6" for x in "nsmlx" for d in (5, 8)):
        new_stem = re.sub(r"(\d+)([nslmx])6(.+)?$", r"\1\2-p6\3", path.stem)
        LOGGER.warning(
            f"Ultralytics YOLO P6 models now use -p6 suffix. Renaming {path.stem} to {new_stem}."
        )
        path = path.with_name(new_stem + path.suffix)

    # Unify path for older YOLO versions (e.g., yolov8x.yaml -> yolov8.yaml).
    unified_path = re.sub(
        r"(\d+)([nslmx])(.+)?$", r"\1\3", str(path)
    )  # i.e. yolov8x.yaml -> yolov8.yaml
    # Check for YAML file existence.
    yaml_file = check_yaml(unified_path, hard=False) or check_yaml(path)
    # Load the YAML content.
    d = YAML.load(yaml_file)  # model dict
    # Guess model scale and add YAML file path to the dictionary.
    d["scale"] = guess_model_scale(path)
    d["yaml_file"] = str(path)
    return d
