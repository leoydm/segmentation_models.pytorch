from . import datasets
from . import encoders
from . import decoders
from . import losses
from . import metrics

from .decoders.unet import Unet, UnetDiff, UnetRefiner
from .decoders.unetplusplus import UnetPlusPlus, UnetPlusPlusDiff
from .decoders.manet import MAnet, MAnetDiff
from .decoders.linknet import Linknet, LinknetDiff
from .decoders.fpn import FPN, FPNDiff
from .decoders.pspnet import PSPNet, PSPNetDiff
from .decoders.deeplabv3 import DeepLabV3, DeepLabV3Plus, DeepLabV3Diff, DeepLabV3PlusDiff, DeepLabV3Refiner, DeepLabV3PlusRefiner, DeepLabV3PlusDiffS
from .decoders.pan import PAN, PANDiff
from .decoders.upernet import UPerNet, UPerNetDiff
from .decoders.segformer import Segformer, SegformerDiff, SegformerRefiner
from .decoders.dpt import DPT
from .base.hub_mixin import from_pretrained

from .__version__ import __version__

# some private imports for create_model function
from typing import Optional as _Optional
import torch as _torch

_MODEL_ARCHITECTURES = [
    Unet,
    UnetDiff,
    UnetRefiner,
    UnetPlusPlus,
    UnetPlusPlusDiff,
    MAnet,
    MAnetDiff,
    Linknet,
    LinknetDiff,
    FPN,
    FPNDiff,
    PSPNet,
    PSPNetDiff,
    DeepLabV3,
    DeepLabV3Plus,
    DeepLabV3Diff,
    DeepLabV3PlusDiff,
    DeepLabV3Refiner,
    DeepLabV3PlusRefiner,
    PAN,
    PANDiff,
    UPerNet,
    UPerNetDiff,
    Segformer,
    SegformerDiff,
    SegformerRefiner,
    DPT,
    DeepLabV3PlusDiffS,
]
MODEL_ARCHITECTURES_MAPPING = {a.__name__.lower(): a for a in _MODEL_ARCHITECTURES}


def create_model(
    arch: str,
    encoder_name: str = "resnet34",
    encoder_weights: _Optional[str] = "imagenet",
    in_channels: int = 3,
    classes: int = 1,
    **kwargs,
) -> _torch.nn.Module:
    """Models entrypoint, allows to create any model architecture just with
    parameters, without using its class
    """

    try:
        model_class = MODEL_ARCHITECTURES_MAPPING[arch.lower()]
    except KeyError:
        raise KeyError(
            "Wrong architecture type `{}`. Available options are: {}".format(
                arch, list(MODEL_ARCHITECTURES_MAPPING.keys())
            )
        )
    return model_class(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
        **kwargs,
    )


__all__ = [
    "datasets",
    "encoders",
    "decoders",
    "losses",
    "metrics",
    "Unet",
    "UnetDiff",
    "UnetPlusPlus",
    "UnetPlusPlusDiff",
    "MAnet",
    "MAnetDiff",
    "Linknet",
    "LinknetDiff",
    "FPN",
    "FPNDiff",
    "PSPNet",
    "PSPNetDiff",
    "DeepLabV3",
    "DeepLabV3Plus",
    "DeepLabV3Diff",
    "DeepLabV3PlusDiff",
    "DeepLabV3PlusDiffS",
    "PAN",
    "PANDiff",
    "UPerNet",
    "UPerNetDiff",
    "Segformer",
    "SegformerDiff",
    "DPT",
    "DPTDiff",
    "from_pretrained",
    "create_model",
    "__version__",
]
