from collections.abc import Iterable
from typing import Any, Literal, Optional


from segmentation_models_pytorch.base import (
    ClassificationHead,
    SegmentationHead,
    SegmentationModel, DiffModel, RefinerModel
)
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base.hub_mixin import supports_config_loading

from .decoder import DeepLabV3Decoder, DeepLabV3PlusDecoder


class DeepLabV3(SegmentationModel):
    """DeepLabV3_ implementation from "Rethinking Atrous Convolution for Semantic Image Segmentation"

    Args:
        encoder_name: Name of the classification model that will be used as an encoder (a.k.a backbone)
            to extract features of different spatial resolution
        encoder_depth: A number of stages used in encoder in range [3, 5]. Each stage generate features
            two times smaller in spatial dimensions than previous one (e.g. for depth 0 we will have features
            with shapes [(N, C, H, W),], for depth 1 - [(N, C, H, W), (N, C, H // 2, W // 2)] and so on).
            Default is 5
        encoder_weights: One of **None** (random initialization), **"imagenet"** (pre-training on ImageNet) and
            other pretrained weights (see table with available weights for each encoder_name)
        decoder_channels: A number of convolution filters in ASPP module. Default is 256
        encoder_output_stride: Downsampling factor for last encoder features (see original paper for explanation)
        decoder_atrous_rates: Dilation rates for ASPP module (should be an iterable of 3 integer values)
        decoder_aspp_separable: Use separable convolutions in ASPP module. Default is False
        decoder_aspp_dropout: Use dropout in ASPP module projection layer. Default is 0.5
        in_channels: A number of input channels for the model, default is 3 (RGB images)
        classes: A number of classes for output mask (or you can think as a number of channels of output mask)
        activation: An activation function to apply after the final convolution layer.
            Available options are **"sigmoid"**, **"softmax"**, **"logsoftmax"**, **"tanh"**, **"identity"**,
            **callable** and **None**. Default is **None**.
        upsampling: Final upsampling factor. Default is **None** to preserve input-output spatial shape identity
        aux_params: Dictionary with parameters of the auxiliary output (classification head). Auxiliary output is build
            on top of encoder if **aux_params** is not **None** (default). Supported params:
                - classes (int): A number of classes
                - pooling (str): One of "max", "avg". Default is "avg"
                - dropout (float): Dropout factor in [0, 1)
                - activation (str): An activation function to apply "sigmoid"/"softmax"
                    (could be **None** to return logits)
        kwargs: Arguments passed to the encoder class ``__init__()`` function. Applies only to ``timm`` models.
            Keys with ``None`` values are pruned before passing.

    Returns:
        ``torch.nn.Module``: **DeepLabV3**

    .. _DeeplabV3:
        https://arxiv.org/abs/1706.05587

    """

    @supports_config_loading
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        encoder_output_stride: Literal[8, 16] = 8,
        decoder_channels: int = 256,
        decoder_atrous_rates: Iterable[int] = (12, 24, 36),
        decoder_aspp_separable: bool = False,
        decoder_aspp_dropout: float = 0.5,
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[str] = None,
        upsampling: Optional[int] = None,
        aux_params: Optional[dict] = None,
        **kwargs: dict[str, Any],
    ):
        super().__init__()

        if encoder_output_stride not in [8, 16]:
            raise ValueError(
                "DeeplabV3 support output stride 8 or 16, got {}.".format(
                    encoder_output_stride
                )
            )
        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
            output_stride=encoder_output_stride,
            **kwargs,
        )

        if upsampling is None:
            if encoder_depth <= 3:
                scale_factor = 2**encoder_depth
            else:
                scale_factor = encoder_output_stride
        else:
            scale_factor = upsampling

        self.decoder = DeepLabV3Decoder(
            in_channels=self.encoder.out_channels[-1],
            out_channels=decoder_channels,
            atrous_rates=decoder_atrous_rates,
            aspp_separable=decoder_aspp_separable,
            aspp_dropout=decoder_aspp_dropout,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels,
            out_channels=classes,
            activation=activation,
            kernel_size=1,
            upsampling=scale_factor,
        )

        if aux_params is not None:
            self.classification_head = ClassificationHead(
                in_channels=self.encoder.out_channels[-1], **aux_params
            )
        else:
            self.classification_head = None
        self.name = "deeplabv3-{}".format(encoder_name)

    def load_state_dict(self, state_dict, *args, **kwargs):
        # For backward compatibility, previously Decoder module was Sequential
        # and was not scriptable.
        keys = list(state_dict.keys())
        for key in keys:
            new_key = key
            if key.startswith("decoder.0."):
                new_key = key.replace("decoder.0.", "decoder.aspp.")
            elif key.startswith("decoder.1."):
                new_key = key.replace("decoder.1.", "decoder.conv.")
            elif key.startswith("decoder.2."):
                new_key = key.replace("decoder.2.", "decoder.bn.")
            state_dict[new_key] = state_dict.pop(key)
        return super().load_state_dict(state_dict, *args, **kwargs)


class DeepLabV3Plus(SegmentationModel):
    """DeepLabV3+ implementation from "Encoder-Decoder with Atrous Separable
    Convolution for Semantic Image Segmentation"

    Args:
        encoder_name: Name of the classification model that will be used as an encoder (a.k.a backbone)
            to extract features of different spatial resolution
        encoder_depth: A number of stages used in encoder in range [3, 5]. Each stage generate features
            two times smaller in spatial dimensions than previous one (e.g. for depth 0 we will have features
            with shapes [(N, C, H, W),], for depth 1 - [(N, C, H, W), (N, C, H // 2, W // 2)] and so on).
            Default is 5
        encoder_weights: One of **None** (random initialization), **"imagenet"** (pre-training on ImageNet) and
            other pretrained weights (see table with available weights for each encoder_name)
        encoder_output_stride: Downsampling factor for last encoder features (see original paper for explanation)
        decoder_atrous_rates: Dilation rates for ASPP module (should be an iterable of 3 integer values)
        decoder_aspp_separable: Use separable convolutions in ASPP module. Default is True
        decoder_aspp_dropout: Use dropout in ASPP module projection layer. Default is 0.5
        decoder_channels: A number of convolution filters in ASPP module. Default is 256
        in_channels: A number of input channels for the model, default is 3 (RGB images)
        classes: A number of classes for output mask (or you can think as a number of channels of output mask)
        activation: An activation function to apply after the final convolution layer.
            Available options are **"sigmoid"**, **"softmax"**, **"logsoftmax"**, **"tanh"**, **"identity"**,
            **callable** and **None**. Default is **None**.
        upsampling: Final upsampling factor. Default is 4 to preserve input-output spatial shape identity.
        aux_params: Dictionary with parameters of the auxiliary output (classification head). Auxiliary output is build
            on top of encoder if **aux_params** is not **None** (default). Supported params:
                - classes (int): A number of classes
                - pooling (str): One of "max", "avg". Default is "avg"
                - dropout (float): Dropout factor in [0, 1)
                - activation (str): An activation function to apply "sigmoid"/"softmax"
                    (could be **None** to return logits)
        kwargs: Arguments passed to the encoder class ``__init__()`` function. Applies only to ``timm`` models.
            Keys with ``None`` values are pruned before passing.

    Returns:
        ``torch.nn.Module``: **DeepLabV3Plus**

    Reference:
        https://arxiv.org/abs/1802.02611v3

    """

    @supports_config_loading
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: Literal[3, 4, 5] = 5,
        encoder_weights: Optional[str] = "imagenet",
        encoder_output_stride: Literal[8, 16] = 16,
        decoder_channels: int = 256,
        decoder_atrous_rates: Iterable[int] = (12, 24, 36),
        decoder_aspp_separable: bool = True,
        decoder_aspp_dropout: float = 0.5,
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[str] = None,
        upsampling: int = 4,
        aux_params: Optional[dict] = None,
        **kwargs: dict[str, Any],
    ):
        super().__init__()

        if encoder_output_stride not in [8, 16]:
            raise ValueError(
                "DeeplabV3Plus support output stride 8 or 16, got {}.".format(
                    encoder_output_stride
                )
            )

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
            output_stride=encoder_output_stride,
            **kwargs,
        )

        self.decoder = DeepLabV3PlusDecoder(
            encoder_channels=self.encoder.out_channels,
            encoder_depth=encoder_depth,
            out_channels=decoder_channels,
            atrous_rates=decoder_atrous_rates,
            output_stride=encoder_output_stride,
            aspp_separable=decoder_aspp_separable,
            aspp_dropout=decoder_aspp_dropout,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels,
            out_channels=classes,
            activation=activation,
            kernel_size=1,
            upsampling=upsampling,
        )

        if aux_params is not None:
            self.classification_head = ClassificationHead(
                in_channels=self.encoder.out_channels[-1], **aux_params
            )
        else:
            self.classification_head = None
        self.name = "deeplabv3plus-{}".format(encoder_name)


class DeepLabV3Diff(DiffModel):
    """DeepLabV3_ implementation from "Rethinking Atrous Convolution for Semantic Image Segmentation"

    Args:
        encoder_name: Name of the classification model that will be used as an encoder (a.k.a backbone)
            to extract features of different spatial resolution
        encoder_depth: A number of stages used in encoder in range [3, 5]. Each stage generate features
            two times smaller in spatial dimensions than previous one (e.g. for depth 0 we will have features
            with shapes [(N, C, H, W),], for depth 1 - [(N, C, H, W), (N, C, H // 2, W // 2)] and so on).
            Default is 5
        encoder_weights: One of **None** (random initialization), **"imagenet"** (pre-training on ImageNet) and
            other pretrained weights (see table with available weights for each encoder_name)
        decoder_channels: A number of convolution filters in ASPP module. Default is 256
        encoder_output_stride: Downsampling factor for last encoder features (see original paper for explanation)
        decoder_atrous_rates: Dilation rates for ASPP module (should be an iterable of 3 integer values)
        decoder_aspp_separable: Use separable convolutions in ASPP module. Default is False
        decoder_aspp_dropout: Use dropout in ASPP module projection layer. Default is 0.5
        in_channels: A number of input channels for the model, default is 3 (RGB images)
        classes: A number of classes for output mask (or you can think as a number of channels of output mask)
        activation: An activation function to apply after the final convolution layer.
            Available options are **"sigmoid"**, **"softmax"**, **"logsoftmax"**, **"tanh"**, **"identity"**,
            **callable** and **None**. Default is **None**.
        upsampling: Final upsampling factor. Default is **None** to preserve input-output spatial shape identity
        aux_params: Dictionary with parameters of the auxiliary output (classification head). Auxiliary output is build
            on top of encoder if **aux_params** is not **None** (default). Supported params:
                - classes (int): A number of classes
                - pooling (str): One of "max", "avg". Default is "avg"
                - dropout (float): Dropout factor in [0, 1)
                - activation (str): An activation function to apply "sigmoid"/"softmax"
                    (could be **None** to return logits)
        kwargs: Arguments passed to the encoder class ``__init__()`` function. Applies only to ``timm`` models.
            Keys with ``None`` values are pruned before passing.

    Returns:
        ``torch.nn.Module``: **DeepLabV3**

    .. _DeeplabV3:
        https://arxiv.org/abs/1706.05587

    """

    @supports_config_loading
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        encoder_output_stride: Literal[8, 16] = 8,
        decoder_channels: int = 256,
        decoder_atrous_rates: Iterable[int] = (12, 24, 36),
        decoder_aspp_separable: bool = False,
        decoder_aspp_dropout: float = 0.5,
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[str] = None,
        upsampling: Optional[int] = None,
        aux_params: Optional[dict] = None,
        **kwargs: dict[str, Any],
    ):
        super().__init__()

        if encoder_output_stride not in [8, 16]:
            raise ValueError(
                "DeeplabV3 support output stride 8 or 16, got {}.".format(
                    encoder_output_stride
                )
            )
        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
            output_stride=encoder_output_stride,
            **kwargs,
        )

        if upsampling is None:
            if encoder_depth <= 3:
                scale_factor = 2**encoder_depth
            else:
                scale_factor = encoder_output_stride
        else:
            scale_factor = upsampling

        encoder_out_channels = [num * 2 for num in self.encoder.out_channels]

        self.decoder = DeepLabV3Decoder(
            in_channels=encoder_out_channels[-1],
            out_channels=decoder_channels,
            atrous_rates=decoder_atrous_rates,
            aspp_separable=decoder_aspp_separable,
            aspp_dropout=decoder_aspp_dropout,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels,
            out_channels=classes,
            activation=activation,
            kernel_size=1,
            upsampling=scale_factor,
        )

        if aux_params is not None:
            self.classification_head = ClassificationHead(
                in_channels=encoder_out_channels[-1], **aux_params
            )
        else:
            self.classification_head = None
        self.name = "deeplabv3-{}".format(encoder_name)

    def load_state_dict(self, state_dict, *args, **kwargs):
        # For backward compatibility, previously Decoder module was Sequential
        # and was not scriptable.
        keys = list(state_dict.keys())
        for key in keys:
            new_key = key
            if key.startswith("decoder.0."):
                new_key = key.replace("decoder.0.", "decoder.aspp.")
            elif key.startswith("decoder.1."):
                new_key = key.replace("decoder.1.", "decoder.conv.")
            elif key.startswith("decoder.2."):
                new_key = key.replace("decoder.2.", "decoder.bn.")
            state_dict[new_key] = state_dict.pop(key)
        return super().load_state_dict(state_dict, *args, **kwargs)


class DeepLabV3PlusDiff(DiffModel):
    """DeepLabV3+ implementation from "Encoder-Decoder with Atrous Separable
    Convolution for Semantic Image Segmentation"

    Args:
        encoder_name: Name of the classification model that will be used as an encoder (a.k.a backbone)
            to extract features of different spatial resolution
        encoder_depth: A number of stages used in encoder in range [3, 5]. Each stage generate features
            two times smaller in spatial dimensions than previous one (e.g. for depth 0 we will have features
            with shapes [(N, C, H, W),], for depth 1 - [(N, C, H, W), (N, C, H // 2, W // 2)] and so on).
            Default is 5
        encoder_weights: One of **None** (random initialization), **"imagenet"** (pre-training on ImageNet) and
            other pretrained weights (see table with available weights for each encoder_name)
        encoder_output_stride: Downsampling factor for last encoder features (see original paper for explanation)
        decoder_atrous_rates: Dilation rates for ASPP module (should be an iterable of 3 integer values)
        decoder_aspp_separable: Use separable convolutions in ASPP module. Default is True
        decoder_aspp_dropout: Use dropout in ASPP module projection layer. Default is 0.5
        decoder_channels: A number of convolution filters in ASPP module. Default is 256
        in_channels: A number of input channels for the model, default is 3 (RGB images)
        classes: A number of classes for output mask (or you can think as a number of channels of output mask)
        activation: An activation function to apply after the final convolution layer.
            Available options are **"sigmoid"**, **"softmax"**, **"logsoftmax"**, **"tanh"**, **"identity"**,
            **callable** and **None**. Default is **None**.
        upsampling: Final upsampling factor. Default is 4 to preserve input-output spatial shape identity.
        aux_params: Dictionary with parameters of the auxiliary output (classification head). Auxiliary output is build
            on top of encoder if **aux_params** is not **None** (default). Supported params:
                - classes (int): A number of classes
                - pooling (str): One of "max", "avg". Default is "avg"
                - dropout (float): Dropout factor in [0, 1)
                - activation (str): An activation function to apply "sigmoid"/"softmax"
                    (could be **None** to return logits)
        kwargs: Arguments passed to the encoder class ``__init__()`` function. Applies only to ``timm`` models.
            Keys with ``None`` values are pruned before passing.

    Returns:
        ``torch.nn.Module``: **DeepLabV3Plus**

    Reference:
        https://arxiv.org/abs/1802.02611v3

    """

    @supports_config_loading
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: Literal[3, 4, 5] = 5,
        encoder_weights: Optional[str] = "imagenet",
        encoder_output_stride: Literal[8, 16] = 16,
        decoder_channels: int = 256,
        decoder_atrous_rates: Iterable[int] = (12, 24, 36),
        decoder_aspp_separable: bool = True,
        decoder_aspp_dropout: float = 0.5,
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[str] = None,
        upsampling: int = 4,
        aux_params: Optional[dict] = None,
        **kwargs: dict[str, Any],
    ):
        super().__init__()

        if encoder_output_stride not in [8, 16]:
            raise ValueError(
                "DeeplabV3Plus support output stride 8 or 16, got {}.".format(
                    encoder_output_stride
                )
            )

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
            output_stride=encoder_output_stride,
            **kwargs,
        )

        self.encoder2 = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
            output_stride=encoder_output_stride,
            **kwargs,
        )

        import functools

        def rsetattr(obj, attr, val):
            pre, _, post = attr.rpartition('.')
            return setattr(rgetattr(obj, pre) if pre else obj, post, val)

        def rgetattr(obj, attr, *args):
            def _getattr(obj, attr):
                return getattr(obj, attr, *args)
            return functools.reduce(_getattr, [obj] + attr.split('.'))

        for name, param in self.encoder.named_parameters():
            rsetattr(self.encoder2, name, param)

        encoder_out_channels = [num * 2 for num in self.encoder.out_channels]

        self.decoder = DeepLabV3PlusDecoder(
            encoder_channels=encoder_out_channels,
            encoder_depth=encoder_depth,
            out_channels=decoder_channels,
            atrous_rates=decoder_atrous_rates,
            output_stride=encoder_output_stride,
            aspp_separable=decoder_aspp_separable,
            aspp_dropout=decoder_aspp_dropout,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels,
            out_channels=classes,
            activation=activation,
            kernel_size=1,
            upsampling=upsampling,
        )

        if aux_params is not None:
            self.classification_head = ClassificationHead(
                in_channels=encoder_out_channels[-1], **aux_params
            )
        else:
            self.classification_head = None
        self.name = "deeplabv3plus-{}".format(encoder_name)



class DeepLabV3Refiner(RefinerModel):
    """DeepLabV3_ implementation from "Rethinking Atrous Convolution for Semantic Image Segmentation"

    Args:
        encoder_name: Name of the classification model that will be used as an encoder (a.k.a backbone)
            to extract features of different spatial resolution
        encoder_depth: A number of stages used in encoder in range [3, 5]. Each stage generate features
            two times smaller in spatial dimensions than previous one (e.g. for depth 0 we will have features
            with shapes [(N, C, H, W),], for depth 1 - [(N, C, H, W), (N, C, H // 2, W // 2)] and so on).
            Default is 5
        encoder_weights: One of **None** (random initialization), **"imagenet"** (pre-training on ImageNet) and
            other pretrained weights (see table with available weights for each encoder_name)
        decoder_channels: A number of convolution filters in ASPP module. Default is 256
        encoder_output_stride: Downsampling factor for last encoder features (see original paper for explanation)
        decoder_atrous_rates: Dilation rates for ASPP module (should be an iterable of 3 integer values)
        decoder_aspp_separable: Use separable convolutions in ASPP module. Default is False
        decoder_aspp_dropout: Use dropout in ASPP module projection layer. Default is 0.5
        in_channels: A number of input channels for the model, default is 3 (RGB images)
        classes: A number of classes for output mask (or you can think as a number of channels of output mask)
        activation: An activation function to apply after the final convolution layer.
            Available options are **"sigmoid"**, **"softmax"**, **"logsoftmax"**, **"tanh"**, **"identity"**,
            **callable** and **None**. Default is **None**.
        upsampling: Final upsampling factor. Default is **None** to preserve input-output spatial shape identity
        aux_params: Dictionary with parameters of the auxiliary output (classification head). Auxiliary output is build
            on top of encoder if **aux_params** is not **None** (default). Supported params:
                - classes (int): A number of classes
                - pooling (str): One of "max", "avg". Default is "avg"
                - dropout (float): Dropout factor in [0, 1)
                - activation (str): An activation function to apply "sigmoid"/"softmax"
                    (could be **None** to return logits)
        kwargs: Arguments passed to the encoder class ``__init__()`` function. Applies only to ``timm`` models.
            Keys with ``None`` values are pruned before passing.

    Returns:
        ``torch.nn.Module``: **DeepLabV3**

    .. _DeeplabV3:
        https://arxiv.org/abs/1706.05587

    """

    @supports_config_loading
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        encoder_output_stride: Literal[8, 16] = 8,
        decoder_channels: int = 256,
        decoder_atrous_rates: Iterable[int] = (12, 24, 36),
        decoder_aspp_separable: bool = False,
        decoder_aspp_dropout: float = 0.5,
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[str] = None,
        upsampling: Optional[int] = None,
        aux_params: Optional[dict] = None,
        **kwargs: dict[str, Any],
    ):
        super().__init__()

        if encoder_output_stride not in [8, 16]:
            raise ValueError(
                "DeeplabV3 support output stride 8 or 16, got {}.".format(
                    encoder_output_stride
                )
            )
        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
            output_stride=encoder_output_stride,
            **kwargs,
        )

        if upsampling is None:
            if encoder_depth <= 3:
                scale_factor = 2**encoder_depth
            else:
                scale_factor = encoder_output_stride
        else:
            scale_factor = upsampling

        encoder_out_channels = [num * 2 for num in self.encoder.out_channels]

        self.decoder = DeepLabV3Decoder(
            in_channels=encoder_out_channels[-1],
            out_channels=decoder_channels,
            atrous_rates=decoder_atrous_rates,
            aspp_separable=decoder_aspp_separable,
            aspp_dropout=decoder_aspp_dropout,
        )

        mask_encoder_out_channels = [num * 3 for num in self.encoder.out_channels]

        self.refiner = DeepLabV3Decoder(
            in_channels=mask_encoder_out_channels[-1],
            out_channels=decoder_channels,
            atrous_rates=decoder_atrous_rates,
            aspp_separable=decoder_aspp_separable,
            aspp_dropout=decoder_aspp_dropout,
        )

        self.mask_head = SegmentationHead(
            in_channels=decoder_channels,
            out_channels=classes,
            activation=activation,
            kernel_size=1,
            upsampling=scale_factor,
        )
        self.mask_feature_head = SegmentationHead(
            in_channels=decoder_channels,
            out_channels=3,
            activation=activation,
            kernel_size=1,
            upsampling=scale_factor,
        )
        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels,
            out_channels=classes,
            activation=activation,
            kernel_size=1,
            upsampling=scale_factor,
        )

        if aux_params is not None:
            self.classification_head = ClassificationHead(
                in_channels=encoder_out_channels[-1], **aux_params
            )
        else:
            self.classification_head = None
        self.name = "deeplabv3-{}".format(encoder_name)

    def load_state_dict(self, state_dict, *args, **kwargs):
        # For backward compatibility, previously Decoder module was Sequential
        # and was not scriptable.
        keys = list(state_dict.keys())
        for key in keys:
            new_key = key
            if key.startswith("decoder.0."):
                new_key = key.replace("decoder.0.", "decoder.aspp.")
            elif key.startswith("decoder.1."):
                new_key = key.replace("decoder.1.", "decoder.conv.")
            elif key.startswith("decoder.2."):
                new_key = key.replace("decoder.2.", "decoder.bn.")
            state_dict[new_key] = state_dict.pop(key)
        return super().load_state_dict(state_dict, *args, **kwargs)


class DeepLabV3PlusRefiner(RefinerModel):
    """DeepLabV3+ implementation from "Encoder-Decoder with Atrous Separable
    Convolution for Semantic Image Segmentation"

    Args:
        encoder_name: Name of the classification model that will be used as an encoder (a.k.a backbone)
            to extract features of different spatial resolution
        encoder_depth: A number of stages used in encoder in range [3, 5]. Each stage generate features
            two times smaller in spatial dimensions than previous one (e.g. for depth 0 we will have features
            with shapes [(N, C, H, W),], for depth 1 - [(N, C, H, W), (N, C, H // 2, W // 2)] and so on).
            Default is 5
        encoder_weights: One of **None** (random initialization), **"imagenet"** (pre-training on ImageNet) and
            other pretrained weights (see table with available weights for each encoder_name)
        encoder_output_stride: Downsampling factor for last encoder features (see original paper for explanation)
        decoder_atrous_rates: Dilation rates for ASPP module (should be an iterable of 3 integer values)
        decoder_aspp_separable: Use separable convolutions in ASPP module. Default is True
        decoder_aspp_dropout: Use dropout in ASPP module projection layer. Default is 0.5
        decoder_channels: A number of convolution filters in ASPP module. Default is 256
        in_channels: A number of input channels for the model, default is 3 (RGB images)
        classes: A number of classes for output mask (or you can think as a number of channels of output mask)
        activation: An activation function to apply after the final convolution layer.
            Available options are **"sigmoid"**, **"softmax"**, **"logsoftmax"**, **"tanh"**, **"identity"**,
            **callable** and **None**. Default is **None**.
        upsampling: Final upsampling factor. Default is 4 to preserve input-output spatial shape identity.
        aux_params: Dictionary with parameters of the auxiliary output (classification head). Auxiliary output is build
            on top of encoder if **aux_params** is not **None** (default). Supported params:
                - classes (int): A number of classes
                - pooling (str): One of "max", "avg". Default is "avg"
                - dropout (float): Dropout factor in [0, 1)
                - activation (str): An activation function to apply "sigmoid"/"softmax"
                    (could be **None** to return logits)
        kwargs: Arguments passed to the encoder class ``__init__()`` function. Applies only to ``timm`` models.
            Keys with ``None`` values are pruned before passing.

    Returns:
        ``torch.nn.Module``: **DeepLabV3Plus**

    Reference:
        https://arxiv.org/abs/1802.02611v3

    """

    @supports_config_loading
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: Literal[3, 4, 5] = 5,
        encoder_weights: Optional[str] = "imagenet",
        encoder_output_stride: Literal[8, 16] = 16,
        decoder_channels: int = 256,
        decoder_atrous_rates: Iterable[int] = (12, 24, 36),
        decoder_aspp_separable: bool = True,
        decoder_aspp_dropout: float = 0.5,
        mid_mask_encoder_name: str = "resnet34",
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[str] = None,
        upsampling: int = 4,
        aux_params: Optional[dict] = None,
        **kwargs: dict[str, Any],
    ):
        super().__init__()

        if encoder_output_stride not in [8, 16]:
            raise ValueError(
                "DeeplabV3Plus support output stride 8 or 16, got {}.".format(
                    encoder_output_stride
                )
            )

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
            output_stride=encoder_output_stride,
            **kwargs,
        )

        encoder_out_channels = [num * 2 for num in self.encoder.out_channels]

        self.decoder = DeepLabV3PlusDecoder(
            encoder_channels=encoder_out_channels,
            encoder_depth=encoder_depth,
            out_channels=decoder_channels,
            atrous_rates=decoder_atrous_rates,
            output_stride=encoder_output_stride,
            aspp_separable=decoder_aspp_separable,
            aspp_dropout=decoder_aspp_dropout,
        )

        self.mid_mask_encoder = get_encoder(
            mid_mask_encoder_name,
            in_channels=1,
            depth=encoder_depth,
            weights=encoder_weights,
            output_stride=encoder_output_stride,
            **kwargs,
        )

        mask_encoder_out_channels = [self.encoder.out_channels[i] * 2 + self.mid_mask_encoder.out_channels[i] for i in range(len(self.encoder.out_channels))]

        self.refiner = DeepLabV3PlusDecoder(
            encoder_channels=mask_encoder_out_channels,
            encoder_depth=encoder_depth,
            out_channels=decoder_channels,
            atrous_rates=decoder_atrous_rates,
            output_stride=encoder_output_stride,
            aspp_separable=decoder_aspp_separable,
            aspp_dropout=decoder_aspp_dropout,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels,
            out_channels=classes,
            activation=activation,
            kernel_size=1,
            upsampling=upsampling,
        )

        if aux_params is not None:
            self.classification_head = ClassificationHead(
                in_channels=encoder_out_channels[-1], **aux_params
            )
        else:
            self.classification_head = None
        self.name = "deeplabv3plus-{}".format(encoder_name)
