"""
EfficientNet V1 and V2 implementation reffered to
https://github.com/google/automl/blob/master/efficientnetv2
"""

"""
EfficientNet V1 and V2 model.
[1] Mingxing Tan, Quoc V. Le
    EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.
    ICML'19, https://arxiv.org/abs/1905.11946
[2] Mingxing Tan, Quoc V. Le
    EfficientNetV2: Smaller Models and Faster Training.
    https://arxiv.org/abs/2104.00298
"""
import copy
from typing import Optional, List
from matplotlib import scale

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import utils
from .utils import Config
from .config import *

from torchvision._internally_replaced_utils import load_state_dict_from_url


__all__ = [
    "EfficientNet",
    "efficientnet_b0",
    "efficientnet_b1",
    "efficientnet_b2",
    "efficientnet_b3",
    "efficientnet_b4",
    "efficientnet_b5",
    "efficientnet_b6",
    "efficientnet_b7",
    "efficientnetv2_s",
    "efficientnetv2_m",
    "efficientnetv2_l",
    "efficientnetv2_xl",
]


model_urls = {
    # Weights ported from https://github.com/rwightman/pytorch-image-models/
    "efficientnet_b0": "https://download.pytorch.org/models/efficientnet_b0_rwightman-3dd342df.pth",
    "efficientnet_b1": "https://download.pytorch.org/models/efficientnet_b1_rwightman-533bc792.pth",
    "efficientnet_b2": "https://download.pytorch.org/models/efficientnet_b2_rwightman-bcdf34b7.pth",
    "efficientnet_b3": "https://download.pytorch.org/models/efficientnet_b3_rwightman-cf984f9c.pth",
    "efficientnet_b4": "https://download.pytorch.org/models/efficientnet_b4_rwightman-7eb33cd5.pth",
    # Weights ported from https://github.com/lukemelas/EfficientNet-PyTorch/
    "efficientnet_b5": "https://download.pytorch.org/models/efficientnet_b5_lukemelas-b6417697.pth",
    "efficientnet_b6": "https://download.pytorch.org/models/efficientnet_b6_lukemelas-c76e70fd.pth",
    "efficientnet_b7": "https://download.pytorch.org/models/efficientnet_b7_lukemelas-dcc49843.pth",
}


class SELayer(nn.Module):
    """Squeeze-and-excitation layer."""

    def __init__(self, mconfig, in_channels, se_filters, output_filters):
        super(SELayer, self).__init__()

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # Squeeze and Excitation layer.
        self.fc1 = nn.Conv2d(
            in_channels,
            se_filters,
            kernel_size=1,
            stride=1,
            padding='same',
            bias=True)
        self.fc2 = nn.Conv2d(
            se_filters,
            output_filters,
            kernel_size=1,
            stride=1,
            padding='same',
            bias=True)

        self.activation = utils.get_act_fn(mconfig.act_fn)(se_filters)
        self.scale_activation = nn.Sigmoid()

    def _scale(self, inputs):
        scale = self.avgpool(inputs)
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        scale = self.scale_activation(scale)
        return scale

    def forward(self, inputs):
        scale = self._scale(inputs)
        return scale * inputs


class MBConvBlock(nn.Module):
    """A class of MBConv: Mobile Inverted Residual Bottleneck."""

    def __init__(self, block_args, mconfig):
        """Initializes a MBConv block.

        Args:
            block_args: BlockArgs, arguments to create a Block.
            mconfig: GlobalParams, a set of global parameters.
        """
        super(MBConvBlock, self).__init__()

        self._block_args = copy.deepcopy(block_args)
        self._mconfig = copy.deepcopy(mconfig)

        self._has_se = (
            self._block_args.se_ratio is not None and
            0 < self._block_args.se_ratio <= 1)

        # Builds the block accordings to arguments.
        self._build()

    def _build(self):
        """Builds block according to the arguments."""
        mconfig = self._mconfig
        block_args = self._block_args
        input_filters = block_args.input_filters
        filters = input_filters * block_args.expand_ratio
        kernel_size = block_args.kernel_size

        layers: List[nn.Module] = []

        # Expansion phase. Called if not using fused convolutions and expansion
        # phase is necessary.
        if block_args.expand_ratio != 1:
            layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        input_filters,
                        filters,
                        kernel_size=1,
                        stride=1,
                        padding='same',
                        bias=False),
                    utils.normalization(
                        mconfig.bn_type,
                        filters,
                        eps=mconfig.bn_eps,
                        momentum=mconfig.bn_momentum,
                        groups=mconfig.gn_groups),
                    utils.get_act_fn(mconfig.act_fn)(filters)
                )
            )

        # Depth-wise convolution phase. Called if not using fused convolutions.
        layers.append(
            nn.Sequential(
                nn.Conv2d(
                    filters,
                    filters,
                    kernel_size=kernel_size,
                    stride=block_args.strides,
                    padding=kernel_size // 2,
                    groups=filters,
                    bias=False),
                utils.normalization(
                    mconfig.bn_type,
                    filters,
                    eps=mconfig.bn_eps,
                    momentum=mconfig.bn_momentum,
                    groups=mconfig.gn_groups),
                utils.get_act_fn(mconfig.act_fn)(filters)
            )
        )

        if mconfig.conv_dropout and block_args.expand_ratio > 1:
            layers.append(nn.Dropout(mconfig.conv_dropout))

        if self._has_se:
            num_reduced_filters = max(
                1, int(input_filters * block_args.se_ratio))
            layers.append(
                SELayer(mconfig, filters, num_reduced_filters, filters))

        # Output phase.
        layers.append(
            nn.Sequential(
                nn.Conv2d(
                    filters,
                    block_args.output_filters,
                    kernel_size=1,
                    stride=1,
                    padding='same',
                    bias=False),
                utils.normalization(
                    mconfig.bn_type,
                    block_args.output_filters,
                    eps=mconfig.bn_eps,
                    momentum=mconfig.bn_momentum,
                    groups=mconfig.gn_groups)
            )
        )

        self.block = nn.Sequential(*layers)

    def residual(self, inputs, x, survival_prob):
        if (self._block_args.strides == 1 and
            self._block_args.input_filters == self._block_args.output_filters):
            # Apply only if skip connection presents.
            if survival_prob:
                x = utils.drop_connect(x, self.training, survival_prob)
            x += inputs

        return x

    def forward(self, inputs, survival_prob=None):
        """Implementation of forward().

        Args:
            inputs: the inputs tensor.
            survival_prob: float, between 0 to 1, drop connect rate.

        Return:
            A output tensor.
        """
        x = self.block(inputs)
        x = self.residual(inputs, x, survival_prob)
        return x


class FusedMBConvBlock(MBConvBlock):
    """Fusing the proj conv1x1 and depthwise_conv into a conv2d."""

    def _build(self):
        """Builds block according to the arguments."""
        mconfig = self._mconfig
        block_args = self._block_args
        input_filters = block_args.input_filters
        filters = input_filters * block_args.expand_ratio
        kernel_size = block_args.kernel_size

        layers: List[nn.Module] = []

        if block_args.expand_ratio != 1:
            # Expansion phase:
            layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        input_filters,
                        filters,
                        kernel_size=kernel_size,
                        stride=block_args.strides,
                        padding=kernel_size // 2,
                        bias=False),
                    utils.normalization(
                        mconfig.bn_type,
                        filters,
                        eps=mconfig.bn_eps,
                        momentum=mconfig.bn_momentum,
                        groups=mconfig.gn_groups),
                    utils.get_act_fn(mconfig.act_fn)(filters)
                )
            )

        if mconfig.conv_dropout and block_args.expand_ratio > 1:
            layers.append(nn.Dropout(mconfig.conv_dropout))

        if self._has_se:
            num_reduced_filters = max(
                1, int(input_filters * block_args.se_ratio))
            layers.append(
                SELayer(mconfig, filters, num_reduced_filters, filters))

        # Output phase:
        stage: List[nn.Module] = [
            nn.Conv2d(
                filters,
                block_args.output_filters,
                kernel_size=1 if block_args.expand_ratio != 1 else kernel_size,
                stride=1 if block_args.expand_ratio != 1 else block_args.strides,
                padding='same' if block_args.expand_ratio != 1 else kernel_size // 2,
                bias=False),
            utils.normalization(
                mconfig.bn_type,
                block_args.output_filters,
                eps=mconfig.bn_eps,
                momentum=mconfig.bn_momentum,
                groups=mconfig.gn_groups)
        ]
        if self._block_args.expand_ratio == 1:
            stage.append(utils.get_act_fn(mconfig.act_fn)(block_args.output_filters))

        layers.append(nn.Sequential(*stage))
        self.block = nn.Sequential(*layers)

    def forward(self, inputs, survival_prob=None):
        """Implementation of forward().

        Args:
            inputs: the inputs tensor.
            survival_prob: float, between 0 to 1, drop connect rate.

        Return:
            A output tensor.
        """
        x = self.block(inputs)
        x = self.residual(inputs, x, survival_prob)
        return x


class Stem(nn.Sequential):
    """Stem layer at the begining of the network."""

    def __init__(self, mconfig, in_channels, stem_filters):
        out_channels = utils.round_filters(stem_filters, mconfig)
        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False),
            utils.normalization(
                mconfig.bn_type,
                out_channels,
                eps=mconfig.bn_eps,
                momentum=mconfig.bn_momentum,
                groups=mconfig.gn_groups),
            utils.get_act_fn(mconfig.act_fn)(out_channels)
        ]
        super().__init__(*layers)


class Head(nn.Sequential):
    """Head layer for network outputs."""

    def __init__(self, mconfig, in_channels):
        out_channels = utils.round_filters(mconfig.feature_size or 1280, mconfig)
        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding='same',
                bias=False),
            utils.normalization(
                mconfig.bn_type,
                out_channels,
                eps=mconfig.bn_eps,
                momentum=mconfig.bn_momentum,
                groups=mconfig.gn_groups),
            utils.get_act_fn(mconfig.act_fn)(out_channels)
        ]
        super().__init__(*layers)


class EfficientNet(nn.Module):
    """A class implements torch.nn.Module.

        Reference: https://arxiv.org/abs/1807.11626
    """

    def __init__(self,
                 model_config=None,
                 include_top=True,
                 image_channels=3):
        """Initializes an instance.

        Args:
            model_config: A dict of model configurations or a string of hparams.
            include_top: If True, include the top layer for classification.
            image_channels: Number of image channels (default: 3)

        Raises:
            ValueError: when blocks_args is not specified as a list.
        """
        super(EfficientNet, self).__init__()
        cfg = copy.deepcopy(base_config)
        cfg.model.override(model_config)
        self.cfg = cfg
        self._mconfig = cfg.model
        self.include_top = include_top
        self.image_channels = image_channels
        self._build()

    def _build(self):
        """Builds a model."""
        layers: List[nn.Module] = []

        # Stem part.
        layers.append(Stem(self._mconfig, self.image_channels, self._mconfig.blocks_args[0].input_filters))

        # Builds blocks.
        for block_args in self._mconfig.blocks_args:
            stage: List[nn.Module] = []
            assert block_args.num_repeat > 0
            # Update block input and output filters based on depth multiplier.
            input_filters = utils.round_filters(block_args.input_filters, self._mconfig)
            output_filters = utils.round_filters(block_args.output_filters, self._mconfig)
            repeats = utils.round_repeats(block_args.num_repeat,
                                    self._mconfig.depth_coefficient)
            block_args.update(
                dict(
                  input_filters=input_filters,
                  output_filters=output_filters,
                  num_repeat=repeats))

            # The first block needs to take care of stride and filter size increase.
            conv_block = {0: MBConvBlock, 1: FusedMBConvBlock}[block_args.conv_type]
            stage.append(
                conv_block(block_args, self._mconfig))
            if block_args.num_repeat > 1:  # rest of blocks with the same block_arg
                block_args.input_filters = block_args.output_filters
                block_args.strides = 1
            for _ in range(block_args.num_repeat - 1):
                stage.append(
                    conv_block(block_args, self._mconfig))
            layers.append(nn.Sequential(*stage))

        # Head part.
        input_filters = self._mconfig.blocks_args[-1].output_filters
        layers.append(Head(self._mconfig, input_filters))

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # top part for classification
        if self.include_top and self._mconfig.num_classes:
            in_channels = utils.round_filters(self._mconfig.feature_size or 1280, self._mconfig)
            self.classifier = nn.Sequential(
                nn.Dropout(self._mconfig.dropout_rate),
                nn.Linear(in_channels, self._mconfig.num_classes)
            )
        else:
            self.classifier = None

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                init_range = 1.0 / np.sqrt(m.out_features)
                nn.init.uniform_(m.weight, -init_range, init_range)
                nn.init.constant_(m.bias, self._mconfig.headbias or 0)

    def forward(self, inputs):
        """Implementation of forward().

        Args:
            inputs: input tensors.

        Returns:
            output tensors.
        """
        outputs = self.features(inputs)

        outputs = self.avgpool(outputs)
        outputs = torch.flatten(outputs, 1)

        outputs = self.classifier(outputs)

        return outputs


def _efficientnet(
    arch: str,
    width_mult: float,
    depth_mult: float,
    isize: int,
    dropout: float,
    pretrained: bool,
    progress: bool,
    option: Optional[dict] = None
) -> EfficientNet:
    cfg = Config(
        model=dict(
          model_name=arch,
          blocks_args=BlockDecoder().decode(v1_block_cfg),
          width_coefficient=width_mult,
          depth_coefficient=depth_mult,
          dropout_rate=dropout),
        eval=dict(isize=isize),
        train=dict(isize=0.8),  # 80% of eval size
        data=dict(augname='effnetv1_autoaug'))
    if option:
        cfg.override(option, allow_new_keys=True)
    model = EfficientNet(cfg.model)
    if pretrained:
        if model_urls.get(arch, None) is None:
            raise ValueError(f"No checkpoint is available for model type {arch}")
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model


def efficientnet_b0(pretrained: bool = False, progress: bool = True) -> EfficientNet:
    """
    Constructs a EfficientNet B0 architecture from
    `"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" <https://arxiv.org/abs/1905.11946>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _efficientnet("efficientnet_b0", 1.0, 1.0, 224, 0.2, pretrained, progress)


def efficientnet_b1(pretrained: bool = False, progress: bool = True) -> EfficientNet:
    """
    Constructs a EfficientNet B1 architecture from
    `"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" <https://arxiv.org/abs/1905.11946>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _efficientnet("efficientnet_b1", 1.0, 1.1, 240, 0.2, pretrained, progress)


def efficientnet_b2(pretrained: bool = False, progress: bool = True) -> EfficientNet:
    """
    Constructs a EfficientNet B2 architecture from
    `"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" <https://arxiv.org/abs/1905.11946>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _efficientnet("efficientnet_b2", 1.1, 1.2, 160, 0.3, pretrained, progress)


def efficientnet_b3(pretrained: bool = False, progress: bool = True) -> EfficientNet:
    """
    Constructs a EfficientNet B3 architecture from
    `"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" <https://arxiv.org/abs/1905.11946>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _efficientnet("efficientnet_b3", 1.2, 1.4, 300, 0.3, pretrained, progress)


def efficientnet_b4(pretrained: bool = False, progress: bool = True) -> EfficientNet:
    """
    Constructs a EfficientNet B4 architecture from
    `"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" <https://arxiv.org/abs/1905.11946>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _efficientnet("efficientnet_b4", 1.4, 1.8, 380, 0.4, pretrained, progress)


def efficientnet_b5(pretrained: bool = False, progress: bool = True) -> EfficientNet:
    """
    Constructs a EfficientNet B5 architecture from
    `"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" <https://arxiv.org/abs/1905.11946>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    base_config.override()
    return _efficientnet(
        "efficientnet_b5",
        1.6,
        2.2,
        456,
        0.4,
        pretrained,
        progress,
        option={'bn_eps': 0.001, 'bn_momentum': 0.01}
    )


def efficientnet_b6(pretrained: bool = False, progress: bool = True) -> EfficientNet:
    """
    Constructs a EfficientNet B6 architecture from
    `"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" <https://arxiv.org/abs/1905.11946>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _efficientnet(
        "efficientnet_b6",
        1.8,
        2.6,
        528,
        0.5,
        pretrained,
        progress,
        option={'bn_eps': 0.001, 'bn_momentum':0.01}
    )


def efficientnet_b7(pretrained: bool = False, progress: bool = True) -> EfficientNet:
    """
    Constructs a EfficientNet B7 architecture from
    `"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" <https://arxiv.org/abs/1905.11946>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _efficientnet(
        "efficientnet_b7",
        2.0,
        3.1,
        600,
        0.5,
        pretrained,
        progress,
        option={'bn_eps': 0.001, 'bn_momentum': 0.01}
    )


def _efficientnetv2(
    arch: str,
    cfg: Config,
    pretrained: bool,
    progress: bool,
) -> EfficientNet:
    model = EfficientNet(cfg.model)
    if pretrained:
        if model_urls.get(arch, None) is None:
            raise ValueError(f"No checkpoint is available for model type {arch}")
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model


def efficientnetv2_s(pretrained: bool = False, progress: bool = True):
    cfg = Config(
        model=dict(
            model_name='efficientnetv2_s',
            blocks_args=BlockDecoder().decode(v2_s_block),
            width_coefficient=1.0,
            depth_coefficient=1.0,
            dropout_rate=0.2,
        ),
        train=dict(isize=300, stages=4, sched=True),
        eval=dict(isize=384),
        data=dict(augname='randaug', ram=10, mixup_alpha=0, cutmix_alpha=0),
    )
    return _efficientnetv2('efficientnetv2_s', cfg, pretrained, progress)


def efficientnetv2_m(pretrained: bool = False, progress: bool = True):
    cfg = Config(
        model=dict(
            model_name='efficientnetv2_m',
            blocks_args=BlockDecoder().decode(v2_m_block),
            width_coefficient=1.0,
            depth_coefficient=1.0,
            dropout_rate=0.3,
        ),
        train=dict(isize=384, stages=4, sched=True),
        eval=dict(isize=480),
        data=dict(augname='randaug', ram=15, mixup_alpha=0.2, cutmix_alpha=0.2),
    )
    return _efficientnetv2('efficientnetv2_m', cfg, pretrained, progress)


def efficientnetv2_l(pretrained: bool = False, progress: bool = True):
    cfg = Config(
        model=dict(
            model_name='efficientnetv2_l',
            blocks_args=BlockDecoder().decode(v2_l_block),
            width_coefficient=1.0,
            depth_coefficient=1.0,
            dropout_rate=0.4,
        ),
        train=dict(isize=384, stages=4, sched=True),
        eval=dict(isize=480),
        data=dict(augname='randaug', ram=20, mixup_alpha=0.5, cutmix_alpha=0.5),
    )
    return _efficientnetv2('efficientnetv2_l', cfg, pretrained, progress)


def efficientnetv2_xl(pretrained: bool = False, progress: bool = True):
    cfg = Config(
        model=dict(
            model_name='efficientnetv2_xl',
            blocks_args=BlockDecoder().decode(v2_xl_block),
            width_coefficient=1.0,
            depth_coefficient=1.0,
            dropout_rate=0.4,
        ),
        train=dict(isize=384, stages=4, sched=True),
        eval=dict(isize=512),
        data=dict(augname='randaug', ram=20, mixup_alpha=0.5, cutmix_alpha=0.5),
    )
    return _efficientnetv2('efficientnetv2_xl', cfg, pretrained, progress)


