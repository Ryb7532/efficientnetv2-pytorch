"""Utilities."""
import functools
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import ast
import collections
import copy
from typing import Any, Dict, Text
import yaml



def actication_fn(features, act_fn):
    """Customized non-linear activation type."""
    if act_fn in ('silu', 'swish'):
        return nn.SiLU(features)
    elif act_fn == 'silu_native':
        return features * torch.sigmoid(features)
    elif act_fn == 'hswish':
        return F.hardswish(features)
    elif act_fn == 'relu':
        return F.relu(features)
    elif act_fn == 'relu6':
        return F.relu6(features)
    elif act_fn == 'elu':
        return F.elu(features)
    elif act_fn == 'leaky_relu':
        return F.leaky_relu(features)
    elif act_fn == 'selu':
        return F.selu(features)
    elif act_fn == 'mish':
        return F.mish(features)
    else:
        raise ValueError('Unsupported act_fn {}'.format(act_fn))


def get_act_fn(act_fn):
    if not act_fn:
        return nn.SiLU
    if isinstance(act_fn, str):
        return functools.partial(actication_fn, act_fn=act_fn)
    return act_fn


def normalization(norm_type,
                  in_channels,
                  eps=0.001, momentum=0.99, groups=8):
    """Normalization after conv layers."""
    if norm_type == 'gn':
        return nn.GroupNorm(groups, in_channels, eps=eps)

    if norm_type == 'gpu_bn':
        return nn.SyncBatchNorm(in_channels, eps=eps, momentum=momentum)

    return nn.BatchNorm2d(in_channels, eps=eps, momentum=momentum)


def drop_connect(inputs, is_training, survival_prob):
    """Drop the entire conv with given survival probability."""
    # "Deep Networks with Stochastic Depth", https://arxiv.org/pdf/1603.09382.pdf
    if not is_training:
        return inputs

    # Compute tensor.
    batch_size = inputs.shape[0]
    random_tensor = survival_prob
    random_tensor += torch.rand([batch_size, 1, 1, 1], dtype=inputs.dtype)
    binary_tensor = torch.floor(random_tensor)
    # Unlike conventional way that multiply survival_prob at test time, here we
    # divide survival_prob at training time, such that no addition compute is
    # needed at test time.
    output = inputs / survival_prob * binary_tensor
    return output


def eval_str_fn(val):
    if '|' in val:
        return [eval_str_fn(v) for v in val.split('|')]
    if val in {'true', 'false'}:
        return val == 'true'
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return val


class Config(dict):
    """A config utility class."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        input_config_dict = dict(*args, **kwargs)
        self.update(input_config_dict)

    def __len__(self):
        return len(self.__dict__)

    def __setattr__(self, k, v):
        if isinstance(v, dict) and not isinstance(v, Config):
            self.__dict__[k] = Config(v)
        else:
            self.__dict__[k] = copy.deepcopy(v)

    def __getattr__(self, k):
        return self.__dict__[k]

    def __setitem__(self, k, v):
        self.__setattr__(k, v)

    def __getitem__(self, k):
        return self.__dict__[k]

    def __iter__(self):
        for key in self.__dict__:
            yield key

    def items(self):
        for key, value in self.__dict__.items():
            yield key, value

    def __repr__(self):
        return repr(self.as_dict())

    def __getstate__(self):
        return self.__dict__

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        for k, v in self.__dict__.items():
            result[k] = v
        return result

    def __str__(self):
        try:
            return yaml.dump(self.as_dict(), indent=4)
        except TypeError:
            return str(self.as_dict())

    def _update(self, config_dict, allow_new_keys=True):
        """Recursively update internal members."""
        if not config_dict:
            return

        for k, v in config_dict.items():
            if k not in self.__dict__:
                if allow_new_keys:
                    self.__setattr__(k, v)
                else:
                    raise KeyError('Key `{}` does not exist for overriding. '.format(k))
            else:
                if isinstance(self.__dict__[k], Config) and isinstance(v, dict):
                    self.__dict__[k]._update(v, allow_new_keys)
                elif isinstance(self.__dict__[k], Config) and isinstance(v, Config):
                    self.__dict__[k]._update(v.as_dict(), allow_new_keys)
                else:
                    self.__setattr__(k, v)

    def get(self, k, default_value=None):
        return self.__dict__.get(k, default_value)

    def update(self, config_dict):
        """Update members while allowing new keys."""
        self._update(config_dict, allow_new_keys=True)

    def keys(self):
        return self.__dict__.keys()

    def override(self, config_dict_or_str, allow_new_keys=False):
        """Update members while disallowing new keys."""
        if not config_dict_or_str:
            return
        if isinstance(config_dict_or_str, str):
            if '=' in config_dict_or_str:
                config_dict = self.parse_from_str(config_dict_or_str)
            elif config_dict_or_str.endswith('.yaml'):
                config_dict = self.parse_from_yaml(config_dict_or_str)
            else:
                raise ValueError(
                        'Invalid string {}, must end with .yaml or contains "=".'.format(
                                config_dict_or_str))
        elif isinstance(config_dict_or_str, dict):
            config_dict = config_dict_or_str
        else:
            raise ValueError('Unknown value type: {}'.format(config_dict_or_str))

        self._update(config_dict, allow_new_keys)

    def parse_from_yaml(self, yaml_file_path: Text) -> Dict[Any, Any]:
        """Parses a yaml file and returns a dictionary."""
        with open(yaml_file_path, 'r') as f:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)
            return config_dict

    def save_to_yaml(self, yaml_file_path):
        """Write a dictionary into a yaml file."""
        with open(yaml_file_path, 'w') as f:
            yaml.dump(self.as_dict(), f, default_flow_style=False)

    def parse_from_str(self, config_str: Text) -> Dict[Any, Any]:
        """Parse a string like 'x.y=1,x.z=2' to nested dict {x: {y: 1, z: 2}}."""
        if not config_str:
            return {}
        config_dict = {}
        try:
            for kv_pair in config_str.split('.'):
                if not kv_pair:
                    continue
                key_str, value_str = kv_pair.split('=')
                key_str = key_str.strip()

                def add_kv_recursive(k, v):
                    """Recursively parse x.y.z=tt to {x: {y: {z: tt}}}."""
                    if '.' not in k:
                        return {k: eval_str_fn(v)}
                    pos = k.index('.')
                    return {k[:pos]: add_kv_recursive(k[pos + 1:], v)}

                def merge_dict_recursive(target, src):
                    """Recursively merge two nested dictionary."""
                    for k in src.keys():
                        if ((k in target and isinstance(target[k], dict) and
                                isinstance(src[k], collections.abc.Mapping))):
                            merge_dict_recursive(target[k], src[k])
                        else:
                            target[k] = src[k]

                merge_dict_recursive(config_dict, add_kv_recursive(key_str, value_str))
            return config_dict
        except ValueError:
            raise ValueError('Invalid config_str: {}'.format(config_str))

    def as_dict(self):
        """Returns a dict representation."""
        config_dict = {}
        for k, v in self.__dict__.items():
            if isinstance(v, Config):
                config_dict[k] = v.as_dict()
            elif isinstance(v, (list, tuple)):
                config_dict[k] = [
                        i.as_dict() if isinstance(i, Config) else copy.deepcopy(i)
                        for i in v
                ]
            else:
                config_dict[k] = copy.deepcopy(v)
        return config_dict


def conv_kernel_initialize(conv):
    """Initializes weight of the convolutional layer.

    Args:
        conv: nn.Module instance
    """
    nn.init.kaiming_normal_(conv.weight, mode='fan_out', nonlinearity='relu')


def linear_weight_initialize(linear, bias):
    """Initializes weight of the linear layer.

    Args:
        linear: nn.Module instance
    """
    init_range = 1.0 / np.sqrt(linear.weight.shape[0])
    nn.init.uniform_(linear.weight, -init_range, init_range)
    nn.init.constant_(linear.bias, bias)


def round_filters(filters, mconfig, skip=False):
    """Round number of filters based on depth multiplier."""
    multiplier = mconfig.width_coefficient
    divisor = mconfig.depth_divisor
    min_depth = mconfig.min_depth
    if skip or not multiplier:
        return filters

    filters *= multiplier
    min_depth = min_depth or divisor
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
    return int(new_filters)


def round_repeats(repeats, multiplier, skip=False):
    """Round number of filters based on depth multiplier."""
    if skip or not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))



