import re

from .utils import Config

base_config = Config(
        # model related params.
        model=dict(
                model_name='efficientnet_b0',
                feature_size=1280,
                bn_type=None,     # 'gpu_bn',
                bn_momentum=0.9,
                bn_eps=1e-3,
                gn_groups=8,
                depth_divisor=8,
                min_depth=8,
                act_fn='silu',
                survival_prob=0.8,
                local_pooling=False,
                headbias=None,
                conv_dropout=None,
                dropout_rate=None,
                depth_coefficient=None,
                width_coefficient=None,
                blocks_args=None,
                num_classes=1000,    # must be the same as data.num_classes
        ),
        # train related params.
        train=dict(
                stages=0,
                epochs=350,
                min_steps=0,
                optimizer='rmsprop',
                lr_sched='exponential',
                lr_base=0.016,
                lr_decay_epoch=2.4,
                lr_decay_factor=0.97,
                lr_warmup_epoch=5,
                lr_min=0,
                ema_decay=0.9999,
                weight_decay=1e-5,
                weight_decay_inc=0.0,
                weight_decay_exclude='.*(bias|gamma|beta).*',
                label_smoothing=0.1,
                gclip=0,
                batch_size=4096,
                isize=None,
                split=None,    # dataset split, default to 'train'
                loss_type=None,    # loss type: sigmoid or softmax
                ft_init_ckpt=None,
                ft_init_ema=True,
                varsexp=None,    # trainable variables.
                sched=None,    # schedule
        ),
        eval=dict(
                batch_size=8,
                isize=None,    # image size
                split=None,    # dataset split, default to 'eval'
        ),
        # data related params.
        data=dict(
                ds_name='imagenet',
                augname='randaug',    # or 'autoaug'
                ra_num_layers=2,
                ram=15,
                mixup_alpha=0.,
                cutmix_alpha=0.,
                ibase=128,
                cache=True,
                resize=None,
                data_dir=None,
                multiclass=None,
                num_classes=1000,
                tfds_name=None,
                try_gcs=False,
                tfds_split=None,
                splits=dict(
                        train=dict(
                                num_images=None, files=None, tfds_split=None, slice=None),
                        eval=dict(num_images=None, files=None, tfds_split=None, slice=None),
                        minival=dict(
                                num_images=None, files=None, tfds_split=None, slice=None),
                        trainval=dict(
                                num_images=None, files=None, tfds_split=None, slice=None),
                ),
        ),
        runtime=dict(
                iterations_per_loop=1000,    # larger value has better utilization.
                skip_host_call=False,
                mixed_precision=True,
                use_async_checkpointing=False,
                log_step_count_steps=64,
                keep_checkpoint_max=5,
                keep_checkpoint_every_n_hours=5,
                strategy='gpu',    # None, gpu, tpu
        ))


"""EfficientNet V1 and V2 model configs."""
class BlockDecoder(object):
    """Block Decoder for readability."""

    def _decode_block_str(self, block_str: str):
        """Gets a block through a string notation of arguments."""
        ops = block_str.split('_')
        options = {}
        for op in ops:
            splits = re.split(r'(\d.*)', op)
            if len(splits) >= 2:
                key, value = splits[:2]
                options[key] = value

        return Config(
            kernel_size=int(options['k']),
            num_repeat=int(options['r']),
            input_filters=int(options['i']),
            output_filters=int(options['o']),
            expand_ratio=int(options['e']),
            se_ratio=float(options['se']) if 'se' in options else None,
            strides=int(options['s']),
            conv_type=int(options['c']) if 'c' in options else 0)

    def decode(self, str_list: list):
        """Decodes a list of string notations to specify blocks inside the network.

        Args:
            str_list: a list of strings, each string is a notation of block.

        Returns:
            A list of namedtuples to represent blocks arguments.
        """
        blocks_args = []
        for block_str in str_list:
            blocks_args.append(self._decode_block_str(block_str))
        return blocks_args


#################### EfficientNet V1 configs ####################
v1_block_cfg = [
    'r1_k3_s1_e1_i32_o16_se0.25',
    'r2_k3_s2_e6_i16_o24_se0.25',
    'r2_k5_s2_e6_i24_o40_se0.25',
    'r3_k3_s2_e6_i40_o80_se0.25',
    'r3_k5_s1_e6_i80_o112_se0.25',
    'r4_k5_s2_e6_i112_o192_se0.25',
    'r1_k3_s1_e6_i192_o320_se0.25',
]


#################### EfficientNet V2 configs ####################
v2_s_block = [  # about base * (width1.4, depth1.8)
    'r2_k3_s1_e1_i24_o24_c1',
    'r4_k3_s2_e4_i24_o48_c1',
    'r4_k3_s2_e4_i48_o64_c1',
    'r6_k3_s2_e4_i64_o128_se0.25',
    'r9_k3_s1_e6_i128_o160_se0.25',
    'r15_k3_s2_e6_i160_o256_se0.25',
]
v2_m_block = [  # about base * (width1.6, depth2.2)
    'r3_k3_s1_e1_i24_o24_c1',
    'r5_k3_s2_e4_i24_o48_c1',
    'r5_k3_s2_e4_i48_o80_c1',
    'r7_k3_s2_e4_i80_o160_se0.25',
    'r14_k3_s1_e6_i160_o176_se0.25',
    'r18_k3_s2_e6_i176_o304_se0.25',
    'r5_k3_s1_e6_i304_o512_se0.25',
]
v2_l_block = [  # about base * (width2.0, depth3.1)
    'r4_k3_s1_e1_i32_o32_c1',
    'r7_k3_s2_e4_i32_o64_c1',
    'r7_k3_s2_e4_i64_o96_c1',
    'r10_k3_s2_e4_i96_o192_se0.25',
    'r19_k3_s1_e6_i192_o224_se0.25',
    'r25_k3_s2_e6_i224_o384_se0.25',
    'r7_k3_s1_e6_i384_o640_se0.25',
]
v2_xl_block = [  # only for 21k pretraining.
    'r4_k3_s1_e1_i32_o32_c1',
    'r8_k3_s2_e4_i32_o64_c1',
    'r8_k3_s2_e4_i64_o96_c1',
    'r16_k3_s2_e4_i96_o192_se0.25',
    'r24_k3_s1_e6_i192_o256_se0.25',
    'r32_k3_s2_e6_i256_o512_se0.25',
    'r8_k3_s1_e6_i512_o640_se0.25',
]



