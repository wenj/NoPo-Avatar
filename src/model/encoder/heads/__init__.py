# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# head factory
# --------------------------------------------------------
from .dpt_gs_head import create_gs_dpt_head
from .linear_head import LinearPts3d
from .dpt_head import create_dpt_head
from .dpt_uv_head import create_uv_dpt_head
from .dpt_head_debug import create_dpt_head_debug
from .dpt_gs_head_debug import create_gs_dpt_head_debug


def head_factory(head_type, output_mode, net, has_conf=False, out_nchan=3, img_nchan=3, skip=False, upsample_type="deconv",
                 n_hooks=4):
    """" build a prediction head for the decoder
    """
    if head_type == 'linear' and output_mode == 'pts3d':
        return LinearPts3d(net, has_conf)
    elif head_type == 'dpt' and output_mode == 'pts3d':
        return create_dpt_head(net, has_conf=has_conf, out_nchan=out_nchan, img_nchan=img_nchan, skip=skip, n_hooks=n_hooks)
    elif head_type == 'dpt_gs' and output_mode == 'pts3d':
        return create_gs_dpt_head(net, has_conf=has_conf, out_nchan=out_nchan, img_nchan=img_nchan)
    elif head_type == 'dpt_debug' and output_mode == 'pts3d':
        return create_dpt_head_debug(net, has_conf=has_conf, out_nchan=out_nchan)
    elif head_type == 'dpt' and output_mode == 'uv':
        return create_uv_dpt_head(net, has_conf=has_conf, out_nchan=out_nchan)
    elif head_type == 'dpt' and output_mode == 'gs_params':
        return create_dpt_head(net, has_conf=False, out_nchan=out_nchan, postprocess_func=None)
    elif head_type == 'dpt_gs' and output_mode == 'gs_params':
        return create_gs_dpt_head(net, has_conf=False, out_nchan=out_nchan, postprocess_func=None, img_nchan=img_nchan, upsample_type=upsample_type, n_hooks=n_hooks)
    elif head_type == 'dpt_gs_debug' and output_mode == 'gs_params':
        return create_gs_dpt_head_debug(net, has_conf=False, out_nchan=out_nchan, postprocess_func=None, img_nchan=img_nchan)
    else:
        raise NotImplementedError(f"unexpected {head_type=} and {output_mode=}")
