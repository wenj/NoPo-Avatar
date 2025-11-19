# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# dpt head implementation for DUST3R
# Downstream heads assume inputs of size B x N x C (where N is the number of tokens) ;
# or if it takes as input the output at every layer, the attribute return_all_layers should be set to True
# the forward function also takes as input a dictionnary img_info with key "height" and "width"
# for PixelwiseTask, the output will be of dimension B x num_channels x H x W
# --------------------------------------------------------
from einops import rearrange
from typing import List, Literal
import torch
import torch.nn as nn
import torch.nn.functional as F
# import dust3r.utils.path_to_croco
from .dpt_block import DPTOutputAdapter
from .postprocess import postprocess


class UVLearner(nn.Module):
    def __init__(self, in_channels: int, attn_type: Literal["global", "local"], window_size: int = 5):
        super().__init__()
        self.attn_type = attn_type
        self.window_size = window_size
        self.scale = in_channels ** -0.5

        grids = self.get_grid(64, 64)
        self.register_buffer("grids", grids)

    @staticmethod
    def get_grid(h, w, device=torch.device("cuda")):
        grid = torch.meshgrid([torch.arange(h, device=device, dtype=torch.float32), torch.arange(w, device=device, dtype=torch.float32)], indexing="xy")
        grid = torch.stack(grid, dim=-1) + 0.5
        return grid

    def forward(self, x, y, uv0=None):
        # print(x.shape, y.shape, self.attn_type)
        b, c, hx, wx = x.shape
        _, _, hy, wy = y.shape

        if self.attn_type == "global":
            x_ = x.reshape(b, c, hx * wx)
            y_ = y.reshape(b, c, hy * wy)
            attn_weights = F.softmax(torch.matmul(x_.transpose(-2, -1), y_) * self.scale, dim=-1)
            uv = torch.matmul(attn_weights, self.grids[:hy, :wy].reshape(-1, 2))
            uv = rearrange(uv, "b (hx wx) c-> b hx wx c", hx=hx, wx=wx).contiguous()
            uv[..., 0] = uv[..., 0] / wy * 2. - 1.
            uv[..., 1] = uv[..., 1] / hy * 2. - 1.
            return uv

        offset_grids = self.grids[:self.window_size, :self.window_size].reshape(-1, 2) - self.window_size / 2.
        # import pdb; pdb.set_trace()
        offset_grids[..., 0] = offset_grids[..., 0] / wy * 2.
        offset_grids[..., 1] = offset_grids[..., 1] / hy * 2.
        uv_sample = uv0.reshape(b, hx * wx, 2).unsqueeze(-2) + offset_grids # b x (hx*wx) x n x 2
        y = torch.cat([y, torch.ones_like(y[:, :1])], dim=1)
        y_ = F.grid_sample(y, uv_sample, align_corners=True)
        y_, valid_ = y_[:, :-1], y_[:, -1] # b (hx*wx) x n

        x_ = rearrange(x, "b c hx wx -> (b hx wx) 1 c").contiguous()
        y_ = rearrange(y_, "b c hw n -> (b hw) c n").contiguous()
        # print(x_.shape, y_.shape)
        attn_weights = F.softmax(torch.matmul(x_, y_) * self.scale, dim=-1)
        attn_weights = attn_weights.masked_fill(valid_.reshape(b * hx * wx, 1, -1) < 1e-3, -float('inf'))
        # print(attn_weights.shape, offset_grids.shape)
        uv = torch.matmul(attn_weights, offset_grids).squeeze(-2)
        uv = rearrange(uv, "(b hx wx) c-> b hx wx c", hx=hx, wx=wx).contiguous()
        return uv


class DPTOutputAdapter_fix(DPTOutputAdapter):
    """
    Adapt croco's DPTOutputAdapter implementation for dust3r:
    remove duplicated weigths, and fix forward for dust3r
    """

    def init(self, dim_tokens_enc=768):
        super().init(dim_tokens_enc)
        # these are duplicated weights
        del self.act_1_postprocess
        del self.act_2_postprocess
        del self.act_3_postprocess
        del self.act_4_postprocess

        self.uv_learner_4 = UVLearner(self.feature_dim, attn_type="global")
        self.uv_learner_3 = UVLearner(self.feature_dim, attn_type="local", window_size=5)
        self.uv_learner_2 = UVLearner(self.feature_dim, attn_type="local", window_size=5)
        self.uv_learner_1 = UVLearner(self.feature_dim, attn_type="local", window_size=5)
        # self.uv_learner_out = UVLearner(self.num_channels, attn_type="local", window_size=5)

    def forward(self, encoder_tokens: List[torch.Tensor], feats_ref: List[torch.Tensor], image_size=None, ray_embedding=None):
        assert self.dim_tokens_enc is not None, 'Need to call init(dim_tokens_enc) function first'
        # H, W = input_info['image_size']
        image_size = self.image_size if image_size is None else image_size
        H, W = image_size
        # Number of patches in height and width
        N_H = H // (self.stride_level * self.P_H)
        N_W = W // (self.stride_level * self.P_W)

        # Hook decoder onto 4 layers from specified ViT layers
        layers = [encoder_tokens[hook] for hook in self.hooks]

        # Extract only task-relevant tokens and ignore global tokens.
        layers = [self.adapt_tokens(l) for l in layers]

        # Reshape tokens to spatial representation
        layers = [rearrange(l, 'b (nh nw) c -> b c nh nw', nh=N_H, nw=N_W).contiguous() for l in layers]

        layers = [self.act_postprocess[idx](l) for idx, l in enumerate(layers)]
        # Project layers to chosen feature dim
        layers = [self.scratch.layer_rn[idx](l) for idx, l in enumerate(layers)]

        # Fuse layers using refinement stages
        path_4 = self.scratch.refinenet4(layers[3])[:, :, :layers[2].shape[2], :layers[2].shape[3]]
        path_3 = self.scratch.refinenet3(path_4, layers[2])
        path_2 = self.scratch.refinenet2(path_3, layers[1])
        path_1 = self.scratch.refinenet1(path_2, layers[0])

        # if ray_embedding is not None:
        #     ray_embedding = F.interpolate(ray_embedding, size=(path_1.shape[2], path_1.shape[3]), mode='bilinear')
        #     path_1 = torch.cat([path_1, ray_embedding], dim=1)

        # Output head
        out = self.head(path_1)

        path_4_ref, path_3_ref, path_2_ref, path_1_ref, out_ref = feats_ref
        uv_4 = self.uv_learner_4(path_4, path_4_ref)
        uv_4 = F.interpolate(uv_4.permute(0, 3, 1, 2), scale_factor=2, mode='bilinear', align_corners=True).permute(0, 2, 3, 1)
        uv_3 = self.uv_learner_3(path_3, path_3_ref, uv_4) + uv_4
        uv_3 = F.interpolate(uv_3.permute(0, 3, 1, 2), scale_factor=2, mode='bilinear', align_corners=True).permute(0, 2, 3, 1)
        uv_2 = self.uv_learner_2(path_2, path_2_ref, uv_3) + uv_3
        uv_2 = F.interpolate(uv_2.permute(0, 3, 1, 2), scale_factor=2, mode='bilinear', align_corners=True).permute(0, 2, 3, 1)
        uv_1 = self.uv_learner_1(path_1, path_1_ref, uv_2) + uv_2
        uv_1 = F.interpolate(uv_1.permute(0, 3, 1, 2), scale_factor=2, mode='bilinear', align_corners=True).permute(0, 2, 3, 1)
        # uv = self.uv_learner_out(out, out_ref, uv_1) + uv_1
        # print(uv.shape)
        uv = uv_1
        # print(uv.shape)

        return uv


class PixelwiseTaskWithDPT(nn.Module):
    """ DPT module for dust3r, can return 3D points + confidence for all pixels"""

    def __init__(self, *, n_cls_token=0, hooks_idx=None, dim_tokens=None,
                 output_width_ratio=1, num_channels=1, postprocess=None, depth_mode=None, conf_mode=None, **kwargs):
        super(PixelwiseTaskWithDPT, self).__init__()
        self.return_all_layers = True  # backbone needs to return all layers
        self.postprocess = postprocess
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode

        assert n_cls_token == 0, "Not implemented"
        dpt_args = dict(output_width_ratio=output_width_ratio,
                        num_channels=num_channels,
                        **kwargs)
        if hooks_idx is not None:
            dpt_args.update(hooks=hooks_idx)
        self.dpt = DPTOutputAdapter_fix(**dpt_args)
        dpt_init_args = {} if dim_tokens is None else {'dim_tokens_enc': dim_tokens}
        self.dpt.init(**dpt_init_args)

    def forward(self, x, img_info, ray_embedding=None):
        out = self.dpt(x[1:], x[0], image_size=(img_info[0], img_info[1]), ray_embedding=ray_embedding)
        return out


def create_uv_dpt_head(net, has_conf=False, out_nchan=3, postprocess_func=postprocess):
    """
    return PixelwiseTaskWithDPT for given net params
    """
    assert net.dec_depth > 9
    l2 = net.dec_depth
    feature_dim = 256
    last_dim = feature_dim//2
    ed = net.enc_embed_dim
    dd = net.dec_embed_dim
    return PixelwiseTaskWithDPT(num_channels=out_nchan + has_conf,
                                feature_dim=feature_dim,
                                last_dim=last_dim,
                                hooks_idx=[0, l2*2//4, l2*3//4, l2],
                                dim_tokens=[ed, dd, dd, dd],
                                postprocess=postprocess_func,
                                depth_mode=net.depth_mode,
                                conf_mode=net.conf_mode,
                                head_type='regression')
