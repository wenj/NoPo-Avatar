from copy import deepcopy
from dataclasses import dataclass
from typing import Literal, List

import torch
from einops import rearrange, repeat
from torch import nn
import torch.utils.checkpoint as checkpoint

from .croco.blocks import DecoderBlock
from .croco.croco import CroCoNet
from .croco.misc import fill_default_args, freeze_all_params, transpose_to_landscape, is_symmetrized, interleave, \
    make_batch_symmetric
from .croco.patch_embed import get_patch_embed
from .backbone import Backbone
from ....geometry.camera_emb import get_intrinsic_embedding

inf = float('inf')


croco_params = {
    'ViTLarge_BaseDecoder': {
        'enc_depth': 24,
        'dec_depth': 12,
        'enc_embed_dim': 1024,
        'dec_embed_dim': 768,
        'enc_num_heads': 16,
        'dec_num_heads': 12,
        'pos_embed': 'RoPE100',
        'img_size': (512, 512),
    },
}

default_dust3r_params = {
    'enc_depth': 24,
    'dec_depth': 12,
    'enc_embed_dim': 1024,
    'dec_embed_dim': 768,
    'enc_num_heads': 16,
    'dec_num_heads': 12,
    'pos_embed': 'RoPE100',
    'patch_embed_cls': 'PatchEmbedDust3R',
    'img_size': (512, 512),
    'head_type': 'dpt',
    'output_mode': 'pts3d',
    'depth_mode': ('exp', -inf, inf),
    'conf_mode': ('exp', 1, inf)
}


@dataclass
class BackboneCrocoMulti2Cfg:
    name: Literal["croco_multi2"]
    model: Literal["ViTLarge_BaseDecoder", "ViTBase_SmallDecoder", "ViTBase_BaseDecoder"]  # keep interface for the last two models, but they are not supported
    patch_embed_cls: str = 'PatchEmbedDust3R'  # PatchEmbedDust3R or ManyAR_PatchEmbed
    asymmetry_decoder: bool = True
    intrinsics_embed_loc: Literal["encoder", "decoder", "none"] = 'none'
    intrinsics_embed_degree: int = 0
    intrinsics_embed_type: Literal["pixelwise", "linear", "token"] = 'token'  # linear or dpt
    template_encoder_free: bool = False
    template_image_size: List[int] = 256, 256
    template_embed_dim: int = 1024
    disable_checkpointing: bool = False


class AsymmetricCroCoMulti2(CroCoNet):
    """ Two siamese encoders, followed by two decoders.
    The goal is to output 3d points directly, both images in view1's frame
    (hence the asymmetry).
    """

    def __init__(self, cfg: BackboneCrocoMulti2Cfg, d_in: int) -> None:

        self.intrinsics_embed_loc = cfg.intrinsics_embed_loc
        self.intrinsics_embed_degree = cfg.intrinsics_embed_degree
        self.intrinsics_embed_type = cfg.intrinsics_embed_type
        self.intrinsics_embed_encoder_dim = 0
        self.intrinsics_embed_decoder_dim = 0
        if self.intrinsics_embed_loc == 'encoder' and self.intrinsics_embed_type == 'pixelwise':
            self.intrinsics_embed_encoder_dim = (self.intrinsics_embed_degree + 1) ** 2 if self.intrinsics_embed_degree > 0 else 3
        elif self.intrinsics_embed_loc == 'decoder' and self.intrinsics_embed_type == 'pixelwise':
            self.intrinsics_embed_decoder_dim = (self.intrinsics_embed_degree + 1) ** 2 if self.intrinsics_embed_degree > 0 else 3

        self.patch_embed_cls = cfg.patch_embed_cls
        self.croco_args = fill_default_args(croco_params[cfg.model], CroCoNet.__init__)

        super().__init__(**croco_params[cfg.model])

        if cfg.asymmetry_decoder:
            self.dec_blocks2 = deepcopy(self.dec_blocks)  # This is used in DUSt3R and MASt3R

        if self.intrinsics_embed_type == 'linear' or self.intrinsics_embed_type == 'token':
            self.intrinsic_encoder = nn.Linear(9, 1024)

        self.template_encoder_free = cfg.template_encoder_free
        self.template_image_size = cfg.template_image_size
        if self.template_encoder_free:
            self.template_embed_dim = cfg.template_embed_dim

            if self.template_embed_dim != croco_params[cfg.model]['enc_embed_dim']:
                self.template_embedder = nn.Linear(self.template_embed_dim, croco_params[cfg.model]['enc_embed_dim'])
            else:
                self.template_embedder = nn.Identity()
            self.template_embed = nn.Parameter(torch.randn(
                (cfg.template_image_size[0] // 16) * (cfg.template_image_size[1] // 16)
                + (self.intrinsics_embed_loc == 'encoder' and self.intrinsics_embed_type == 'token'),
                self.template_embed_dim,
            ))

        # self.set_downstream_head(output_mode, head_type, landscape_only, depth_mode, conf_mode, **croco_kwargs)
        # self.set_freeze(freeze)
        self.disable_checkpointing = cfg.disable_checkpointing

    def _set_patch_embed(self, img_size=224, patch_size=16, enc_embed_dim=768, in_chans=3):
        in_chans = in_chans + self.intrinsics_embed_encoder_dim
        self.patch_embed = get_patch_embed(self.patch_embed_cls, img_size, patch_size, enc_embed_dim, in_chans)

        self.patch_embed_smpl = get_patch_embed(self.patch_embed_cls, img_size, patch_size, enc_embed_dim, 24 + 3 + self.intrinsics_embed_encoder_dim)
        self.patch_embed_smplx = get_patch_embed(self.patch_embed_cls, img_size, patch_size, enc_embed_dim, 55 + 3 + self.intrinsics_embed_encoder_dim)

    def _set_decoder(self, enc_embed_dim, dec_embed_dim, dec_num_heads, dec_depth, mlp_ratio, norm_layer, norm_im2_in_dec):
        self.dec_depth = dec_depth
        self.dec_embed_dim = dec_embed_dim
        # transfer from encoder to decoder
        enc_embed_dim = enc_embed_dim + self.intrinsics_embed_decoder_dim
        self.decoder_embed = nn.Linear(enc_embed_dim, dec_embed_dim, bias=True)
        # transformer for the decoder
        self.dec_blocks = nn.ModuleList([
            DecoderBlock(dec_embed_dim, dec_num_heads, mlp_ratio=mlp_ratio, qkv_bias=True, norm_layer=norm_layer, norm_mem=norm_im2_in_dec, rope=self.rope)
            for i in range(dec_depth)])
        # final norm layer
        self.dec_norm = norm_layer(dec_embed_dim)

    def load_state_dict(self, ckpt, **kw):
        # duplicate all weights for the second decoder if not present
        new_ckpt = dict(ckpt)
        if not any(k.startswith('dec_blocks2') for k in ckpt):
            for key, value in ckpt.items():
                if key.startswith('dec_blocks'):
                    new_ckpt[key.replace('dec_blocks', 'dec_blocks2')] = value
        return super().load_state_dict(new_ckpt, **kw)

    def set_freeze(self, freeze):  # this is for use by downstream models
        assert freeze in ['none', 'mask', 'encoder'], f"unexpected freeze={freeze}"
        to_be_frozen = {
            'none':     [],
            'mask':     [self.mask_token],
            'encoder':  [self.mask_token, self.patch_embed, self.enc_blocks],
            'encoder_decoder':  [self.mask_token, self.patch_embed, self.enc_blocks, self.enc_norm, self.decoder_embed, self.dec_blocks, self.dec_blocks2, self.dec_norm],
        }
        freeze_all_params(to_be_frozen[freeze])

    def _set_prediction_head(self, *args, **kwargs):
        """ No prediction head """
        return

    def _encode_image(self, template, template_shape, image, true_shape, intrinsics_embed=None):
        # embed the image into patches  (x has size B x Npatches x C)
        x, pos = self.patch_embed(image, true_shape=true_shape)
        if template.shape[1] == 24 + 3:
            template, pos_template = self.patch_embed_smpl(template, true_shape=template_shape)
        else:
            template, pos_template = self.patch_embed_smplx(template, true_shape=template_shape)

        if intrinsics_embed is not None:

            if self.intrinsics_embed_type == 'linear':
                x = x + intrinsics_embed
            elif self.intrinsics_embed_type == 'token':
                x = torch.cat((x, intrinsics_embed), dim=1)
                add_pose = pos[:, 0:1, :].clone()
                add_pose[:, :, 0] += (pos[:, -1, 0].unsqueeze(-1) + 1)
                pos = torch.cat((pos, add_pose), dim=1)

                if not self.template_encoder_free:
                    template = torch.cat((template, intrinsics_embed.new_zeros(template.shape[0], 1, *intrinsics_embed.shape[2:])), dim=1)
                add_pose = pos_template[:, 0:1, :].clone()
                add_pose[:, :, 0] += (pos_template[:, -1, 0].unsqueeze(-1) + 1)
                pos_template = torch.cat((pos_template, add_pose), dim=1)

        # add positional embedding without cls token
        assert self.enc_pos_embed is None

        is_same_shape = x.shape[1] == template.shape[1]
        # now apply the transformer encoder and normalization
        for blk in self.enc_blocks:
            if not self.template_encoder_free:
                if is_same_shape:
                    x, template = blk(torch.cat([x, template], dim=0), torch.cat([pos, pos_template], dim=0)).split([x.shape[0], template.shape[0]], dim=0)
                else:
                    x = blk(x, pos)
                    template = blk(template, pos_template)
            else:
                x = blk(x, pos)

        x = self.enc_norm(x)
        if not self.template_encoder_free:
            template = self.enc_norm(template)
        else:
            template = repeat(self.template_embed, "l c -> b l c", b=template.shape[0])
            template = self.template_embedder(template)
        return template, pos_template, x, pos, None

    def _decoder(self, feat_template, pose_template, feat, pose, extra_embed=None):
        b, v, l, c = feat.shape
        final_output = [(feat_template, feat)]  # before projection
        if extra_embed is not None:
            feat = torch.cat((feat, extra_embed), dim=-1)

        # project to decoder dim
        f = rearrange(feat, "b v l c -> (b v) l c").contiguous()
        f = self.decoder_embed(f)
        f = rearrange(f, "(b v) l c -> b v l c", b=b, v=v).contiguous()
        f_template = self.decoder_embed(feat_template)
        final_output.append((f_template, f))

        def generate_ctx_views(x):
            b, v, l, c = x.shape
            ctx_views = x.unsqueeze(1).expand(b, v, v, l, c)
            mask = torch.arange(v).unsqueeze(0) != torch.arange(v).unsqueeze(1)
            ctx_views = ctx_views[:, mask].reshape(b, v, v - 1, l, c)  # B, V, V-1, L, C
            ctx_views = ctx_views.flatten(2, 3)  # B, V, (V-1)*L, C
            return ctx_views.contiguous()

        pos_ctx = generate_ctx_views(pose)
        for blk1, blk2 in zip(self.dec_blocks, self.dec_blocks2):
            feat_template_current, feat_current = final_output[-1]
            feat_current_ctx = generate_ctx_views(feat_current)
            # img1 side
            f1, _ = blk1(
                feat_template_current,
                rearrange(feat_current, "b v l c -> b (v l) c").contiguous(),
                pose_template,
                rearrange(pose, "b v l c -> b (v l) c").contiguous())
            # img2 side
            f2, _ = blk2(
                rearrange(feat_current, "b v l c -> (b v) l c").contiguous(),
                torch.cat([
                    repeat(feat_template_current, "b l c -> (b v) l c", v=v).contiguous(),
                    rearrange(feat_current_ctx, "b v l c -> (b v) l c").contiguous(),
                ], dim=1),
                rearrange(pose, "b v l c -> (b v) l c").contiguous(),
                torch.cat([
                    repeat(pose_template, "b l c -> (b v) l c", v=v).contiguous(),
                    rearrange(pos_ctx, "b v l c -> (b v) l c").contiguous()
                ], dim=1)
            )
            f2 = rearrange(f2, "(b v) l c -> b v l c", b=b, v=v).contiguous()
            # store the result
            final_output.append((f1, f2))

        # normalize last output
        del final_output[1]  # duplicate with final_output[0]
        f1, f2 = final_output[-1]
        last_feat_template = self.dec_norm(f1)
        last_feat = rearrange(f2, "b v l c -> (b v) l c").contiguous()
        last_feat = self.dec_norm(last_feat)
        final_output[-1] = (last_feat_template, rearrange(last_feat, "(b v) l c -> b v l c", b=b, v=v).contiguous())
        return [tok[0] for tok in final_output], [tok[1] for tok in final_output]

    def forward(self,
                context: dict,
                symmetrize_batch=False,
                return_views=False,
                return_pos=False,
                ):
        b, v, _, h, w = context["image"].shape
        images_all = context["image"]

        # camera embedding in the encoder
        if self.intrinsics_embed_loc == 'encoder' and self.intrinsics_embed_type == 'pixelwise':
            intrinsic_embedding = get_intrinsic_embedding(context, degree=self.intrinsics_embed_degree)
            images_all = torch.cat((images_all, intrinsic_embedding), dim=2)

        intrinsic_embedding_all = None
        if self.intrinsics_embed_loc == 'encoder' and (self.intrinsics_embed_type == 'token' or self.intrinsics_embed_type == 'linear'):
            intrinsic_embedding = self.intrinsic_encoder(context["intrinsics"].flatten(2))
            intrinsic_embedding_all = rearrange(intrinsic_embedding, "b v c -> (b v) c").contiguous().unsqueeze(1)

        # step 1: encoder input images
        images_all = rearrange(images_all, "b v c h w -> (b v) c h w").contiguous()
        shape_all = torch.tensor(images_all.shape[-2:])[None].repeat(b*v, 1)

        template = torch.cat([context["template_3d"], context["template_lbs_weights"]], dim=-1)
        template = rearrange(template, "b h w c -> b c h w").contiguous()
        shape_template = torch.tensor(template.shape[-2:])[None].repeat(b, 1)

        if self.disable_checkpointing:
            feat_template, pose_template, feat, pose, _ = self._encode_image(template, shape_template, images_all, shape_all, intrinsic_embedding_all)
        else:
            feat_template, pose_template, feat, pose, _ = checkpoint.checkpoint(self._encode_image, template, shape_template, images_all, shape_all, intrinsic_embedding_all, use_reentrant=False)

        feat = rearrange(feat, "(b v) l c -> b v l c", b=b, v=v).contiguous()
        pose = rearrange(pose, "(b v) l c -> b v l c", b=b, v=v).contiguous()

        # step 2: decoder
        if self.disable_checkpointing:
            dec_feat_template, dec_feat = self._decoder(feat_template, pose_template, feat, pose)
        else:
            dec_feat_template, dec_feat = checkpoint.checkpoint(self._decoder, feat_template, pose_template, feat, pose, use_reentrant=False)
        shape = rearrange(shape_all, "(b v) c -> b v c", b=b, v=v).contiguous()
        images = rearrange(images_all, "(b v) c h w -> b v c h w", b=b, v=v).contiguous()

        if self.intrinsics_embed_loc == 'encoder' and self.intrinsics_embed_type == 'token':
            dec_feat_template = list(dec_feat_template)
            for i in range(len(dec_feat_template)):
                dec_feat_template[i] = dec_feat_template[i][:, :-1]
            dec_feat = list(dec_feat)
            for i in range(len(dec_feat)):
                dec_feat[i] = dec_feat[i][:, :, :-1]

        if return_pos:
            return dec_feat, shape, images, pose[:, :, :-1]
        return [dec_feat_template, dec_feat], [shape_template, shape], [template, images]

    @property
    def patch_size(self) -> int:
        return 16

    @property
    def d_out(self) -> int:
        return 1024
