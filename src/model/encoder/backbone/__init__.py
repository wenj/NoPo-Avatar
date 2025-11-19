from typing import Any
import torch.nn as nn

from .backbone import Backbone
from .backbone_croco_multiview import AsymmetricCroCoMulti
from .backbone_dino import BackboneDino, BackboneDinoCfg
from .backbone_resnet import BackboneResnet, BackboneResnetCfg
from .backbone_croco import AsymmetricCroCo, BackboneCrocoCfg
from .backbone_croco_multiview2 import AsymmetricCroCoMulti2, BackboneCrocoMulti2Cfg

BACKBONES: dict[str, Backbone[Any]] = {
    "resnet": BackboneResnet,
    "dino": BackboneDino,
    "croco": AsymmetricCroCo,
    "croco_multi": AsymmetricCroCoMulti,
    "croco_multi2": AsymmetricCroCoMulti2,
}

BackboneCfg = BackboneResnetCfg | BackboneDinoCfg | BackboneCrocoCfg | BackboneCrocoMulti2Cfg


def get_backbone(cfg: BackboneCfg, d_in: int = 3) -> nn.Module:
    return BACKBONES[cfg.name](cfg, d_in)
