from .loss import Loss
from .loss_depth import LossDepth, LossDepthCfgWrapper
from .loss_lpips import LossLpips, LossLpipsCfgWrapper
from .loss_ssim import LossSsim, LossSsimCfgWrapper
from .loss_mse import LossMse, LossMseCfgWrapper
from .loss_lbs_weights import LossLBSWeights, LossLBSWeightsCfgWrapper
from .loss_chamfer import LossChamfer, LossChamferCfgWrapper
from .loss_projection import LossProjection, LossProjectionCfgWrapper
from .loss_projection2 import LossProjection2, LossProjection2CfgWrapper
from .loss_pts3d import LossPts3D, LossPts3DCfgWrapper
from .loss_chamfer2d import LossChamfer2D, LossChamfer2DCfgWrapper

LOSSES = {
    LossDepthCfgWrapper: LossDepth,
    LossLpipsCfgWrapper: LossLpips,
    LossSsimCfgWrapper: LossSsim,
    LossMseCfgWrapper: LossMse,
    LossLBSWeightsCfgWrapper: LossLBSWeights,
    LossChamferCfgWrapper: LossChamfer,
    LossProjectionCfgWrapper: LossProjection,
    LossProjection2CfgWrapper: LossProjection2,
    LossPts3DCfgWrapper: LossPts3D,
    LossChamfer2DCfgWrapper: LossChamfer2D,
}

LossCfgWrapper = LossDepthCfgWrapper | LossLpipsCfgWrapper | LossMseCfgWrapper | LossLBSWeightsCfgWrapper \
                 | LossChamferCfgWrapper | LossProjectionCfgWrapper | LossProjection2CfgWrapper | LossPts3DCfgWrapper \
                 | LossChamfer2DCfgWrapper | LossSsimCfgWrapper


def get_losses(cfgs: list[LossCfgWrapper]) -> list[Loss]:
    return [LOSSES[type(cfg)](cfg) for cfg in cfgs]
