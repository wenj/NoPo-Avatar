from .decoder import Decoder
from .decoder_splatting_cuda import DecoderSplattingCUDA, DecoderSplattingCUDACfg
from .decoder_lbs_splatting_cuda import DecoderLBSSplattingCUDA, DecoderLBSSplattingCUDACfg

DECODERS = {
    "splatting_cuda": DecoderSplattingCUDA,
    "lbs_splatting_cuda": DecoderLBSSplattingCUDA,
}

DecoderCfg = DecoderSplattingCUDACfg | DecoderLBSSplattingCUDACfg


def get_decoder(decoder_cfg: DecoderCfg) -> Decoder:
    return DECODERS[decoder_cfg.name](decoder_cfg)
