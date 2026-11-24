import torch
from model.encoder.pvt import pvt_v2_b5

def get_encoder(cfg):
    assert cfg.MODEL.ENCODER == 'pvt'
    backbone = pvt_v2_b5()
    weights = torch.load(cfg.TRAIN.PRETRAINED_PVTV2_PATH, map_location='cpu', weights_only=True)
    state = backbone.state_dict()
    state.update({k:v for k,v in weights.items() if k in state})
    backbone.load_state_dict(state)
    return backbone, [64, 128, 320, 512]
