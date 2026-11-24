from model.decoder.simple_decoder import simple_decoder_no_modulelist

def get_decoder(cfg):
    assert cfg.MODEL.DECODER == 'simple_no_module'
    return simple_decoder_no_modulelist(cfg.MODEL.NECK_CHANNEL)
