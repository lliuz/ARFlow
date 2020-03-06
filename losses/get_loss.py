from .flow_loss import unFlowLoss

def get_loss(cfg):
    if cfg.type == 'unflow':
        loss = unFlowLoss(cfg)
    else:
        raise NotImplementedError(cfg.type)
    return loss
