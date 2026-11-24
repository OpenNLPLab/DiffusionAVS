import torch
from datasets.loader_avs import S4Dataset, MS3Dataset


def get_loader(cfg, evaluate_only=False):
    eval_split = 'test' if evaluate_only else 'val'
    if cfg.TRAIN.TASK.lower() == "s4":
        train_dataset = None if evaluate_only else S4Dataset(cfg=cfg, split='train')
        eval_dataset = S4Dataset(cfg=cfg, split=eval_split)
    elif cfg.TRAIN.TASK.lower() == "ms3":
        train_dataset = None if evaluate_only else MS3Dataset(cfg=cfg, split='train')
        eval_dataset = MS3Dataset(cfg=cfg, split=eval_split)
    else:
        raise ValueError(f'Unknown task: {cfg.TRAIN.TASK}')
        
    train_dataloader = None if evaluate_only else torch.utils.data.DataLoader(train_dataset, batch_size=cfg.PARAM.BATCH_SIZE, shuffle=True,
                                                   num_workers=cfg.PARAM.NUM_WORKERS, pin_memory=False)
    eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=1, shuffle=False,
                                                 num_workers=cfg.PARAM.NUM_WORKERS, pin_memory=False)

    return train_dataloader, eval_dataloader
