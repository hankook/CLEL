import os

from omegaconf import OmegaConf

import torch
import torch.nn as nn

from ignite import distributed as idist
from ignite.utils import convert_tensor

import utils
import models
import datasets
import architectures

def main(cfg: OmegaConf):
    print(cfg)

    device = idist.device()

    # Data
    dataset     = datasets.get_dataset(name=cfg.data.name,     root=cfg.data.root)
    ood_dataset = datasets.get_dataset(name=cfg.ood_data.name, root=cfg.ood_data.root)

    # Model
    model = models.get_model(cfg, dataset).to(device)

    step = cfg.get('step', 'last')
    ckpt = torch.load(os.path.join(cfg.logdir, f'{step}.ckpt'), map_location='cpu')
    print(ckpt['engine'])
    print(model.load_state_dict(ckpt['model']))
    model.eval()

    if cfg.get('use_ema', False):
        model.ebm.load_state_dict(model.ema_ebm.state_dict())
        model.encoder.load_state_dict(model.ema_encoder.state_dict())

    ood_evaluation = utils.OODEvaluation(dataset['test'], ood_dataset['test'], device)
    auroc = ood_evaluation.evaluate(model)
    print(f'[AUROC: {auroc:.4f}]')

if __name__ == '__main__':
    cfg = utils.parse_config()
    main(cfg)

