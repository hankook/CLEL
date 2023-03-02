import os

from omegaconf import OmegaConf

import torch

from ignite.engine import Engine
from ignite import distributed as idist
from ignite.utils import convert_tensor

import utils
import models
import datasets
import architectures

def main(cfg: OmegaConf):
    device = idist.device()
    logger, tb_logger = utils.setup_logger(cfg)

    # Data
    dataset = datasets.get_dataset(name=cfg.data.name, root=cfg.data.root)
    trainloader = idist.auto_dataloader(dataset['train'],
                                        batch_size=cfg.data.batch_size,
                                        num_workers=cfg.data.num_workers,
                                        shuffle=True, drop_last=True)

    # Model
    model = models.get_model(cfg, dataset).to(device)

    # Optimizers
    optimizer1, scheduler1 = utils.get_adam_optimizer(
            model.param_groups['ebm'], lr=1e-4, momentum=(0., 0.999), lr_warmup=2000)
    optimizer2, scheduler2 = utils.get_sgd_optimizer(
            model.param_groups['encoder'], lr=3e-2, weight_decay=5e-4, num_iterations=cfg.num_iterations)
    optimizers = [optimizer1, optimizer2]
    schedulers = [scheduler1, scheduler2]

    # EMA
    ema_decay = 0.5 ** (cfg.data.batch_size / cfg.ema.halflife)
    ema_start = cfg.ema.halflife / cfg.data.batch_size
    logger.info(f'Apply EMA (decay: {ema_decay:.8f}) after {int(ema_start)} iterations')

    def training_step(engine, batch):
        model.train()
        model.update_ema_modules(decay=0. if engine.state.iteration < ema_start else ema_decay)
        batch = convert_tensor(batch, device=device)

        all_outputs = {}
        for idx, (optimizer, scheduler) in enumerate(zip(optimizers, schedulers)):
            optimizer.zero_grad()
        outputs = model(batch)
        outputs['metrics/loss'].backward()
        for idx, (optimizer, scheduler) in enumerate(zip(optimizers, schedulers)):
            optimizer.step()
            scheduler.step()
        return outputs

    engine = Engine(training_step)
    evaluations = [
        utils.FIDEvaluation(dataset['test'], device),
        utils.NNEvaluation(dataset['train'], dataset['test'], device),
    ]
    utils.add_default_handlers(cfg, engine, model, optimizers, schedulers,
                               logger, tb_logger, evaluations)
    engine.run(trainloader, max_iters=cfg.num_iterations)


if __name__ == '__main__':
    cfg = utils.parse_config()
    main(cfg)

