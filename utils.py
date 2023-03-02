from typing import Any, Mapping, Optional, Union

import os
import datetime
import logging

from omegaconf import OmegaConf

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.tensorboard as tb
import torchvision
from torch.utils.data import DataLoader

import ignite
from ignite.engine import Engine, Events
from ignite import distributed as idist
from ignite.utils import convert_tensor

import models


def build_ema_module(fn, *args, **kwargs):
    module     = fn(*args, **kwargs)
    ema_module = fn(*args, **kwargs)
    ema_module.load_state_dict(module.state_dict())
    ema_module.requires_grad_(False)
    return module, ema_module


def parse_config() -> OmegaConf:
    cfg = OmegaConf.create()
    for x in os.sys.argv[1:]:
        if x.endswith('.yaml') and '=' not in x:
            cfg = OmegaConf.merge(cfg, OmegaConf.load(x))
        else:
            cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist([x]))
    return cfg


def setup_logger(cfg: OmegaConf):
    logdir = cfg.logdir
    os.makedirs(logdir, exist_ok=cfg.get('resume', False))
    OmegaConf.save(cfg, os.path.join(logdir, 'config.yaml'))
    logger = ignite.utils.setup_logger(name='train', filepath=os.path.join(logdir, 'log.txt'))
    tb_logger = tb.SummaryWriter(logdir)
    with open(os.path.join(logdir, 'config.yaml')) as f:
        logger.info('\n' + f.read())
    return logger, tb_logger


def get_adam_optimizer(params,
                       lr: float,
                       momentum: list[float],
                       lr_warmup: int):

    def _linear_warmup_fn(step):
        if step < lr_warmup:
            return (step+1) / lr_warmup
        else:
            return 1.0

    optimizer = torch.optim.Adam(params, lr=lr, betas=momentum)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _linear_warmup_fn)
    return optimizer, scheduler


def get_sgd_optimizer(params,
                      lr: float,
                      weight_decay: float,
                      num_iterations: int):

    optimizer = torch.optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=0.9) 
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_iterations)
    return optimizer, scheduler


def _convert_uint8_images(x): # [0, 1] -> [0, 1, ..., 255]
    return (x*256).long().clamp(0, 255).byte()


class Evaluation:
    def __init__(self, name: str, mode: str = 'max'):
        assert mode in ['max', 'min']
        self.name = name
        self.curr_score = float('-inf') if mode == 'max' else float('inf')
        self.best_score = float('-inf') if mode == 'max' else float('inf')
        self.mode = mode
        self.is_best = True

    def evaluate(self, model: nn.Module) -> float:
        raise NotImplementedError

    def update(self, model: nn.Module) -> tuple[float, float]:
        self.curr_score = self.evaluate(model)
        if (self.mode == 'max' and self.best_score < self.curr_score) \
                or (self.mode == 'min' and self.best_score > self.curr_score):
            self.best_score = self.curr_score
            self.is_best = True
        else:
            self.is_best = False

        return self.curr_score, self.best_score


class FIDEvaluation(Evaluation):
    def __init__(self, dataset, device):
        super().__init__(name='fid', mode='min')

        import torchmetrics
        self.device = device
        self.fid = torchmetrics.image.fid.FrechetInceptionDistance().to(device)

        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=128, num_workers=4,
                                                 shuffle=False, drop_last=False)
        for batch in dataloader:
            x = _convert_uint8_images(convert_tensor(batch[0], device=device))
            self.fid.update(x, real=True)

    def evaluate(self, model: nn.Module) -> float:
        self.fid.fake_features = []
        if isinstance(model.sampler, models.MCMCSampler):
            # Use Memory Buffer
            for i in range(0, model.sampler.memory_size, 128):
                x = model.sampler.memory[i:i+128]
                x = _convert_uint8_images(x)
                self.fid.update(x, real=False)
        else:
            raise NotImplementedError

        return self.fid.compute().item()


class ISEvaluation(Evaluation):
    def __init__(self, device):
        super().__init__(name='is', mode='max')

        import torchmetrics
        self.device = device
        self.inception = torchmetrics.image.inception.InceptionScore().to(device)

    def evaluate(self, model: nn.Module) -> float:
        self.inception.features = []
        if isinstance(model.sampler, models.MCMCSampler):
            # Use Memory Buffer
            for i in range(0, model.sampler.memory_size, 128):
                x = model.sampler.memory[i:i+128]
                x = _convert_uint8_images(x)
                self.inception.update(x)
        else:
            raise NotImplementedError

        return self.inception.compute()[0].item()


@torch.no_grad()
def _collect_features(model: nn.Module, dataloader: DataLoader, device: torch.device):
    X = []
    Y = []
    for x, y in dataloader:
        X.append(model.encoder(x.to(device)).detach())
        Y.append(y.to(device))
    X = torch.cat(X).detach()
    Y = torch.cat(Y).detach()
    return X, Y


@torch.no_grad()
def _collect_energies(model: nn.Module, dataloader: DataLoader, device: torch.device):
    E = []
    for x, y in dataloader:
        E.append(model.compute_ood_scores(x.to(device)).detach())
    E = torch.cat(E).detach()
    return E


class OODEvaluation(Evaluation):
    def __init__(self, ind, ood, device):
        super().__init__(name='ood', mode='max')

        self.indloader = DataLoader(ind,
                                    batch_size=128, num_workers=4,
                                    shuffle=False, drop_last=False)
        self.oodloader = DataLoader(ood,
                                    batch_size=128, num_workers=4,
                                    shuffle=False, drop_last=False)
        self.device = device

    @torch.no_grad()
    def evaluate(self, model: nn.Module) -> float:
        from sklearn.metrics import roc_auc_score

        ind_scores = _collect_energies(model, self.indloader, self.device).mul(-1)
        ood_scores = _collect_energies(model, self.oodloader, self.device).mul(-1)
        ind_labels = torch.ones_like(ind_scores)
        ood_labels = torch.zeros_like(ood_scores)

        scores = torch.cat([ind_scores, ood_scores], dim=0).detach().cpu().numpy()
        labels = torch.cat([ind_labels, ood_labels], dim=0).detach().cpu().numpy()
        return roc_auc_score(labels, scores)


class NNEvaluation(Evaluation):
    def __init__(self, train, test, device):
        super().__init__(name='nn', mode='max')

        self.trainloader = DataLoader(train,
                                      batch_size=128, num_workers=4,
                                      shuffle=False, drop_last=False)
        self.testloader  = DataLoader(test,
                                      batch_size=128, num_workers=4,
                                      shuffle=False, drop_last=False)
        self.device = device

    @torch.no_grad()
    def evaluate(self, model: nn.Module) -> float:
        X_train, Y_train = _collect_features(model, self.trainloader, self.device)
        X_test,  Y_test  = _collect_features(model, self.testloader,  self.device)
        num_corrects = 0
        for i in range(0, X_test.shape[0], 256):
            X_batch = X_test[i:i+256]
            Y_batch = Y_test[i:i+256]
            distance = X_batch.pow(2).sum(dim=1, keepdim=True) \
                       + X_train.pow(2).sum(dim=1).unsqueeze(0) \
                       - 2 * torch.mm(X_batch, X_train.T)
            Y_pred = Y_train[distance.argmin(1)]
            num_corrects += (Y_pred == Y_batch).long().sum().item()
        return num_corrects / X_test.shape[0]


def add_default_handlers(cfg: OmegaConf,
                         engine: Engine,
                         model: nn.Module,
                         optimizers: list[optim.Optimizer],
                         schedulers: list[optim.lr_scheduler._LRScheduler],
                         logger: logging.Logger,
                         tb_logger: tb.SummaryWriter,
                         evaluations: list[Evaluation]):

    engine.state_dict_user_keys.extend([e.name for e in evaluations])

    @engine.on(Events.STARTED)
    def init_user_value(_):
        for e in evaluations:
            setattr(engine.state, e.name, (e.curr_score, e.best_score))

    # Resume
    if cfg.get('resume', False):
        @engine.on(Events.STARTED)
        def resume(_):
            states = torch.load(os.path.join(cfg.logdir, 'last.ckpt'))
            model.load_state_dict(states['model'])
            for optimizer, state in zip(optimizers, states['optimizers']):
                optimizer.load_state_dict(state)
            for scheduler, state in zip(schedulers, states['schedulers']):
                scheduler.load_state_dict(state)
            engine.load_state_dict(states['engine'])

    # Log Metrics
    @engine.on(Events.ITERATION_COMPLETED(every=cfg.log_freq))
    def log_metrics(_):
        for k, v in engine.state.output.items():
            if k.startswith('metrics/'):
                tb_logger.add_scalar(k, v, engine.state.iteration)

    # Evaluation & Save Checkpoints
    @engine.on(Events.ITERATION_COMPLETED(every=cfg.val_freq))
    def eval_and_save_checkpoint(_):
        model.eval()
        for e in evaluations:
            e.update(model)
            setattr(engine.state, e.name, (e.curr_score, e.best_score))
            tb_logger.add_scalar(f'metrics/{e.name}',
                                 e.curr_score,
                                 global_step=engine.state.iteration)
            tb_logger.add_scalar(f'metrics/{e.name}_best',
                                 e.best_score,
                                 global_step=engine.state.iteration)

        state = {
            'model': model.state_dict(),
            'optimizers': [_.state_dict() for _ in optimizers],
            'schedulers': [_.state_dict() for _ in schedulers],
            'engine': engine.state_dict(),
        }
        torch.save(state, os.path.join(cfg.logdir, 'last.ckpt'))
        if evaluations[0].is_best:
            torch.save(state, os.path.join(cfg.logdir, 'best.ckpt'))

        if engine.state.iteration % cfg.save_freq == 0:
            torch.save(state, os.path.join(cfg.logdir, f'{engine.state.iteration}.ckpt'))

        for k in ['images/x_pos', 'images/x_neg']:
            if k in engine.state.output:
                x = _convert_uint8_images(engine.state.output[k][:16])
                tb_logger.add_image(k,
                                    torchvision.utils.make_grid(x, nrow=4),
                                    global_step=engine.state.iteration)

    # Print Progress
    @engine.on(Events.ITERATION_COMPLETED)
    def print_iteration(_):
        state = engine.state
        text = f'[{state.iteration:6d} / {state.max_iters:6d}] [Loss {state.output["metrics/loss"]:.4f}] '
        for e in evaluations:
            text += f'[{e.name.upper()} {e.curr_score:.2f} | {e.best_score:.2f}] '
        print(text, end='\r')
        if state.iteration % cfg.val_freq == 0:
            logger.info(text)

