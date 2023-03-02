from typing import Any, Mapping, Optional, Union

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseModel
from .sampler import Sampler, sgld

import datasets
import architectures
import utils


def simclr(z, temperature: float = 0.2, k: Optional[torch.Tensor] = None):
    n = z.shape[0] // 2
    z = F.normalize(z)
    if k is None:
        logits = torch.mm(z, z.T).div(temperature)
    else:
        k = F.normalize(k)
        logits = torch.mm(z, torch.cat([z, k]).T).div(temperature)
    logits.fill_diagonal_(float('-inf'))
    labels = torch.tensor(list(range(n, 2*n))+list(range(n)), device=logits.device)
    return F.cross_entropy(logits, labels)


class JointModel(BaseModel):
    def __init__(self,
                 ebm: nn.Module, ema_ebm: nn.Module,
                 encoder: nn.Module, ema_encoder: nn.Module,
                 sampler: Sampler,

                 input_shape: tuple[int] = (3, 32, 32),
                 ebm_augmentation: str = 'none',
                 encoder_augmentation: str = 'strong',
                 alpha: float = 1.,
                 beta: float = 0.,
                 use_neg: bool = False,
                 gamma: float = 0.,

                 latent_mode: str = 'simclr',
                 temperature: float = 0.2,

                 **kwargs,
             ):
        super().__init__()

        self.param_groups = { 'ebm': [], 'encoder': [] }
        self.input_shape = input_shape
        self.ebm_augmentation = datasets.get_augmentation(ebm_augmentation, input_shape)
        self.encoder_augmentation = datasets.get_augmentation(encoder_augmentation, input_shape)

        # EBM
        self.ebm = ebm
        self.ema_ebm = ema_ebm
        self.sampler = sampler
        self.alpha = alpha
        self.beta = beta
        self.ema_modules.append('ebm')
        self.param_groups['ebm'].extend(list(self.ebm.parameters()))

        # Encoder
        self.encoder = encoder
        self.ema_encoder = ema_encoder
        self.temperature = temperature
        self.latent_mode = latent_mode
        self.use_neg = use_neg
        self.ema_modules.append('encoder')
        self.param_groups['encoder'].extend(list(self.encoder.parameters()))

        if latent_mode == 'simclr':
            pass

        assert len(kwargs) == 0

    def compute_divergence(self, e_pos: torch.Tensor, e_neg: torch.Tensor):
        return (e_pos - e_neg).mean()

    def energy_fn(self, x: torch.Tensor):
        f, z = self.ebm(x)
        return f.pow(2).sum(dim=1).div(2)

    @torch.no_grad()
    def compute_ood_scores(self, x: torch.Tensor):
        f, z = self.ebm(x)
        y = self.encoder(self.ebm_augmentation(x)).detach()
        e = f.pow(2).sum(dim=1).div(2)
        return e - self.beta * F.cosine_similarity(y, z)

    def sample(self, num_samples: Union[int, list[int]], train: bool = True):
        return self.sampler.sample(num_samples, energy_fn=self.energy_fn, train=train)

    def forward(self, batch):
        # 1. EBM
        x_pos, _ = batch
        x_neg = self.sample(x_pos.shape[0], train=True)
        f_pos, z_pos = self.ebm(x_pos)
        f_neg, z_neg = self.ebm(x_neg)
        e_pos = f_pos.pow(2).sum(dim=1).div(2)
        e_neg = f_neg.pow(2).sum(dim=1).div(2)
        loss = self.compute_divergence(e_pos, e_neg) + self.alpha * (e_pos ** 2 + e_neg ** 2).mean()

        with torch.no_grad():
            y_pos = self.encoder(self.ebm_augmentation(x_pos)).detach()
        s_pos = F.cosine_similarity(z_pos, y_pos)
        loss = loss - self.beta * s_pos.mean()

        # 2. Latent Encoder
        if self.latent_mode == 'simclr':
            y_pos = self.encoder(self.encoder_augmentation(torch.cat([x_pos, x_pos])))
            k = z_neg.detach() if self.use_neg else None
            latent_loss = simclr(y_pos, self.temperature, k=k)

        else:
            raise NotImplementedError

        loss = loss + latent_loss

        outputs = {
            'metrics/loss': loss,
            'metrics/e_pos': e_pos.mean(),
            'metrics/e_neg': e_neg.mean(),
            'metrics/e_diff': e_pos.mean() - e_neg.mean(),
            'metrics/s_pos': s_pos.mean(),
            'metrics/latent_loss': latent_loss,
            'images/x_pos': x_pos,
            'images/x_neg': x_neg,
        }

        return outputs

    @torch.no_grad()
    def update_queue(self, z):
        ptr = self.queue_ptr[0].item()
        self.queue[ptr:ptr+z.shape[0]] = z.detach()
        ptr = (ptr + z.shape[0]) % self.queue.shape[0]
        self.queue_ptr[0] = ptr

