from typing import Any, Mapping, Optional, Union

import random

import torch
import torch.nn as nn

import datasets


@torch.enable_grad()
def sgld(x, energy_fn, num_steps: int = 60, step_size: float = 100., step_noise: float = 0.001, step_clip: Optional[float] = None):
    x = x.clone().detach()
    for k in range(num_steps):
        x.requires_grad = True
        energy = energy_fn(x)
        g = torch.autograd.grad(energy.sum(), [x])[0]
        if step_clip is not None:
            g = g.clamp(-step_clip, step_clip)
        x = x - g * step_size + torch.randn_like(x) * step_noise
        x = x.clamp(0, 1).clone().detach()
    return x


class Sampler(nn.Module):
    def sample(self,
               num_samples: Union[int, list[int]],
               energy_fn: Mapping[Any, Any],
               train: bool = True):
        raise NotImplementedError


class MCMCSampler(Sampler):
    def __init__(self,
                 input_shape: tuple[int] = (3, 32, 32),
                 num_steps: float = 20,
                 step_size: float = 10,
                 step_noise: float = 0.005,
                 step_clip: Optional[float] = None,
                 memory_size: int = 10000,
                 memory_init: float = 0.05,
                 augmentation: str = 'none'):
        super().__init__()
        self.num_steps = num_steps
        self.step_size = step_size
        self.step_noise = step_noise
        self.step_clip = step_clip
        self.memory_size = memory_size
        self.memory_init = memory_init
        self.register_buffer('memory', torch.zeros(memory_size, *input_shape).uniform_(0, 1))
        self.augmentation = datasets.get_augmentation(augmentation, input_shape)

    @torch.enable_grad()
    def sample(self,
               num_samples: Union[int, list[int]],
               energy_fn: Mapping[Any, Any],
               train: bool = True):
        if type(num_samples) is list:
            indices = num_samples
            num_samples = len(indices)
        elif type(num_samples) is int:
            indices = random.sample(list(range(self.memory_size)), num_samples)

        # Initialize
        x = self.memory[indices].clone().detach()
        for i in range(num_samples):
            if random.random() < self.memory_init:
                x[i].uniform_(0, 1)

        # Apply Data Augmentation
        x = self.augmentation(x).clone().detach()

        # SGLD
        x = sgld(x, energy_fn=energy_fn,
                 num_steps=self.num_steps,
                 step_size=self.step_size,
                 step_clip=self.step_size,
                 step_noise=self.step_noise)

        # Update Replay Buffer
        if train:
            self.memory[indices] = x.detach()

        return x

