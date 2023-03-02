from typing import Any, Mapping, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.ema_modules = []

    def update_ema_modules(self, decay: float = 1.):
        for name in self.ema_modules:
            module_dst = getattr(self, f'ema_{name}')
            module_src = getattr(self, name)
            params_dst = dict(module_dst.named_parameters())
            params_src = dict(module_src.named_parameters())
            buf_dst = dict(module_dst.named_buffers())
            buf_src = dict(module_src.named_buffers())

            for k in params_dst.keys():
                params_dst[k].data.mul_(decay).add_(params_src[k].data, alpha=1-decay)
            for k in buf_dst.keys():
                buf_dst[k].data.copy_(buf_src[k].data)

