import architectures
import utils

from .sampler import Sampler, MCMCSampler
from .ebm import JointModel

SAMPLER_DICT = {
    'MCMCSampler': MCMCSampler,
}

MODLE_DICT = {
    'JointModel': JointModel,
}

def get_model(cfg, dataset):
    # Energy-based Model 
    ebm, ema_ebm = utils.build_ema_module(architectures.get_ebm,
                                          input_shape=dataset['input_shape'],
                                          mean=dataset['mean'],
                                          std=dataset['std'],
                                          **cfg.ebm)

    # Encoder
    encoder, ema_encoder = utils.build_ema_module(architectures.get_encoder,
                                                  input_shape=dataset['input_shape'],
                                                  mean=dataset['mean'],
                                                  std=dataset['std'],
                                                  **cfg.encoder)

    # Sampler
    sampler_name = cfg.sampler.pop('name')
    sampler = SAMPLER_DICT[sampler_name](input_shape=dataset['input_shape'],
                                         **cfg.sampler)

    # Model
    model_name = cfg.model.pop('name')
    model = MODLE_DICT[model_name](ebm=ebm, ema_ebm=ema_ebm,
                                   encoder=encoder, ema_encoder=ema_encoder,
                                   sampler=sampler,
                                   input_shape=dataset['input_shape'],
                                   **cfg.model)

    return model

