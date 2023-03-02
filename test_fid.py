import os

from omegaconf import OmegaConf

import torch
import torch.nn.functional as F

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
    dataset = datasets.get_dataset(name=cfg.data.name, root=cfg.data.root)

    # Model
    model = models.get_model(cfg, dataset).to(device)

    # Setup
    step = cfg.get('step', 'last')
    ckpt = torch.load(os.path.join(cfg.logdir, f'{step}.ckpt'), map_location='cpu')
    print(ckpt['engine'])
    model.load_state_dict(ckpt['model'])
    model.eval()

    if cfg.get('use_ema', False):
        model.ebm.load_state_dict(model.ema_ebm.state_dict())

    if cfg.data.name in ['cifar10']:
        data = dataset['train']
    elif cfg.data.name in ['imagenet32']:
        data = dataset['test']
    else:
        raise Exception

    fid_evaluation = utils.FIDEvaluation(data, device)
    is_evaluation = utils.ISEvaluation(device)
    NUM_SAMPLES = len(data)
    if isinstance(model.sampler, models.MCMCSampler):
        NUM_ITERATIONS = cfg.get('num_iters', 10)
        BATCH_SIZE = 512
        STEP_NOISE = cfg.sampler.step_noise

        model.sampler.memory_size = NUM_SAMPLES
        model.sampler.register_buffer('memory', torch.zeros(NUM_SAMPLES, *dataset['input_shape']).uniform_(0, 1).to(device))

        for i in range(NUM_ITERATIONS):
            for j in range(0, NUM_SAMPLES, BATCH_SIZE):
                model.sample(list(range(j, min(j+BATCH_SIZE, NUM_SAMPLES))), train=True)
                print(j, end='\r')

            print(f'[{i+1:3d}] [FID: {fid_evaluation.evaluate(model):.2f}] [IS: {is_evaluation.evaluate(model):.2f}]')

    torch.save(model.sampler.memory, os.path.join(cfg.logdir, 'samples.pth'))

if __name__ == '__main__':
    cfg = utils.parse_config()
    main(cfg)

