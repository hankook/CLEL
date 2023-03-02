from typing import Any, Mapping, Optional, Union

import os

import torch
import torch.nn as nn

import torchvision.datasets as D
import torchvision.transforms as T
import kornia.augmentation as K

from ignite import distributed as idist


class PixelNoiseTransform(object):
    # This noise can be removed by utils._convert_uint8_images
    def __call__(self, img):
        x = T.functional.to_tensor(img)
        x = x * 255 / 256
        x = x + torch.zeros_like(x).uniform_(0, 1/256)
        return x


class CIFAR10Interpolation(D.CIFAR10):
    offset = 100

    def __len__(self):
        return super().__len__() - self.offset

    def __getitem__(self, idx):
        a, _ = super().__getitem__(idx)
        b, _ = super().__getitem__(idx+self.offset)
        return (a + b) / 2, 0


class DTD(D.ImageFolder):
    def __getitem__(self, idx):
        x, y = super().__getitem__(idx)
        x = x[:, :32:, :32]
        return x, y


def get_dataset(name: str = 'cifar10', root: str = '/data'):
    transform_train = PixelNoiseTransform()
    transform_test  = PixelNoiseTransform()

    if name == 'cifar10':
        train = D.CIFAR10(root, train=True,  transform=transform_train)
        test  = D.CIFAR10(root, train=False, transform=transform_test)
        return dict(train=train, test=test, num_classes=10, input_shape=(3, 32, 32),
                    mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])

    if name == 'cifar100':
        train = D.CIFAR100(root, train=True,  transform=transform_train)
        test  = D.CIFAR100(root, train=False, transform=transform_test)
        return dict(train=train, test=test, num_classes=100, input_shape=(3, 32, 32),
                    mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2762])

    elif name == 'svhn':
        train = D.SVHN(root, split='train', transform=transform_train)
        test  = D.SVHN(root, split='test',  transform=transform_test)
        return dict(train=train, test=test, num_classes=10, input_shape=(3, 32, 32),
                    mean=[0.4377, 0.4438, 0.4728], std=[0.1980, 0.2010, 0.1970])

    elif name == 'cifar10_interp':
        train = CIFAR10Interpolation(root, train=True,  transform=transform_train)
        test  = CIFAR10Interpolation(root, train=False, transform=transform_test)
        return dict(train=train, test=test, num_classes=10, input_shape=(3, 32, 32),
                    mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])

    elif name == 'dtd':
        train = DTD(root, transform=transform_train)
        test  = DTD(root, transform=transform_test)
        return dict(train=train, test=test, num_classes=10, input_shape=(3, 32, 32),
                    mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])

    elif name == 'imagenet32':
        train = D.ImageFolder(os.path.join(root, 'train'), transform=transform_train)
        test  = D.ImageFolder(os.path.join(root, 'val'),   transform=transform_test)
        return dict(train=train, test=test, num_classes=100, input_shape=(3, 32, 32),
                    mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]) # Just use the CIFAR10 statistics 

    elif name == 'celeba64':
        train = D.CelebA(root, split='train', transform=T.Compose([T.CenterCrop(140), T.Resize((64, 64)), transform_train]))
        test  = D.CelebA(root, split='valid', transform=T.Compose([T.CenterCrop(140), T.Resize((64, 64)), transform_test]))
        return dict(train=train, test=test, num_classes=1, input_shape=(3, 64, 64),
                    mean=[0.5186, 0.4182, 0.3647], std=[0.2996, 0.2715, 0.2668])

    else:
        raise Exception(f'Unknown Dataset: {name}')


def get_loaders(name: str = 'cifar10',
                root: str = '/data',
                batch_size: int = 128,
                num_workers: int = 4):
    train, test, input_shape = get_datasets(name, root)
    trainloader = idist.auto_dataloader(train, batch_size=batch_size, num_workers=num_workers, shuffle=True,  drop_last=True)
    testloader  = idist.auto_dataloader(test,  batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=False)
    return trainloader, testloader, input_shape


def get_augmentation(name, input_shape):
    if name == 'none':
        return K.AugmentationSequential(
            return_transform=False,
            same_on_batch=False,
        )

    elif name == 'weak':
        assert input_shape == (3, 32, 32)
        aug = K.AugmentationSequential(
            K.RandomCrop(input_shape[1:], padding=4, padding_mode='reflect', resample='BICUBIC'),
            K.RandomHorizontalFlip(),
            return_transform=False,
            same_on_batch=False,
        )
        return aug

    elif name == 'strong':
        aug = K.AugmentationSequential(
            K.RandomResizedCrop(input_shape[1:], scale=(0.08, 1.0), resample='BICUBIC'),
            K.ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
            K.RandomGrayscale(p=0.2),
            K.RandomHorizontalFlip(),
            return_transform=False,
            same_on_batch=False,
        )
        return aug

    else:
        raise Exception(f'Unknown Augmentation: {name}')

