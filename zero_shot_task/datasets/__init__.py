import os
import torch
import numbers
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from datasets.celeba import CelebA
from datasets.lsun import LSUN
from torch.utils.data import Subset
import numpy as np
import torchvision
from PIL import Image
from functools import partial

class Crop(object):
    def __init__(self, x1, x2, y1, y2):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def __call__(self, img):
        return F.crop(img, self.x1, self.y1, self.x2 - self.x1, self.y2 - self.y1)

    def __repr__(self):
        return self.__class__.__name__ + "(x1={}, x2={}, y1={}, y2={})".format(
            self.x1, self.x2, self.y1, self.y2
        )

def center_crop_arr(pil_image, image_size = 256):
    # Imported from openai/guided-diffusion
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def get_dataset(args, config):
    print('config.data.image_size ---> ',config.data.image_size)
    if config.data.random_flip is False:
        tran_transform = test_transform = transforms.Compose(
            [transforms.Resize(config.data.image_size), transforms.ToTensor()]
        )
    else:
        tran_transform = transforms.Compose(
            [
                transforms.Resize(config.data.image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
            ]
        )
        test_transform = transforms.Compose(
            [transforms.Resize(config.data.image_size), transforms.ToTensor()]
        )

    print('config.data.dataset ---> ', config.data.dataset)

    if config.data.dataset == 'ImageNet':
        # only use validation dataset here
        
        if config.data.subset_1k:
            from datasets.imagenet_subset import ImageDataset
            print('PATH ---> ',os.path.join(args.exp, 'datasets', 'imagenet', 'imagenet'))
            print('imagenet_val_1k PATH ---> ',os.path.join(args.exp, 'imagenet_val_1k.txt'))
            '''
            dataset = ImageDataset(os.path.join(args.exp, 'datasets', 'imagenet', 'imagenet'),
                     os.path.join(args.exp, 'imagenet_val_1k.txt'),
                     image_size=config.data.image_size,
                     normalize=False)
            os.path.join(args.exp, 'datasets', 'imagenet', 'MRI_test'),
            os.path.join(args.input_folder),
            '''
            dataset = ImageDataset(
                    # os.path.join(args.exp, 'datasets', 'imagenet', 'MRI_test'),
                    os.path.join(args.input_folder),
                     os.path.join(args.exp, 'imagenet_val_1k.txt'),
                     image_size=config.data.image_size,
                     normalize=False,
                     test_data_num=config.data.test_data_num,
                     tag_name = args.tag_name)
            test_dataset = dataset
            print('test_dataset 2 ---===> ',len(test_dataset), config.data.test_data_num)


    return dataset, test_dataset


def logit_transform(image, lam=1e-6):
    image = lam + (1 - 2 * lam) * image
    return torch.log(image) - torch.log1p(-image)


def data_transform(config, X):
    if config.data.uniform_dequantization:
        X = X / 256.0 * 255.0 + torch.rand_like(X) / 256.0
    if config.data.gaussian_dequantization:
        X = X + torch.randn_like(X) * 0.01

    if config.data.rescaled:
        X = 2 * X - 1.0
    elif config.data.logit_transform:
        X = logit_transform(X)

    if hasattr(config, "image_mean"):
        return X - config.image_mean.to(X.device)[None, ...]

    return X


def inverse_data_transform(config, X):
    if hasattr(config, "image_mean"):
        X = X + config.image_mean.to(X.device)[None, ...]

    if config.data.logit_transform:
        X = torch.sigmoid(X)
    elif config.data.rescaled:
        X = (X + 1.0) / 2.0

    return torch.clamp(X, 0.0, 1.0)
