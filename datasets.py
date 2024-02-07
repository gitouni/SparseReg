# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
from PIL import Image
from torchvision import datasets, transforms
from torch import Tensor
from torch.utils.data import Dataset, Subset, ConcatDataset
import numpy as np
import torch
from torch import Generator, randperm
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from typing import Union, Tuple
from collections.abc import Iterable, Callable
from functools import partial
import math
from torchvision.utils import make_grid

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

__transform_dict__ = {"center_crop":transforms.CenterCrop, "random_crop":transforms.RandomCrop,
        "resize":partial(transforms.Resize, interpolation=transforms.InterpolationMode.BICUBIC)}

def is_image_file(filename:str):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def pil_loader(path, mode='RGB') -> Image:
    return Image.open(path).convert(mode)

def kpt_loader(path) -> Tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path)
    return data[:, :2].astype(np.int32), data[:, 2:].astype(np.float32)

def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    root = os.path.join(args.data_path, 'train' if is_train else 'val')
    dataset = datasets.ImageFolder(root, transform=transform)

    print(dataset)

    return dataset

def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)


def find_imgs_indir(dirname:Union[str, Iterable]):
    if isinstance(dirname, str):
        filenames = [os.path.join(dirname,file) for file in sorted(os.listdir(dirname)) if is_image_file(file)]
        return filenames
    elif isinstance(dirname, Iterable):
        filenames = []
        for subdir in dirname:
            subfilenames = [os.path.join(subdir,file) for file in sorted(os.listdir(subdir)) if is_image_file(file)]
            filenames.extend(subfilenames)
        return filenames
    else:
        raise NotImplementedError("dirname must be str or Iterable, found {}.".format(type(dirname)))

def finds_kpts_indir(dirname:str):
    return [os.path.join(dirname, file) for file in sorted(os.listdir(dirname)) if os.path.splitext(file)[1] == '.txt']

class ImageKptDataset(Dataset):
    def __init__(self, data_path:str, kpt_path:str,
                  img_loader:Callable, kpt_loader:Callable,
                  img_transform:Callable, kpt_transform:float, split:Union[str, None]):
        self.imgs = find_imgs_indir(data_path)
        self.kpts = finds_kpts_indir(kpt_path)
        self.img_loader = img_loader
        self.kpt_loader = kpt_loader
        self.img_transform = img_transform
        self.kpt_transform = kpt_transform
        if split is None:
            self.split = np.arange(len(self.imgs))
        else:
            self.split = np.loadtxt(split, dtype=np.int32)

    def __getitem__(self, index_) -> Tuple[Tensor, Tensor]:
        index = self.split[index_]
        img:Image = self.img_loader(self.imgs[index])
        kpts, kpt_shift = self.kpt_loader(self.kpts[index])
        mask = torch.zeros((1, img.height, img.width), dtype=torch.float32)
        mask_shift = torch.zeros((2, img.height, img.width), dtype=torch.float32)
        mask[:, kpts[:,1], kpts[:, 0]] = 1.0
        mask_shift[:, kpts[:,1], kpts[:, 0]] = torch.from_numpy(kpt_shift).T
        return self.img_transform(img), mask, self.kpt_transform(mask_shift)

    def __len__(self):
        return len(self.split)

def subset_split(dataset, lengths, generator):
    """
    split a dataset into non-overlapping new datasets of given lengths. main code is from random_split function in pytorch
    """
    indices = randperm(sum(lengths), generator=generator).tolist()
    Subsets = []
    for offset, length in zip(np.add.accumulate(lengths), lengths):
        if length == 0:
            Subsets.append(None)
        else:
            Subsets.append(Subset(dataset, indices[offset - length : offset]))
    return Subsets

def make_dataset(opt:dict):
    phase:str = opt['phase']
    dataset_args:dict = opt['dataset']
    img_size = dataset_args['img_size']
    scale = dataset_args['scale']
    general_transforms = []
    if dataset_args['img_trans'] != 'none':
        general_transforms.append(__transform_dict__[dataset_args['img_trans']](size=img_size))
    img_tf = transforms.Compose(general_transforms + [transforms.ToTensor(),transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)])
    kpt_tf = transforms.Compose(general_transforms + [transforms.Normalize(mean=0, std=scale)])
    img_loader = pil_loader
    mask_loader = kpt_loader
    if isinstance(dataset_args['data_path'], str):
        assert isinstance(dataset_args['kpt_path'], str)
        if 'split' in dataset_args.keys():
            split = dataset_args['split']
        else:
            split = None
        dataset = ImageKptDataset(dataset_args['data_path'], dataset_args['kpt_path'],
            img_loader, mask_loader, img_tf, kpt_tf, split)
        if phase == 'train':
            assert 'val_split' in dataset_args.keys()
            total_length = len(dataset)
            val_length = int(total_length * dataset_args['val_split'])
            if val_length > 0:
                train_length = total_length - val_length
                train_dataset, val_dataset = subset_split(dataset, [train_length, val_length], Generator().manual_seed(dataset_args['seed']))
                return train_dataset, val_dataset
            else:
                return dataset, None
        else:
            return dataset
    elif isinstance(dataset_args['data_path'], Iterable):
        assert isinstance(dataset_args['kpt_path'], Iterable)
        if phase == 'train':
            train_dataset_list = []
            val_dataset_list = []
        else:
            dataset_list = []
        for i, (data_path, kpt_path) in enumerate(zip(dataset_args['data_path'], dataset_args['kpt_path'])):
            if 'split' in dataset_args.keys():
                split = dataset_args['split'][i]
            else:
                split = None
            dataset = ImageKptDataset(data_path, kpt_path,
                img_loader, mask_loader, img_tf, kpt_tf, split)
            if phase == 'train':
                total_length = len(dataset)
                val_length = int(total_length * dataset_args['val_split'])
                if val_length > 0:
                    train_length = total_length - val_length
                    train_dataset, val_dataset = subset_split(dataset, [train_length, val_length], Generator().manual_seed(dataset_args['seed']))
                train_dataset_list.append(train_dataset)
                val_dataset_list.append(val_dataset)
            else:
                dataset_list.append(dataset)
        if phase == 'train':
            return ConcatDataset(train_dataset_list), ConcatDataset(val_dataset_list)
        else:
            return ConcatDataset(dataset_list)
    else:
        raise NotImplementedError("data path must be Iterable or str")

def tensor2img(tensor:Tensor, out_type=np.uint8, mean_value:Union[float, Iterable]=IMAGENET_DEFAULT_MEAN, std_value:Union[float, Iterable]=IMAGENET_DEFAULT_STD):
	'''
	Converts a torch Tensor into an image Numpy array
	Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
	Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
	'''
	n_dim = tensor.dim()
	if isinstance(mean_value, Iterable):
		mean = np.array(mean_value)[None, None, :]  # (1,1,3)
		std = np.array(std_value)[None, None, :]  # (1,1,3)
	else:
		mean = mean_value
		std = std_value
	if n_dim == 4:
		n_img = len(tensor)
		img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), normalize=False).cpu().detach().numpy()
		img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
	elif n_dim == 3:
		img_np = tensor.cpu().detach().numpy()
		img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
	elif n_dim == 2:
		img_np = tensor.cpu().detach().numpy()
	else:
		raise TypeError('Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
	if out_type == np.uint8:
		img_np = (img_np * std) + mean
		img_np = (np.clip(img_np, 0, 1) * 255).round()
		# Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
	return img_np.astype(out_type).squeeze()

if __name__ == "__main__":
    import yaml
    config = yaml.load(open('cfg/train_val1.yml','r'),yaml.SafeLoader)
    train_dataset, val_dataset = make_dataset(config)
    print("len of train/val = {}/{}".format(len(train_dataset), len(val_dataset)))