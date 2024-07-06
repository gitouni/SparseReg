import os
from PIL import Image
from torchvision import transforms
from torch import Tensor
from torch.utils.data import Dataset, Subset, ConcatDataset, DataLoader
from torch.utils.data import BatchSampler
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
from typing import List
import json
from copy import deepcopy

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


class DynamicBatchSampler(BatchSampler):
    def __init__(self, num_sequences:int, dataset_len:int=1024,
            max_images:int=128, images_per_seq:Tuple[int,int]=(3, 10),
            motion_type:List[str]=['normal','shear']):
        # Batch sampler with a dynamic number of sequences
        # max_images >= number_of_sequences * images_per_sequence
        assert images_per_seq[1] >= images_per_seq[0] >= 2, 'invalid image_per_seq:{}'.format(images_per_seq)
        self.max_images = max_images
        self.images_per_seq = list(range(images_per_seq[0], images_per_seq[1]))
        self.num_sequences = num_sequences
        self.dataset_len = dataset_len
        self.motion_type = motion_type

    def __iter__(self):
        for _ in range(self.dataset_len):
            # number per sequence
            n_per_seq = np.random.choice(self.images_per_seq)
            # number of sequences
            n_seqs = self.max_images // n_per_seq

            # randomly select sequences
            chosen_seq = self._capped_random_choice(self.num_sequences, n_seqs)

            # get item
            batches = [(bidx, n_per_seq) for bidx in chosen_seq]
            yield batches

    def _capped_random_choice(self, num_seq:int, sampling_size:int, keep_order:bool=True):
        if sampling_size <= num_seq:
            seq_idx = np.random.choice(num_seq, size=sampling_size, replace=False)
        else:
            seq_idx = np.hstack((np.random.permutation(num_seq), np.random.choice(num_seq, sampling_size - num_seq, replace=True)))
        if keep_order:
            seq_idx.sort()
        return seq_idx

    def __len__(self):
        return self.dataset_len

class FixSeqBatchSampler(BatchSampler):
    def __init__(self, num_sequences, dataset_len=1024, seq_idx=0, images_per_seq=10):
        # Batch sampler with a dynamic number of sequences
        # max_images >= number_of_sequences * images_per_sequence

        self.seq_idx = seq_idx
        self.images_per_seq = images_per_seq
        self.num_sequences = num_sequences
        self.dataset_len = dataset_len

    def __iter__(self):
        for _ in range(self.dataset_len):
            # number per sequence
            n_per_seq = self.images_per_seq

            # get item
            batches = [(self.seq_idx, n_per_seq) for _ in range(self.num_sequences)]
            yield batches

    def __len__(self):
        return self.dataset_len

class BasicDataset(Dataset):
    def __init__(self, img_paths:List[str], kpt_paths:List[str],
                 img_loader:Callable, kpt_loader:Callable,
                 img_transform:Callable, kpt_transform:float):
        self.imgs = img_paths
        self.kpts = kpt_paths
        self.img_loader = img_loader
        self.kpt_loader = kpt_loader
        self.img_transform = img_transform
        self.kpt_transform = kpt_transform

    def __getitem__(self, index) -> Tuple[Tensor, Tensor, Tensor]:
        img = self.img_loader(self.imgs[index])
        kpts, kpt_shift = self.kpt_loader(self.kpts[index])
        if isinstance(img, Image.Image):
            h, w = img.height, img.width
        elif isinstance(img, np.ndarray):
            h, w = img.shape[:2]
        else:
            raise TypeError("Unspported img type.")
        mask = torch.zeros((1, h, w), dtype=torch.float32)
        mask_shift = torch.zeros((2, h, w), dtype=torch.float32)
        mask[:, kpts[:,1], kpts[:, 0]] = 1.0
        mask_shift[:, kpts[:,1], kpts[:, 0]] = torch.from_numpy(kpt_shift).T
        return self.img_transform(img), mask, self.kpt_transform(mask_shift)  # (3, H, W), (1, H, W), (2, H, W)

    def __len__(self):
        return len(self.imgs)
    
class ImgKptDataset(BasicDataset):
    def __init__(self, data_path:str, kpt_path:str,
                  img_loader:Callable, kpt_loader:Callable,
                  img_transform:Callable, kpt_transform:float, split_file:Union[str, None]):
        imgs = find_imgs_indir(data_path)
        kpts = finds_kpts_indir(kpt_path)
        if split_file is None:
            split = np.arange(len(self.imgs))
        else:
            split = np.loadtxt(split_file, dtype=np.int32)
        imgs = imgs[split]
        kpts = kpts[split]
        super().__init__(imgs, kpts, img_loader, kpt_loader, img_transform, kpt_transform)

class SeqDataset(Dataset):
    def __init__(self, data_path:str, kpt_path:str,
                 img_loader:Callable, kpt_loader:Callable,
                  img_transform:Callable, kpt_transform:float,
                  seq_json:str):
        seq_dict:dict = json.load(open(seq_json, 'r'))
        imgs = find_imgs_indir(data_path)
        kpts = finds_kpts_indir(kpt_path)
        self.dataset_list = []
        self.key_list = []
        dataset_argv = dict(img_loader=img_loader, kpt_loader=kpt_loader,
                            img_transform=img_transform, kpt_transform=kpt_transform)
        for key, topdir in seq_dict.items():
            for subkey, subdir in topdir.items():
                for subsubkey, subsubdir in subdir.items():
                    self.key_list.append([key, subkey, subsubkey])
                    img_file_list = [imgs[i] for i in subsubdir]
                    key_file_list = [kpts[i] for i in subsubdir]
                    self.dataset_list.append(BasicDataset(img_file_list, key_file_list, **dataset_argv))
    
    def getdata(self, seq_idx:int, file_idx:int) -> Tuple[Tensor, Tensor]:
        return self.dataset_list[seq_idx][file_idx]
    
    def __len__(self):
        return len(self.dataset_list)
    
    def split_itself(self, idx_list:Iterable[int]):
        sub_dataset = deepcopy(self)
        sub_dataset.key_list = [self.key_list[idx] for idx in idx_list]
        sub_dataset.dataset_list = [self.dataset_list[idx] for idx in idx_list]
        return sub_dataset
    
    def __getitem__(self, seq_file_index:Tuple[np.ndarray,int]) -> Tuple[Tensor, Tensor, Tensor]:
        seq_indices, n_file = seq_file_index
        seq_torch_data = []
        for seq_idx in seq_indices:
            seq_files = self.dataset_list[seq_idx]
            seq_len = len(seq_files)
            if len(seq_files) <= self.__len__():
                file_ids = np.random.choice(seq_len, n_file, replace=False)
            else:
                file_ids = np.random.choice(seq_len, n_file, replace=True)
            file_ids.sort()
            one_seq_data = []
            for file_idx in file_ids:
                data = self.getdata(seq_idx, file_idx)  # img, mask, kpt
                one_seq_data.append(data)
            one_seq_torch = [torch.stack(value) for value in zip(*one_seq_data)] # [seq_img, seq_mask, seq_kpt]
            seq_torch_data.append(one_seq_torch)  # [ [seq_img, seq_mask, seq_kpt],  [seq_img, seq_mask, seq_kpt], ...]
        return [torch.stack(value) for value in zip(*seq_torch_data)]  # [batch_seq_img, batch_seq_mask, batch_seq_kpt]
        
    

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
        dataset = ImgKptDataset(dataset_args['data_path'], dataset_args['kpt_path'],
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
            dataset = ImgKptDataset(data_path, kpt_path,
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

def make_seq_dataloader(opt:dict):
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
    if phase == 'train':
        seq_train_json = dataset_args['seq_json']
        dataset = SeqDataset(dataset_args['data_path'], dataset_args['kpt_path'], img_loader, mask_loader, img_tf, kpt_tf, seq_train_json)
        total_length = len(dataset)
        val_length = int(total_length * dataset_args['val_split'])
        if val_length > 0:
            train_length = total_length - val_length
            indices = randperm(total_length, generator=Generator().manual_seed(dataset_args['seed'])).tolist()
            train_dataset = dataset.split_itself(indices[:train_length])
            val_dataset = dataset.split_itself(indices[train_length:])
            train_batch_sampler = DynamicBatchSampler(len(train_dataset), **dataset_args['train_batch_sampler'])
            val_batch_sampler = DynamicBatchSampler(len(val_dataset), **dataset_args['val_batch_sampler'])
            train_dataloader = DataLoader(train_dataset, batch_sampler=train_batch_sampler, **dataset_args['train_dataloader'])
            val_dataloader = DataLoader(val_dataset, batch_sampler=val_batch_sampler, **dataset_args['val_dataloader'])
            return train_dataloader, val_dataloader
        
    elif phase == 'test':
        seq_test_json = dataset_args['seq_json']
        test_dataset = SeqDataset(dataset_args['data_path'], dataset_args['kpt_path'], img_loader, mask_loader, img_tf, kpt_tf, seq_test_json)
        test_batch_sampler = DynamicBatchSampler(len(test_dataset), **dataset_args['test_batch_sampler'])
        test_dataloader = DataLoader(test_dataset, batch_sampler=test_batch_sampler, **dataset_args['test_dataloader'])
        return test_dataloader
    
    else:
        raise NotImplementedError("Unknown phase:{}".format(phase))

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