import argparse
import os
from PIL import Image
import time
import datetime
import sys

# import torchvision.transforms as transforms
# from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR
# from torchvision import datasets
# from torch.autograd import Variable

from models.base import GRUDeepLabV3
from datasets import make_seq_dataloader, tensor2img
from utils.utils import plot_marker_displacement2, refresh_dir
# import torch.nn as nn
# import torch.nn.functional as F
import torch
from tqdm import trange
import yaml
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="cfg/test_seen.yml")
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--resume", type=str,default="checkpoint/model_40.pth")
parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
parser.add_argument("--warmup_epochs",type=int,default=20)
parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
parser.add_argument("--lr", type=float, default=5e-4, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.99, help="adam: decay of first order momentum of gradient")
parser.add_argument("--weight_decay", type=float, default=1e-3)
parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
parser.add_argument(
    "--save_interval", type=int, default=1, help="interval between sampling of images from generators"
)
parser.add_argument("--checkpoint_interval", type=int, default=20, help="interval between model checkpoints")
opt = parser.parse_args()

config = yaml.load(open(opt.config,'r'), yaml.SafeLoader)
phase = config['phase']
dataset_args = config['dataset']
model_args = config['model']
dataset_name = config['name']
clip_gradient = config['experiment']['clip_gradient']
if phase == 'train':
    img_save_path = "images/%s" % dataset_name
model_save_path = "saved_models/%s" % dataset_name

cuda = True if torch.cuda.is_available() else False

if cuda:
    device = torch.device("cuda:0")
else:
    device = torch.device('cpu')

# Loss functions
criterion = torch.nn.MSELoss(reduction='sum').to(device)
model = GRUDeepLabV3(**model_args).to(device)

# Optimizers
optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), weight_decay=opt.weight_decay)
scheduler = ExponentialLR(optimizer, gamma=config['experiment']['exp_gamma'])

@torch.inference_mode()
def val_epoch(dataloader:DataLoader, save_interval:int, gt_save_path:str, pred_save_path:str, scale:float):
    save_cnt = 0
    model.eval()
    avg_loss = 0
    tmeter = trange(len(dataloader), desc='inference')
    with tmeter:
        for i,(seq_img, seq_mask, seq_gt_shift) in enumerate(dataloader):
            # Model inputs
            seq_img:torch.Tensor = img.to(device)  # (B, K, 3, H, W)
            seq_mask:torch.Tensor = mask.to(device)  # (B, K, 1, H, W)
            seq_gt_shift:torch.Tensor = gt_shift.to(device)  # (B, K, 1, H, W)
            Nseq = img.shape[1]
            total_loss = 0
            # RNN training
            for i in range(Nseq):
                model.reset_state()
                img = seq_img[:,i,...]
                mask = seq_mask[:,i,...]
                gt_shift = seq_gt_shift[:,i,...]
                pred_shift:torch.Tensor = model(img)
                loss:torch.Tensor = criterion(pred_shift*mask, gt_shift*mask) / mask.sum()
                total_loss += loss
            total_loss /= Nseq
            avg_loss += total_loss.item()
            if i % save_interval == 0:  # draw the marker motion prediction on the last image
                img_np = tensor2img(img)
                mask_np = mask.squeeze().cpu().detach().numpy()
                pred_shift_np = pred_shift.squeeze(0).permute(1, 2, 0).cpu().detach().numpy() * scale
                gt_shift_np = gt_shift.squeeze(0).permute(1, 2, 0).cpu().detach().numpy() * scale
                out_pred = plot_marker_displacement2(img_np, mask_np, pred_shift_np)
                out_gt = plot_marker_displacement2(img_np, mask_np, gt_shift_np)
                Image.fromarray(out_gt).save(os.path.join(gt_save_path, "%04d.png"%save_cnt))
                Image.fromarray(out_pred).save(os.path.join(pred_save_path, "%04d.png"%save_cnt))
            save_cnt += 1
            tmeter.update(1)
    return avg_loss / (i+1)

@torch.inference_mode()
def test_epoch(dataloader:DataLoader, save_interval:int, gt_save_path:str, pred_save_path:str, pred_shift_save_path:str, scale:float):
    save_cnt = 0
    model.eval()
    avg_loss = 0
    tmeter = trange(len(dataloader), desc='inference')
    with tmeter:
        for i,(seq_img, seq_mask, seq_gt_shift) in enumerate(dataloader):
            # Model inputs
            seq_img:torch.Tensor = img.to(device)  # (B, K, 3, H, W)
            seq_mask:torch.Tensor = mask.to(device)  # (B, K, 1, H, W)
            seq_gt_shift:torch.Tensor = gt_shift.to(device)  # (B, K, 1, H, W)
            Nseq = img.shape[1]
            total_loss = 0
            # RNN training
            for i in range(Nseq):
                model.reset_state()
                img = seq_img[:,i,...]
                mask = seq_mask[:,i,...]
                gt_shift = seq_gt_shift[:,i,...]
                pred_shift:torch.Tensor = model(img)
                loss:torch.Tensor = criterion(pred_shift*mask, gt_shift*mask) / mask.sum()
                total_loss += loss
            total_loss /= Nseq
            avg_loss += total_loss.item()
            if i % save_interval == 0:  # draw the marker motion prediction on the last image
                img_np = tensor2img(img)
                mask_np = mask.squeeze().cpu().detach().numpy()  # H, W
                pred_shift_np = pred_shift.squeeze(0).permute(1, 2, 0).cpu().detach().numpy() * scale  # H, W, 2
                gt_shift_np = gt_shift.squeeze(0).permute(1, 2, 0).cpu().detach().numpy() * scale   # H, W, 2
                kpty, kptx = np.nonzero(mask_np)
                delta = pred_shift_np[kpty, kptx]
                np.savetxt(os.path.join(pred_shift_save_path, "%04d.txt"%save_cnt), np.hstack((kptx[:,None], kpty[:, None], delta)),fmt='%.4f')
                out_pred = plot_marker_displacement2(img_np, mask_np, pred_shift_np)
                out_gt = plot_marker_displacement2(img_np, mask_np, gt_shift_np)
                Image.fromarray(out_gt).save(os.path.join(gt_save_path, "%04d.png"%save_cnt))
                Image.fromarray(out_pred).save(os.path.join(pred_save_path, "%04d.png"%save_cnt))
            save_cnt += 1
            tmeter.update(1)
    return avg_loss / (i+1)

def train_epoch(train_dataloader:DataLoader, epoch:int):
    for i,(seq_img, seq_mask, seq_gt_shift) in enumerate(train_dataloader):
        optimizer.zero_grad()
        # Model inputs
        seq_img:torch.Tensor = img.to(device)  # (B, K, 3, H, W)
        seq_mask:torch.Tensor = mask.to(device)  # (B, K, 1, H, W)
        seq_gt_shift:torch.Tensor = gt_shift.to(device)  # (B, K, 1, H, W)
        Nseq = img.shape[1]
        total_loss = 0
        # RNN training
        for i in range(Nseq):
            model.reset_state()
            img = seq_img[:,i,...]
            mask = seq_mask[:,i,...]
            gt_shift = seq_gt_shift[:,i,...]
            pred_shift:torch.Tensor = model(img)
            loss:torch.Tensor = criterion(pred_shift*mask, gt_shift*mask) / mask.sum()
            total_loss += loss
        total_loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=clip_gradient)
        optimizer.step()

        # --------------
        #  Log Progress
        # --------------

        # Determine approximate time left
        batches_done = epoch * len(train_dataloader) + i
        batches_left = opt.n_epochs * len(train_dataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        # Print log
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [mse loss: %f] [lr: %f] ETA: %s"
            % (
                epoch,
                opt.n_epochs,
                i,
                len(train_dataloader),
                loss.item(),
                optimizer.param_groups[0]['lr'],
                time_left,
            )
        )

        

# Configure dataloaders
if phase == 'train':
    if opt.epoch != 0:
        # Load pretrained models
        model.load_state_dict(torch.load("saved_models/%s/model_%d.pth" % (dataset_name, opt.epoch), map_location='cpu'))
        os.makedirs(img_save_path, exist_ok=True)
        os.makedirs(model_save_path, exist_ok=True)
    else:
        refresh_dir(img_save_path)
        refresh_dir(model_save_path)
    train_dataloader, val_dataloader = make_seq_dataloader(config)
    prev_time = time.time()
    model.train()
    for epoch in range(opt.epoch, opt.n_epochs):
        train_epoch(train_dataloader, epoch)
        if (opt.checkpoint_interval != -1 and (epoch+1) % opt.checkpoint_interval == 0) or epoch == opt.n_epochs - 1:
            # Save model checkpoints
            torch.save(model.state_dict(), "saved_models/%s/model_%d.pth" % (dataset_name, epoch+1))
            gt_img_save_path = os.path.join(img_save_path,'%03d'%(epoch+1), 'gt')
            pred_img_save_path = os.path.join(img_save_path,'%03d'%(epoch+1), 'pred')
            refresh_dir(gt_img_save_path)
            refresh_dir(pred_img_save_path)
            val_pixel_loss = val_epoch(val_dataloader, opt.save_interval, gt_img_save_path, pred_img_save_path, dataset_args['scale'])
            print("Epoch [%03d] val_loss: %.4f"%(epoch+1, val_pixel_loss))
            model.train()

else:
    test_dataloader = make_seq_dataloader(config)
    model.load_state_dict(torch.load(opt.resume, map_location='cpu'))
    img_save_path = "%s/%s" % (config['output_dir'], dataset_name)
    refresh_dir(img_save_path)
    gt_img_save_path = os.path.join(img_save_path, 'gt')
    pred_img_save_path = os.path.join(img_save_path, 'pred')
    pred_shift_save_path = os.path.join(img_save_path, 'pred_shift')
    refresh_dir(gt_img_save_path)
    refresh_dir(pred_img_save_path)
    refresh_dir(pred_shift_save_path)
    final_loss = test_epoch(test_dataloader, 1, gt_img_save_path, pred_img_save_path,pred_shift_save_path, dataset_args['scale'])
    print('final mse loss:{}'.format(final_loss))
