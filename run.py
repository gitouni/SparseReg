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

from models import get_deeplabv3
from datasets import make_dataset, tensor2img
from utils import plot_marker_displacement2, refresh_dir
# import torch.nn as nn
# import torch.nn.functional as F
import torch
from tqdm import tqdm
import yaml

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="cfg/test1.yml")
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--resume", type=str,default="saved_models/gelsight/model_200.pth")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="gelsight", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")
parser.add_argument("--lr", type=float, default=2e-4, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--weight_decay", type=float, default=0.001)
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
parser.add_argument(
    "--save_interval", type=int, default=1, help="interval between sampling of images from generators"
)
parser.add_argument("--checkpoint_interval", type=int, default=20, help="interval between model checkpoints")
opt = parser.parse_args()

config = yaml.load(open(opt.config,'r'), yaml.SafeLoader)
phase = config['phase']
if phase == 'train':
    img_save_path = "images/%s" % opt.dataset_name
model_save_path = "saved_models/%s" % opt.dataset_name

cuda = True if torch.cuda.is_available() else False

if cuda:
    device = torch.device("cuda:0")
else:
    device = torch.device('cpu')

# Loss functions
criterion = torch.nn.MSELoss(reduction='sum').to(device)
model = get_deeplabv3(2).to(device)

# Optimizers
optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), weight_decay=opt.weight_decay)
scheduler = ExponentialLR(optimizer, gamma=0.98)

@torch.no_grad()
def val_epoch(dataloader:DataLoader, save_interval:int, gt_save_path:str, pred_save_path:str):
    save_cnt = 0
    model.eval()
    avg_loss = 0
    for i,(img, mask, gt_shift) in tqdm(enumerate(dataloader),total=len(dataloader),desc='inference'):
        img:torch.Tensor = img.to(device)
        mask:torch.Tensor = mask.to(device)
        gt_shift:torch.Tensor = gt_shift.to(device)
        pred_shift:torch.Tensor = model(img)['out']
        avg_loss += criterion(pred_shift*mask, gt_shift*mask) / mask.sum()
        if i % save_interval == 0:
            img_np = tensor2img(img)
            mask_np = mask.squeeze().cpu().detach().numpy()
            pred_shift_np = pred_shift.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
            gt_shift_np = gt_shift.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
            out_pred = plot_marker_displacement2(img_np, mask_np, pred_shift_np)
            out_gt = plot_marker_displacement2(img_np, mask_np, gt_shift_np)
            Image.fromarray(out_gt).save(os.path.join(gt_save_path, "%04d.png"%save_cnt))
            Image.fromarray(out_pred).save(os.path.join(pred_save_path, "%04d.png"%save_cnt))
        save_cnt += 1
    return avg_loss / (i+1)


# Configure dataloaders
if phase == 'train':
    if opt.epoch != 0:
        # Load pretrained models
        model.load_state_dict(torch.load("saved_models/%s/model_%d.pth" % (opt.dataset_name, opt.epoch), map_location='cpu'))
        os.makedirs(img_save_path, exist_ok=True)
        os.makedirs(model_save_path, exist_ok=True)
    else:
        refresh_dir(img_save_path)
        refresh_dir(model_save_path)
    train_dataset, val_dataset = make_dataset(config)

    dataloader = DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
        drop_last=True
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=opt.n_cpu,
        drop_last=False
    )

    

    # ----------
    #  Training
    # ----------

    prev_time = time.time()

    for epoch in range(opt.epoch, opt.n_epochs):
        for i,(img, mask, gt_shift) in enumerate(dataloader):
            optimizer.zero_grad()
            # Model inputs
            img:torch.Tensor = img.to(device)
            mask:torch.Tensor = mask.to(device)
            gt_shift:torch.Tensor = gt_shift.to(device)
            pred_shift:torch.Tensor = model(img)['out']
            loss = criterion(pred_shift*mask, gt_shift*mask) / mask.sum()
            loss.backward()

            optimizer.step()

            # --------------
            #  Log Progress
            # --------------

            # Determine approximate time left
            batches_done = epoch * len(dataloader) + i
            batches_left = opt.n_epochs * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [mse loss: %f] ETA: %s"
                % (
                    epoch,
                    opt.n_epochs,
                    i,
                    len(dataloader),
                    loss.item(),
                    time_left,
                )
            )

        if (opt.checkpoint_interval != -1 and (epoch+1) % opt.checkpoint_interval == 0) or epoch == opt.n_epochs - 1:
            # Save model checkpoints
            torch.save(model.state_dict(), "saved_models/%s/model_%d.pth" % (opt.dataset_name, epoch+1))
            gt_img_save_path = os.path.join(img_save_path,'%03d'%(epoch+1), 'gt')
            pred_img_save_path = os.path.join(img_save_path,'%03d'%(epoch+1), 'pred')
            refresh_dir(gt_img_save_path)
            refresh_dir(pred_img_save_path)
            val_pixel_loss = val_epoch(val_dataloader, opt.save_interval, gt_img_save_path, pred_img_save_path)
            print("Epoch [%03d] val_loss: %.4f"%(epoch+1, val_pixel_loss))
            model.train()

else:
    test_dataset = make_dataset(config)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=opt.n_cpu,
        drop_last=False
    )
    model.load_state_dict(torch.load(opt.resume, map_location='cpu'))
    img_save_path = "%s/%s" % (config['output_dir'], opt.dataset_name)
    refresh_dir(img_save_path)
    gt_img_save_path = os.path.join(img_save_path, 'gt')
    pred_img_save_path = os.path.join(img_save_path, 'pred')
    refresh_dir(gt_img_save_path)
    refresh_dir(pred_img_save_path)
    final_loss = val_epoch(test_dataloader, 1, gt_img_save_path, pred_img_save_path)
    print('final mse loss:{}'.format(final_loss))
