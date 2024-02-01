import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large

def get_deeplabv3(out_chan:int=2):
    model = deeplabv3_mobilenet_v3_large(aux_loss=False, num_classes=out_chan)
    return model

if __name__ == "__main__":
    model = get_deeplabv3().cuda()
    img = torch.rand(2,3,480,640).cuda()
    out = model(img)
    print(out.keys())
    print("input shape:{}, out shape:{}".format(img.shape, out['out'].shape))