import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
from torchvision.models.mobilenetv3 import mobilenet_v3_large, MobileNet_V3_Large_Weights
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torch.nn import functional as F
from collections import OrderedDict
from typing import Dict
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from models.ConvRNN import CGRU_cell

def get_deeplabv3(out_chan:int=2):
    model = deeplabv3_mobilenet_v3_large(aux_loss=False, num_classes=out_chan)
    return model

class GRUDeepLabV3(nn.Module):
    def __init__(self, hidden_chan:int, out_chan:int, k_size:int=3, group_norm_size:int=32):
        super().__init__()
        backbone = mobilenet_v3_large(parameters=MobileNet_V3_Large_Weights.DEFAULT).features
        stage_indices = [0] + [i for i, b in enumerate(backbone) if getattr(b, "_is_cn", False)] + [len(backbone) - 1]
        out_pos = stage_indices[-1]  # use C5 which has output_stride = 16
        out_inplanes = backbone[out_pos].out_channels
        self.backbone = backbone
        self.gru_cell = CGRU_cell(out_inplanes, k_size=k_size, num_features=hidden_chan, group_norm_size=group_norm_size)
        self.decoder = DeepLabHead(hidden_chan, 2)

    def forward(self, x:torch.Tensor):
        feat = self.backbone(x)
        feat = self.gru_cell(feat)
        print('before:{}'.format(feat.shape))
        feat = self.decoder(feat)
        out = F.interpolate(feat, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return out
    
    def reset_state(self):
        self.gru_cell.hidden_state = None

if __name__ == "__main__":
    # model = get_deeplabv3().cuda()
    # img = torch.rand(2,3,480,640).cuda()
    # out = model(img)
    # print(out.keys())
    # print("input shape:{}, out shape:{}".format(img.shape, out['out'].shape))
    # model = mobilenet_v3_large(MobileNet_V3_Large_Weights.DEFAULT).cuda()
    # feat_extractor = model.features
    # img = torch.rand(2,3,480,640).cuda()
    # feat = feat_extractor(img)
    # print(feat.shape) # (2, 960, 15, 20)
    model = GRUDeepLabV3(960, 2).cuda()
    # model = get_deeplabv3(2).cuda()
    img = torch.rand(2,3,480,640).cuda()
    feat = model(img)
    print(feat.shape)