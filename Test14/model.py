import torch.nn as nn
from Spatial_Channel import DAT
from edge_sv import SqueezeBodyEdge
from Diff_3 import diff3
from Restnet import ResNet18
class Total_model(nn.Module):
    def __init__(self, opt):
        super(Total_model, self).__init__()
        self.DAT = DAT(
            in_chans_4=opt['DAT_options']['in_chans_4'],
            in_chans_1=opt['DAT_options']['in_chans_1'],
            img_size=opt['DAT_options']['img_size'],
            embed_dim=opt['DAT_options']['embed_dim'],
            num_heads=opt['DAT_options']['num_heads'],
            expansion_factor=opt['DAT_options']['expansion_factor'],
            split_size=opt['DAT_options']['split_size'],
            proj_drop_rate=opt['DAT_options']['proj_drop_rate'],
            attn_drop_rate=opt['DAT_options']['attn_drop_rate'],
            drop_paths=opt['DAT_options']['drop_paths']
        )


    def forward(self, msf, pan):  
        msf = msf.float()
        pan = pan.float()

        MT_1, MP_1, PT_1 = diff3(msf, pan)
        cls_prd = self.DAT(MT_1, MP_1, PT_1)  
        return cls_prd
    def cuda(self):
        self.DAT.cuda()
        return self
