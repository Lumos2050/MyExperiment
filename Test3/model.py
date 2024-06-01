import torch.nn as nn
from Spatial_Channel import DAT
from edge_sv import SqueezeBodyEdge

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
            drop_rate=opt['DAT_options']['drop_rate'],#######################
            attn_drop_rate=opt['DAT_options']['attn_drop_rate']##################
        )
        
        self.edge_sv = SqueezeBodyEdge(
            inplane=opt['edge_sv_options']['inplane']
        )

    def forward(self, msf, fu, pan):  
        msf = msf.float()
        fu = fu.float()
        pan = pan.float()

        x, msf_init, pan_init = self.DAT(msf, fu, pan)  
        seg_flow_warp, seg_edge, mse_msf, mse_pan, cls_prd= self.edge_sv(x, msf_init, pan_init)
        #print("edge_sv后的output形状", cls_prd.shape)
        return cls_prd, mse_msf, mse_pan
    
    def cuda(self):
        self.DAT.cuda()
        self.edge_sv.cuda()
        return self
