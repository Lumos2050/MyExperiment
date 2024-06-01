import torch.nn as nn
from Spatial_Channel import DAT
import torch.nn.functional as F
from edge_sv import SqueezeBodyEdge
import torch
class ResBlk(nn.Module):
    """
    resnet block
    """
    def __init__(self, ch_in, ch_out, stride=1):
        """
        :param ch_in:
        :param ch_out:
        """
        super(ResBlk, self).__init__()
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)

        self.extra = nn.Sequential()
        if ch_out != ch_in:
            # [b, ch_in, h, w] => [b, ch_out, h , w]
            self.extra = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=stride),
                nn.BatchNorm2d(ch_out)
            )

    def forward(self, x):
        """
        :param x: [b, ch, h, w]
        :return:
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # short cut
        # element_wise add:[b, ch_in, h, w] with [b, ch_out, h ,w]
        out = self.extra(x) + out

        return out

class ResNet18(nn.Module):

    def __init__(self, inplane):
        super(ResNet18, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(inplane, inplane, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(inplane)
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #followed 4 blocks
        self.blk1_1 = ResBlk(inplane, inplane, stride=1)
        self.blk2_1 = ResBlk(inplane, 128, stride=2)
        self.blk3_1 = ResBlk(128, 128, stride=1)
        self.blk4_1 = ResBlk(128, 256, stride=2)
        self.blk5_1 = ResBlk(256, 256, stride=1)
        self.blk6_1 = ResBlk(256, 512, stride=2)

        self.outlayer1 = nn.Linear(512, 256)
        self.outlayer2 = nn.Linear(256, 128)
        self.outlayer3 = nn.Linear(128, 11)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        :param x:#(B, 4, H, W)
        :return:
        """
        #print("input x", x.shape)
        x = F.relu(self.conv1(x))
        x = self.blk3_1(x)
        x = self.blk4_1(x)
        x = F.adaptive_avg_pool2d(x, [1, 1])


        #print('x', x.shape)

        #print(x.size())
        x = x.view(x.size()[0],  -1)
        #print('ssss', x.shape)
        x = F.relu(self.outlayer1(x))
        x = F.relu(self.outlayer2(x))
        x = F.relu(self.outlayer3(x))
        x = F.softmax(x, dim=1)
        #print('ssssssssss',x.shape)
        return x
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
        
        self.edge_sv = SqueezeBodyEdge(
            inplane=opt['edge_sv_options']['inplane']
        )
        self.fc = ResNet18(inplane=opt['edge_sv_options']['inplane'])
    def forward(self, msf, fu, pan):  
        msf = msf.float()
        fu = fu.float()
        pan = pan.float()

        x, msf_init, pan_init = self.DAT(msf, fu, pan)  
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x =x.to(device)

        #seg_flow_warp, seg_edge, mse_msf, mse_pan, cls_prd= self.edge_sv(x, msf_init, pan_init)
        #print("edge_sv后的output形状", cls_prd.shape)
        _, inplane, _, _ = x.shape
        cls_prd = self.fc(x)
        #return cls_prd, mse_msf, mse_pan
        return cls_prd
    
    def cuda(self):
        self.DAT.cuda()
        self.edge_sv.cuda()
        self.fc.cuda()
        return self
