import torch
import cv2
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from Restnet import ResBlk
class SqueezeBodyEdge(nn.Module):
    """
    input: x,texture相同尺度
    output: final相同尺度
    """


    def __init__(self, inplane):
        """
        implementation of body generation part
        :param inplane:
        :param norm_layer:
        """
        super(SqueezeBodyEdge, self).__init__()
        #四倍降采样，还用的是深度可分离卷积
        self.flow_make = nn.Sequential(
            nn.Conv2d(inplane, 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True))
        self.fine_edge = ResBlk(inplane*2, inplane) 
        self.sigmoid_edge = nn.Sigmoid()
    def flow_warp(self, input, flow, size):
        out_h, out_w = size
        n, c, h, w = input.size()

        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(input).to(input.device)
        # new
        h_grid = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        w_gird = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((w_gird.unsqueeze(2), h_grid.unsqueeze(2)), 2)

        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(input, grid, align_corners=True)
        return output
    
    def forward(self, x, texture):
        """
        input: x:torch.Size([1, 4, H/2, W/2])
        x(B, 4, H/2, W/2), msf_init, pan_init(B, 1, H, W)
        """
        #print("边缘监督中的释放显存1")
        torch.cuda.empty_cache()


        size = x.size()[2:]#H和W32
        flow = self.flow_make(x)#(B, 2, H/2, W/2)
        seg_body = self.flow_warp(x, flow, size)#(B, 64, H/2, W/2)
        seg_edge = x - seg_body#(B, 64, H/2, W/2)

        #-----------------Enhancement--------------------------------------

        #print("seg_edge,weighted_iamge的数据类型", seg_edge.shape, weighted_image.shape)
        fine_edge = self.fine_edge(torch.cat([seg_edge, texture], dim = 1))#(B, 4, H/2, W/2)
        fine_edge = self.sigmoid_edge(fine_edge)
        final = seg_body + fine_edge#(B, 4, H/2, W/2)
        #print("边缘监督中的释放显存2")
        torch.cuda.empty_cache()
        return final

"""
if __name__ == '__main__':
    model = SqueezeBodyEdge(
        inplane = 48
    ).cuda().eval()
    x = torch.randn((5, 48, 64, 64),dtype=torch.float32).cuda()
    texture = torch.randn((5, 48, 64, 64),dtype=torch.float32).cuda()
    f= model(x, texture)
    print(f.shape)
"""

