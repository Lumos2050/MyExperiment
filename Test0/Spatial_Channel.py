import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from torch.nn import functional as F
from timm.models.layers import DropPath, trunc_normal_
from einops.layers.torch import Rearrange
from einops import rearrange

import math
import numpy as np

def img2windows(img, H_sp, W_sp):
    """
    Input: Image (B, C, H, W)
    Output: Window Partition (B', N, C)
    """
    B, C, H, W = img.shape
    img_reshape = img.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)#(B, H // H_sp, W // W_sp, H_sp,W_sp, B)
    img_perm = img_reshape.permute(0, 2, 4, 3, 5, 1).contiguous().reshape(-1, H_sp* W_sp, C)
    return img_perm


def windows2img(img_splits_hw, H_sp, W_sp, H, W):
    """
    Input: Window Partition (B', N, C)
    Output: Image (B, H, W, C)
    """
    B = int(img_splits_hw.shape[0] / (H * W / H_sp / W_sp))

    img = img_splits_hw.view(B, H // H_sp, W // W_sp, H_sp, W_sp, -1)
    img = img.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return img


class DynamicPosBias(nn.Module):
    # The implementation builds on Crossformer code https://github.com/cheerss/CrossFormer/blob/main/models/crossformer.py
    """ Dynamic Relative Position Bias.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        residual (bool):  If True, use residual strage to connect conv.
    """
    def __init__(self, dim, num_heads, residual):
        super().__init__()
        self.residual = residual
        self.num_heads = num_heads
        self.pos_dim = dim // 4 if dim >= 4 else 1
        self.pos_proj = nn.Linear(2, self.pos_dim)
        self.pos1 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim),
        )
        self.pos2 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim)
        )
        self.pos3 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.num_heads)
        )
    def forward(self, biases):
        if self.residual:
            pos = self.pos_proj(biases) # 2Gh-1 * 2Gw-1, heads
            pos = pos + self.pos1(pos)
            pos = pos + self.pos2(pos)
            pos = self.pos3(pos)
        else:
            pos = self.pos3(self.pos2(self.pos1(self.pos_proj(biases))))
        return pos
    

    #对标的是Spatial_Attention, 不要mask，idx这里默认都是0
class Process_Spatial_Attention(nn.Module):
    def __init__(self, dim, idx = 0, split_size=[8,8], dim_out = None , num_heads=6, attn_drop=0., proj_drop=0., qk_scale=None, position_bias=True):
        super().__init__()

        self.dim = dim
        self.dim_out = dim_out or dim
        self.split_size = split_size
        self.num_heads = num_heads
        self.idx = idx
        self.position_bias = position_bias
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv1 = nn.Linear(dim*4 , dim * 12, bias=False)
        self.qkv2 = nn.Linear(dim, dim * 3, bias=False)

        if idx == 0:
            H_sp, W_sp = self.split_size[0], self.split_size[1]
        elif idx == 1:
            W_sp, H_sp = self.split_size[0], self.split_size[1]
        else:
            print ("ERROR MODE", idx)
            exit(0)
        self.H_sp = H_sp
        self.W_sp = W_sp
        self.change_fu = nn.Conv2d(720, 180, 3, 1, 1)

        #!self.pos是位置编码！
        if self.position_bias:
            self.pos = DynamicPosBias(1, self.num_heads, residual=False)
            # generate mother-set
            position_bias_h = torch.arange(1 - self.H_sp, self.H_sp)
            position_bias_w = torch.arange(1 - self.W_sp, self.W_sp)
            biases = torch.stack(torch.meshgrid([position_bias_h, position_bias_w]))
            biases = biases.flatten(1).transpose(0, 1).contiguous().float()
            self.register_buffer('rpe_biases', biases)

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(self.H_sp)
            coords_w = torch.arange(self.W_sp)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += self.H_sp - 1
            relative_coords[:, :, 1] += self.W_sp - 1
            relative_coords[:, :, 0] *= 2 * self.W_sp - 1
            relative_position_index = relative_coords.sum(-1)
            self.register_buffer('relative_position_index', relative_position_index)

        self.attn_drop = nn.Dropout(attn_drop)

    def im2win(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(-2,-1).contiguous().view(B, C, H, W)
        x = img2windows(x, self.H_sp, self.W_sp)
        x = x.reshape(-1, self.H_sp* self.W_sp, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        return x

    def forward(self, x, y, H, W):
        """
        x是fu(B,L,720), y是pan(B,L,180通道)
        Input: qkv: (B, 3*L, C),torch.Size([1, 4096, 180]),  H, W, mask: (B, N, N), N is the window size
        Output: x: (B, H*W, C)
        """
        B, L, C1 = x.shape
        B, L, C = y.shape
        assert L == H * W, "flatten img_tokens has wrong size"
        #print("qkv1", x.shape) #torch.Size([1, 4096, 720])
        #print("qkv2", y.shape) #torch.Size([1, 4096, 180])
        #print("qkv(x)", self.qkv(x).shape)#torch.Size([1, 4096, 540])
        qkv1 = self.qkv1(x).reshape(B, -1, 3, C1).permute(2, 0, 1, 3) # 3, B, HW, C
        qkv2 = self.qkv2(y).reshape(B, -1, 3, C).permute(2, 0, 1, 3) # 3, B, HW, C    
        #print("qkv1", qkv1.shape) #qkv1 torch.Size([3, 1, 4096, 720])   
        #print("qkv2", qkv2.shape) #qkv2 torch.Size([3, 1, 4096, 180])
        # V without partition
        v2 = qkv2[2].transpose(-2,-1).contiguous().view(B, C, H, W)
        k2, v2 = qkv2[1], qkv2[2]
        # image padding
        #这个填充的目的是确保输入张量的高度和宽度都可以被 max_split_size 整除，以便后续的处理能够进行有效地分块操作
        max_split_size = max(self.split_size[0], self.split_size[1])
        pad_l = pad_t = 0
        pad_r = (max_split_size - W % max_split_size) % max_split_size
        pad_b = (max_split_size - H % max_split_size) % max_split_size

        qkv1 = qkv1.reshape(3*B, H, W, C1).permute(0, 3, 1, 2) # 3B C H W
        qkv1 = F.pad(qkv1, (pad_l, pad_r, pad_t, pad_b)).reshape(3, B, C1, -1).transpose(-2, -1) # l r t b
        qkv2 = qkv2.reshape(3*B, H, W, C).permute(0, 3, 1, 2) # 3B C H W
        qkv2 = F.pad(qkv2, (pad_l, pad_r, pad_t, pad_b)).reshape(3, B, C, -1).transpose(-2, -1) # l r t b
        _H = pad_b + H
        _W = pad_r + W
        _L = _H * _W
        """

        else:
            x1 = self.attns[0](qkv[:,:,:,:C//2], _H, _W)[:, :H, :W, :].reshape(B, L, C//2)
            x2 = self.attns[1](qkv[:,:,:,C//2:], _H, _W)[:, :H, :W, :].reshape(B, L, C//2)
            # attention output
            attened_x = torch.cat([x1,x2], dim=2)"""
        #3, B, H*W, C1
        #把fu的四个通道拆开
        qkv11 = qkv1[:, :, :, 0:180] 
        q11 = qkv11[0]
        qkv12 = qkv1[:, :, :, 180:360]
        q12 = qkv12[0]
        qkv13 = qkv1[:, :, :, 360:540]
        q13 = qkv13[0]
        qkv14 = qkv1[:, :, :, 540:720]
        q14 = qkv14[0]
        #print("q11的形状",q11.shape)#torch.Size([1, 4096, 180])

        
        # partition the q,k,v, image to window
        q11 = self.im2win(q11, H, W)
        q12 = self.im2win(q12, H, W)
        q13 = self.im2win(q13, H, W)
        q14 = self.im2win(q14, H, W)

        k2 = self.im2win(k2, H, W)
        v2 = self.im2win(v2, H, W)
        
        #print("Process_Spatial Attention释放显存")
        torch.cuda.empty_cache()
        q11 = q11 * self.scale
        q12 = q12 * self.scale
        q13 = q13 * self.scale
        q14 = q14 * self.scale

        attn1 = (q11 @ k2.transpose(-2, -1))  # B head N C @ B head C N --> B head N N
        attn2 = (q12 @ k2.transpose(-2, -1))  # B head N C @ B head C N --> B head N N
        attn3 = (q13 @ k2.transpose(-2, -1))  # B head N C @ B head C N --> B head N N
        attn4 = (q14 @ k2.transpose(-2, -1))  # B head N C @ B head C N --> B head N N


        #先q@k，再位置编码，@v
        # calculate drpe
        if self.position_bias:
            pos = self.pos(self.rpe_biases)
            # select position bias
            relative_position_bias = pos[self.relative_position_index.view(-1)].view(
                self.H_sp * self.W_sp, self.H_sp * self.W_sp, -1)
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
            attn1 = attn1 + relative_position_bias.unsqueeze(0)
            attn2 = attn2 + relative_position_bias.unsqueeze(0)
            attn3 = attn3 + relative_position_bias.unsqueeze(0)
            attn4 = attn4 + relative_position_bias.unsqueeze(0)

        N = attn1.shape[3]

        attn1 = nn.functional.softmax(attn1, dim=-1, dtype=attn1.dtype)
        attn1 = self.attn_drop(attn1)
        attn2 = nn.functional.softmax(attn2, dim=-1, dtype=attn1.dtype)
        attn2 = self.attn_drop(attn2)
        attn3 = nn.functional.softmax(attn3, dim=-1, dtype=attn1.dtype)
        attn3 = self.attn_drop(attn3)
        attn4 = nn.functional.softmax(attn4, dim=-1, dtype=attn1.dtype)
        attn4 = self.attn_drop(attn4)

        z1 = (attn1 @ v2).transpose(1, 2).reshape(-1, self.H_sp* self.W_sp, C)  # B head N N @ B head N C
        z2 = (attn2 @ v2).transpose(1, 2).reshape(-1, self.H_sp* self.W_sp, C)  # B head N N @ B head N C
        z3 = (attn3 @ v2).transpose(1, 2).reshape(-1, self.H_sp* self.W_sp, C)  # B head N N @ B head N C
        z4 = (attn4 @ v2).transpose(1, 2).reshape(-1, self.H_sp* self.W_sp, C)  # B head N N @ B head N C


        # merge the window, window to image
        z1 = windows2img(z1, self.H_sp, self.W_sp, H, W)[:, :H, :W, :].reshape(B, L, C)  # B H' W' C
        z2 = windows2img(z2, self.H_sp, self.W_sp, H, W)[:, :H, :W, :].reshape(B, L, C)  # B H' W' C  # B H' W' C
        z3 = windows2img(z3, self.H_sp, self.W_sp, H, W)[:, :H, :W, :].reshape(B, L, C)  # B H' W' C  # B H' W' C
        z4 = windows2img(z4, self.H_sp, self.W_sp, H, W)[:, :H, :W, :].reshape(B, L, C)  # B H' W' C  # B H' W'
        z = torch.cat([z1,z2,z3,z4], dim=2)
        #print("PSA中的z形状", z.shape)#torch.Size([1, 4096, 720])
        z = z.transpose(-2,-1).contiguous().view(B, C1, H, W)
        #torch.Size([1, 720, 64, 64])
        z = self.change_fu(z)
        #torch.Size([1, 180, 64, 64])
        z = z.permute(0, 2, 3, 1).contiguous().view(B, L, C)
        #print("PSA中的z形状", z.shape)torch.Size([1, 4096, 180])
        return z
    

#idx 管理窗口0还是1
class Spatial_Attention(nn.Module):
    """ Spatial Window Self-Attention.
    It supports rectangle window (containing square window).
    Args:
        dim (int): Number of input channels.
        idx (int): The indentix of window. (0/1)
        split_size (tuple(int)): Height and Width of spatial window.
        dim_out (int | None): The dimension of the attention output. Default: None
        num_heads (int): Number of attention heads. Default: 6
        attn_drop (float): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float): Dropout ratio of output. Default: 0.0
        qk_scale (float | None): Override default qk scale of head_dim ** -0.5 if set
        position_bias (bool): The dynamic relative position bias. Default: True
        
空间窗口自注意力机制。
支持矩形窗口（包含正方形窗口）。
参数：
- dim（int）：输入通道数。
- idx（int）：窗口的索引。 (0/1)
- split_size（tuple(int)）：空间窗口的高度和宽度。
- dim_out（int | None）：注意力输出的维度。默认值：None
- num_heads（int）：注意力头的数量。默认值：6
- attn_drop（float）：注意力权重的 dropout 比率。默认值：0.0
- proj_drop（float）：输出的 dropout 比率。默认值：0.0
- qk_scale（float | None）：如果设置，则覆盖默认的 qk 缩放为 head_dim ** -0.5。
- position_bias（bool）：是否使用动态相对位置偏置。默认值：True
    """
    def __init__(self, dim, idx, split_size=[8,8], dim_out=None, num_heads=6, attn_drop=0., proj_drop=0., qk_scale=None, position_bias=True):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out or dim
        self.split_size = split_size
        self.num_heads = num_heads
        self.idx = idx
        self.position_bias = position_bias

        head_dim = dim // num_heads if dim >= num_heads else 1
        self.scale = qk_scale or head_dim ** -0.5

        if idx == 0:
            H_sp, W_sp = self.split_size[0], self.split_size[1]
        elif idx == 1:
            W_sp, H_sp = self.split_size[0], self.split_size[1]
        else:
            print ("ERROR MODE", idx)
            exit(0)
        self.H_sp = H_sp
        self.W_sp = W_sp
        #!self.pos是位置编码！
        if self.position_bias:
            self.pos = DynamicPosBias(self.dim // 4 if dim >= num_heads else 1, self.num_heads, residual=False)
            # generate mother-set
            position_bias_h = torch.arange(1 - self.H_sp, self.H_sp)
            position_bias_w = torch.arange(1 - self.W_sp, self.W_sp)
            biases = torch.stack(torch.meshgrid([position_bias_h, position_bias_w]))
            biases = biases.flatten(1).transpose(0, 1).contiguous().float()
            self.register_buffer('rpe_biases', biases)

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(self.H_sp)
            coords_w = torch.arange(self.W_sp)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += self.H_sp - 1
            relative_coords[:, :, 1] += self.W_sp - 1
            relative_coords[:, :, 0] *= 2 * self.W_sp - 1
            relative_position_index = relative_coords.sum(-1)
            self.register_buffer('relative_position_index', relative_position_index)

        self.attn_drop = nn.Dropout(attn_drop)

    def im2win(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(-2,-1).contiguous().view(B, C, H, W)
        x = img2windows(x, self.H_sp, self.W_sp)
        x = x.reshape(-1, self.H_sp* self.W_sp, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        return x

    def forward(self, qkv1,qkv2, H, W, mask=None):
        """
        qkv1是4通道,qkv2是4通道, 大小相同
        Input: qkv: (B, 3*L, C), H, W, mask: (B, N, N), N is the window size
        Output: x (B, H, W, C)
        """
        q1, v1 = qkv1[0], qkv1[2]
        k2 = qkv2[1]
        B, L, C = q1.shape   
        assert L == H * W, "flatten img_tokens has wrong size"

        # partition the q,k,v, image to window
        q1 = self.im2win(q1, H, W)
        k2 = self.im2win(k2, H, W)
        v1 = self.im2win(v1, H, W)

        q1 = q1 * self.scale
        attn = (q1 @ k2.transpose(-2, -1))  # B head N C @ B head C N --> B head N N
        #先q@k，再位置编码，@v
        # calculate drpe
        if self.position_bias:
            pos = self.pos(self.rpe_biases)
            # select position bias
            relative_position_bias = pos[self.relative_position_index.view(-1)].view(
                self.H_sp * self.W_sp, self.H_sp * self.W_sp, -1)
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
            attn = attn + relative_position_bias.unsqueeze(0)

        N = attn.shape[3]

        # use mask for shift window
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = nn.functional.softmax(attn, dim=-1, dtype=attn.dtype)
        attn = self.attn_drop(attn)

        z = (attn @ v1)
        z = z.transpose(1, 2).reshape(-1, self.H_sp* self.W_sp, C)  # B head N N @ B head N C

        # merge the window, window to image
        z = windows2img(z, self.H_sp, self.W_sp, H, W)  # B H' W' C

        return z


class Adaptive_Channel_Attention(nn.Module):
    # The implementation builds on XCiT code https://github.com/facebookresearch/xcit
    """ Adaptive Channel Self-Attention
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads. Default: 6
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None): Override default qk scale of head_dim ** -0.5 if set.
        attn_drop (float): Attention dropout rate. Default: 0.0
        drop_path (float): Stochastic depth rate. Default: 0.0
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        #nn.Parameter() 是 PyTorch 提供的一个类，用于标记一个张量是模型的可学习参数。
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.dwconv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1,groups=dim),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )
        self.channel_interaction = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 8, kernel_size=1),
            nn.BatchNorm2d(dim // 8),
            nn.GELU(),
            nn.Conv2d(dim // 8, dim, kernel_size=1),
        )
        self.spatial_interaction = nn.Sequential(
            nn.Conv2d(dim, dim // 16, kernel_size=1),
            nn.BatchNorm2d(dim // 16),
            nn.GELU(),
            nn.Conv2d(dim // 16, 1, kernel_size=1)
        )

    def forward(self, x, H, W):
        """
        Input: x: (B, H*W, C), H, W
        Output: x: (B, H*W, C)
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)

        v_ = v.reshape(B, C, N).contiguous().view(B, C, H, W)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # attention output
        attened_x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)

        # convolution output
        conv_x = self.dwconv(v_)

        #print("con_x的形状", conv_x.shape)
        # C-Map (before sigmoid)
        channel_map = self.channel_interaction(conv_x).permute(0, 2, 3, 1).contiguous().view(B, 1, C)
        #print("channel_map",channel_map.shape)
        # C-I
        attened_x = attened_x * torch.sigmoid(channel_map)#对应位置相乘，(B, 1, C)


    
        z = attened_x

        z = self.proj(z)
        z = self.proj_drop(z)

        return z
    

    #至此，在初始化部分只有Adaptive_Spatial_Attention初始化了mask

class Adaptive_Spatial_Attention(nn.Module):
    # The implementation builds on CAT code https://github.com/Zhengchen1999/CAT
    """ Adaptive Spatial Self-Attention
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads. Default: 6
        split_size (tuple(int)): Height and Width of spatial window.
        shift_size (tuple(int)): Shift size for spatial window.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None): Override default qk scale of head_dim ** -0.5 if set.
        drop (float): Dropout rate. Default: 0.0
        attn_drop (float): Attention dropout rate. Default: 0.0
        rg_idx (int): The indentix of Residual Group (RG)
        b_idx (int): The indentix of Block in each RG
    """
    def __init__(self, dim, num_heads, 
                 reso=64, split_size=[8,8], shift_size=[1,2], qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., mask = None):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.split_size = split_size
        self.shift_size = shift_size
        self.mask = mask
       # self.b_idx  = b_idx
       # self.rg_idx = rg_idx
        self.patches_resolution = reso
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        assert 0 <= self.shift_size[0] < self.split_size[0], "shift_size must in 0-split_size0"
        assert 0 <= self.shift_size[1] < self.split_size[1], "shift_size must in 0-split_size1"
     
        self.branch_num = 2

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)

        self.attns = nn.ModuleList([
                Spatial_Attention(
                    dim//2 if dim >= 4 else 1, idx = i,
                    split_size=split_size, num_heads=num_heads//2, dim_out=dim//2 if dim >= 4 else 1,
                    qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, position_bias=True)
                for i in range(self.branch_num)])
        
        if self.mask is not None:
            attn_mask = self.calculate_mask(self.patches_resolution, self.patches_resolution)
            self.register_buffer("attn_mask_0", attn_mask[0])
            self.register_buffer("attn_mask_1", attn_mask[1])
        else:
            attn_mask = None
            self.register_buffer("attn_mask_0", None)
            self.register_buffer("attn_mask_1", None)       
             
        self.dwconv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1,groups=dim),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )
        self.spatial_interaction = nn.Sequential(
            nn.Conv2d(dim, dim // 2, kernel_size=1),
            nn.BatchNorm2d(dim // 2),
            nn.GELU(),
            nn.Conv2d(dim // 2, 1, kernel_size=1)
        )
    def calculate_mask(self, H, W):
        # The implementation builds on Swin Transformer code https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py
        # calculate attention mask for shift window
        img_mask_0 = torch.zeros((1, H, W, 1))  # 1 H W 1 idx=0
        img_mask_1 = torch.zeros((1, H, W, 1))  # 1 H W 1 idx=1
        h_slices_0 = (slice(0, -self.split_size[0]),
                    slice(-self.split_size[0], -self.shift_size[0]),
                    slice(-self.shift_size[0], None))
        w_slices_0 = (slice(0, -self.split_size[1]),
                    slice(-self.split_size[1], -self.shift_size[1]),
                    slice(-self.shift_size[1], None))

        h_slices_1 = (slice(0, -self.split_size[1]),
                    slice(-self.split_size[1], -self.shift_size[1]),
                    slice(-self.shift_size[1], None))
        w_slices_1 = (slice(0, -self.split_size[0]),
                    slice(-self.split_size[0], -self.shift_size[0]),
                    slice(-self.shift_size[0], None))
        cnt = 0
        for h in h_slices_0:
            for w in w_slices_0:
                img_mask_0[:, h, w, :] = cnt
                cnt += 1
        cnt = 0
        for h in h_slices_1:
            for w in w_slices_1:
                img_mask_1[:, h, w, :] = cnt
                cnt += 1

        # calculate mask for window-0
        img_mask_0 = img_mask_0.view(1, H // self.split_size[0], self.split_size[0], W // self.split_size[1], self.split_size[1], 1)
        img_mask_0 = img_mask_0.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, self.split_size[0], self.split_size[1], 1) # nW, sw[0], sw[1], 1
        mask_windows_0 = img_mask_0.view(-1, self.split_size[0] * self.split_size[1])
        attn_mask_0 = mask_windows_0.unsqueeze(1) - mask_windows_0.unsqueeze(2)
        attn_mask_0 = attn_mask_0.masked_fill(attn_mask_0 != 0, float(-100.0)).masked_fill(attn_mask_0 == 0, float(0.0))

        # calculate mask for window-1
        img_mask_1 = img_mask_1.view(1, H // self.split_size[1], self.split_size[1], W // self.split_size[0], self.split_size[0], 1)
        img_mask_1 = img_mask_1.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, self.split_size[1], self.split_size[0], 1) # nW, sw[1], sw[0], 1
        mask_windows_1 = img_mask_1.view(-1, self.split_size[1] * self.split_size[0])
        attn_mask_1 = mask_windows_1.unsqueeze(1) - mask_windows_1.unsqueeze(2)
        attn_mask_1 = attn_mask_1.masked_fill(attn_mask_1 != 0, float(-100.0)).masked_fill(attn_mask_1 == 0, float(0.0))

        return attn_mask_0, attn_mask_1

    def forward(self, x, y, H, W):
        """
        Input: x: (B, H*W, C), y: (B, H*W, C), H, W
        Output: z: (B, H*W, C)
        """
        B, L, C = x.shape
        assert L == H * W, "flatten img_tokens has wrong size"
        #print("ASA中的input形状", x.shape, y.shape)
        qkv1 = self.qkv(x).reshape(B, -1, 3, C).permute(2, 0, 1, 3) # 3, B, HW, C
        qkv2 = self.qkv(y).reshape(B, -1, 3, C).permute(2, 0, 1, 3) # 3, B, HW, C
        # V without partition
        #print("ASA中的qkv2形状", qkv2.shape)
        v2 = qkv2[2].transpose(-2,-1).contiguous().view(B, C, H, W)

        # image padding
        max_split_size = max(self.split_size[0], self.split_size[1])
        pad_l = pad_t = 0
        pad_r = (max_split_size - W % max_split_size) % max_split_size
        pad_b = (max_split_size - H % max_split_size) % max_split_size

        qkv1 = qkv1.reshape(3*B, H, W, C).permute(0, 3, 1, 2) # 3B C H W
        qkv1 = F.pad(qkv1, (pad_l, pad_r, pad_t, pad_b)).reshape(3, B, C, -1).transpose(-2, -1) # l r t b
        qkv2 = qkv2.reshape(3*B, H, W, C).permute(0, 3, 1, 2) # 3B C H W
        qkv2 = F.pad(qkv2, (pad_l, pad_r, pad_t, pad_b)).reshape(3, B, C, -1).transpose(-2, -1) # l r t b
        _H = pad_b + H
        _W = pad_r + W
        _L = _H * _W
        # window-0 and window-1 on split channels [C/2, C/2]; for square windows (e.g., 8x8), window-0 and window-1 can be merged
        # shift in block: (0, 4, 8, ...), (2, 6, 10, ...), (0, 4, 8, ...), (2, 6, 10, ...), ...
        if self.mask is not None: 
            qkv1 = qkv1.view(3, B, _H, _W, C)
            qkv2 = qkv2.view(3, B, _H, _W, C)
            qkv_01 = torch.roll(qkv1[:,:,:,:,:C//2], shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(2, 3))
            qkv_01 = qkv_01.view(3, B, _L, C//2)
            qkv_02 = torch.roll(qkv2[:,:,:,:,:C//2], shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(2, 3))
            qkv_02 = qkv_02.view(3, B, _L, C//2)
            qkv_11 = torch.roll(qkv1[:,:,:,:,C//2:], shifts=(-self.shift_size[1], -self.shift_size[0]), dims=(2, 3))
            qkv_11 = qkv_11.view(3, B, _L, C//2)
            qkv_12 = torch.roll(qkv2[:,:,:,:,C//2:], shifts=(-self.shift_size[1], -self.shift_size[0]), dims=(2, 3))
            qkv_12 = qkv_12.view(3, B, _L, C//2)

            if self.patches_resolution != _H or self.patches_resolution != _W:
                mask_tmp = self.calculate_mask(_H, _W)
                x1_shift = self.attns[0](qkv_01, qkv_02, _H, _W, mask=mask_tmp[0].to(x.device))
                x2_shift = self.attns[1](qkv_11, qkv_12, _H, _W, mask=mask_tmp[1].to(x.device))
            else:
                x1_shift = self.attns[0](qkv_01, qkv_02,  _H, _W, mask=self.attn_mask_0)
                x2_shift = self.attns[1](qkv_11, qkv_12,  _H, _W, mask=self.attn_mask_1)

            x1 = torch.roll(x1_shift, shifts=(self.shift_size[0], self.shift_size[1]), dims=(1, 2))
            x2 = torch.roll(x2_shift, shifts=(self.shift_size[1], self.shift_size[0]), dims=(1, 2))
            x1 = x1[:, :H, :W, :].reshape(B, L, C//2)
            x2 = x2[:, :H, :W, :].reshape(B, L, C//2)
            # attention output
            attened_x = torch.cat([x1,x2], dim=2)

        else:
            x1 = self.attns[0](qkv1[:,:,:,:C//2], qkv2[:,:,:,:C//2], _H, _W)[:, :H, :W, :].reshape(B, L, C//2)
            x2 = self.attns[1](qkv1[:,:,:,C//2:], qkv2[:,:,:,C//2:],  _H, _W)[:, :H, :W, :].reshape(B, L, C//2)
            # attention output
            attened_x = torch.cat([x1,x2], dim=2)
        
        # convolution output
        conv_x = self.dwconv(v2)#v1(B, C, H, W)

        # Adaptive Interaction Module (AIM)

        # S-Map (before sigmoid)
        spatial_map = self.spatial_interaction(conv_x).permute(0, 2, 3, 1).contiguous().view(B, L, 1)

        # S-I
        attened_x = attened_x * torch.sigmoid(spatial_map)
        """
        # S-Map (before sigmoid)
        attention_reshape = attened_x.transpose(-2,-1).contiguous().view(B, C, H, W)
        spatial_map = self.spatial_interaction(attention_reshape)
        # S-I
        conv_x = torch.sigmoid(spatial_map) * conv_x
        conv_x = conv_x.permute(0, 2, 3, 1).contiguous().view(B, L, C)
        """
        z = attened_x 

        z = self.proj(z)
        z = self.proj_drop(z)

        return z
    

class FFN(nn.Module):
    """ 
    前向传播模块，会被整合到下面两个里面
    Spatial-Gate Feed-Fsasorward Network.
    Args:
        in_features (int): Number of input channels.
        hidden_features (int | None): Number of hidden channels. Default: None
        out_features (int | None): Number of output channels. Default: None
        act_layer (nn.Module): Activation layer. Default: nn.GELU
        drop (float): Dropout rate. Default: 0.0
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        #self.sg = SpatialGate(hidden_features//2)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        """
        Input: x: (B, H*W, C), H, W
        Output: x: (B, H*W, C)
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)

        x = self.fc2(x)
        x = self.drop(x)
        return x
    

    #初始化了mask，设置idx=0
#self, dim, idx, split_size=[8,8], dim_out = None , num_heads=6, attn_drop=0., proj_drop=0., qk_scale=None, position_bias=True
class Process_Spatial_FFN(nn.Module):
    def __init__(self, dim, num_heads, reso=64, split_size=[2,4],shift_size=[1,2], expansion_factor=4., qkv_bias=False, qk_scale=None, drop=0.,
                 attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()

        self.norm0 = norm_layer(dim*4)
        self.norm1 = norm_layer(dim)
        #Proscess Spatial cross transformer block
        self.attn1 = Process_Spatial_Attention(
            dim, idx=0, split_size=split_size, num_heads=num_heads, attn_drop=attn_drop, proj_drop=drop, qk_scale=qk_scale, position_bias=True)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        ffn_hidden_dim = int(dim * expansion_factor)
        self.ffn =FFN(in_features=dim, hidden_features=ffn_hidden_dim, out_features=dim, act_layer=act_layer)
        self.norm2 = norm_layer(dim)

    def forward(self, x, y, x_size):
        """
        fu(B,H*W,720), pan(B,H*W,180)
        Input: x: (B, H*W, C), x_size: (H, W)
        Output: x: (B, H*W, C)
        """        
        H , W = x_size
        #print("PSF中的输入fu", x.shape)
        #print("PSF中的输入pan", y.shape) 
        #PSF中的输入fu torch.Size([1, 4096, 720])
        #PSF中的输入pan torch.Size([1, 4096, 180])
        z = self.drop_path(self.attn1(self.norm0(x),self.norm1(y), H, W))
        z = z + self.drop_path(self.ffn(self.norm2(z) , H, W))

        return z
    

class Spatial_Channel_FFN(nn.Module):
    def __init__(self, dim, num_heads, reso=64, split_size=[2,4],shift_size=[1,2], expansion_factor=4., qkv_bias=False, qk_scale=None, drop=0.,
                 attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, mask=None):
        super().__init__()

        self.norm1 = norm_layer(dim)
        #Spatial cross transformer block
        self.attn1 = Adaptive_Spatial_Attention(
            dim, num_heads=num_heads, reso=reso, split_size=split_size, shift_size=shift_size, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop, attn_drop=attn_drop, mask=None)

        #Channel cross transformer block
        self.attn2 = Adaptive_Channel_Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
            proj_drop=drop)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        ffn_hidden_dim = int(dim * expansion_factor)
        self.ffn =FFN(in_features=dim, hidden_features=ffn_hidden_dim, out_features=dim, act_layer=act_layer)
        self.norm2 = norm_layer(dim)

    def forward(self, x, y, x_size):
        """
        Input: x: (B, H*W, C), x_size: (H, W)
        Output: x: (B, H*W, C)
        """        
        H , W = x_size
        z = self.drop_path(self.attn1(self.norm1(x),self.norm1(y), H, W))
        z = self.drop_path(self.attn2(self.norm1(z), H, W))
        z = z + self.drop_path(self.ffn(z, H, W))#只在feedforward时候加残差快

        return z
    

class ResidualGroup(nn.Module):
    """ ResidualGroup
    Args:
        dim (int): Number of input channels.
        reso (int): Input resolution.
        num_heads (int): Number of attention heads.
        split_size (tuple(int)): Height and Width of spatial window.
        expansion_factor (float): Ratio of ffn hidden dim to embedding dim.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop (float): Dropout rate. Default: 0
        attn_drop(float): Attention dropout rate. Default: 0
        drop_paths (float | None): Stochastic depth rate.
        act_layer (nn.Module): Activation layer. Default: nn.GELU
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm
        depth (int): Number of dual aggregation Transformer blocks in residual group.
        use_chk (bool): Whether to use checkpointing to save memory.
        resi_connection: The convolutional block before residual connection. '1conv'/'3conv'
    """
    def __init__(   self,
                    dim,
                    reso,
                    num_heads,
                    split_size=[2,4],
                    expansion_factor=4.,
                    qkv_bias=False,
                    qk_scale=None,
                    drop=0.,
                    attn_drop=0.,
                    drop_paths=[0.1, 0.08, 0.06, 0.04, 0.02, 0.0],
                    act_layer=nn.GELU,
                    norm_layer=nn.LayerNorm,
                    use_chk=False,
                    ):
        super().__init__()
        self.use_chk = use_chk
        self.reso = reso

        self.blocks1 = Process_Spatial_FFN(
            dim=dim,
            num_heads=num_heads,
            reso = reso,
            split_size = split_size,
            shift_size = [split_size[0]//2, split_size[1]//2],
            expansion_factor=expansion_factor,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop,
            attn_drop=attn_drop,
            act_layer=act_layer,
            norm_layer=norm_layer,
            )


        self.blocks2 = nn.ModuleList([
        Spatial_Channel_FFN(
            dim=dim,
            num_heads=num_heads,
            reso = reso,
            split_size = split_size,
            shift_size = [split_size[0]//2, split_size[1]//2],
            expansion_factor=expansion_factor,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_paths[i],
            act_layer=act_layer,
            norm_layer=norm_layer,
            mask=None if i % 2 == 0 else 1
            )for i in range(5)])
        self.change_fu = nn.Conv2d(dim*4, dim, 3, 1, 1)
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
    def forward(self, msf, fu, pan, x_size):
        """
        Input: x: (B, H*W, C), x_size: (H, W), fu(B,H*W,720)
        Output: x: (B, H*W, C)在这里x是输出（没有完全改）
        """
        H, W = x_size
        B, L, C = msf.shape
        x1_size = int(H/2), int(W/2)
        L1 = int(H * W / 4)

        z1 = self.blocks1(fu, pan, x_size)
        blk1 = self.blocks2[0]
        blk2 = self.blocks2[1]
        blk3 = self.blocks2[2]
        blk4 = self.blocks2[3]
        blk5 = self.blocks2[4]
        fu = rearrange(fu, "b (h w) c -> b c h w", h=H, w=W)
        fu = self.change_fu(fu)
        fu = rearrange(fu, "b c h w -> b (h w) c", h=H, w=W)
        if self.use_chk:
            z2 = checkpoint.checkpoint(blk1, z2, x_size)
        else:
            z2 = blk1(z1, msf, x_size)
            z3 = blk2(z2, fu, x_size)
        #print("z3的形状是什么", z3.shape)#torch.Size([1, 4096, 180])



        z3 = z3.transpose(-2,-1).contiguous().view(B, C, H, W)
        #z3 = cv2.resize(z3, dsize=None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
        z3 = F.interpolate(z3, size=(int(H/2), int(W/2)), mode='bilinear', align_corners=False)
        z3 = z3.permute(0, 2, 3, 1).contiguous().view(B, L1, C)

        z2 = z2.transpose(-2,-1).contiguous().view(B, C, H, W)
        #z3 = cv2.resize(z3, dsize=None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
        z2 = F.interpolate(z2, size=(int(H/2), int(W/2)), mode='bilinear', align_corners=False)
        z2 = z2.permute(0, 2, 3, 1).contiguous().view(B, L1, C)

        z1 = z1.transpose(-2,-1).contiguous().view(B, C, H, W)
        #z3 = cv2.resize(z3, dsize=None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
        z1 = F.interpolate(z1, size=(int(H/2), int(W/2)), mode='bilinear', align_corners=False)
        z1 = z1.permute(0, 2, 3, 1).contiguous().view(B, L1, C)

        if self.use_chk:
            z4 = checkpoint.checkpoint(blk3, z4, x1_size)
        else:
            z4 = blk3(z3, z1, x1_size)
            z5 = blk4(z4, z2, x1_size)
            z6 = blk5(z5, z3, x1_size)
                

        z6 = rearrange(z6, "b (h w) c -> b c h w", h=int(H/2), w=int(W/2))
        z6 = self.conv(z6)
        z6 = rearrange(z6, "b c h w -> b (h w) c")

        return z6
    

    #@ARCH_REGISTRY.register()
class DAT(nn.Module):
    """ Dual Aggregation Transformer
    Args:
        img_size (int): Input image size. Default: 64
        in_chans (int): Number of input image channels. Default: 3
        embed_dim (int): Patch embedding dimension. Default: 180
        depths (tuple(int)): Depth of each residual group (number of DATB in each RG).
        split_size (tuple(int)): Height and Width of spatial window.
        num_heads (tuple(int)): Number of attention heads in different residual groups.
        expansion_factor (float): Ratio of ffn hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        act_layer (nn.Module): Activation layer. Default: nn.GELU
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm
        use_chk (bool): Whether to use checkpointing to save memory.
        upscale: Upscale factor. 2/3/4 for image SR
        img_range: Image range. 1. or 255.
        resi_connection: The convolutional block before residual connection. '1conv'/'3conv'
    """
    def __init__(self,
                img_size=64,
                in_chans_4=4,
                in_chans_1=1,
                embed_dim = 180,
                split_size=[2,4],
                num_heads= 6,
                expansion_factor=4.,
                qkv_bias=True,
                qk_scale=None,
                drop_rate=0.,
                attn_drop_rate=0.,
                drop_path_rate=0.1,
                act_layer=nn.GELU,
                norm_layer=nn.LayerNorm,
                use_chk=False,
                upscale=2,
                img_range=1.,
                **kwargs):
        super().__init__()

        num_in_ch_4 = in_chans_4
        num_in_ch_1 = in_chans_1
        num_out_ch = in_chans_4
        num_feat = 64
        self.img_range = img_range


        self.upscale = upscale

        # ------------------------- 1, Shallow Feature Extraction ------------------------- #
        self.conv_first_4 = nn.Conv2d(num_in_ch_4, embed_dim, 3, 1, 1)
        self.conv_first_1 = nn.Conv2d(num_in_ch_1, embed_dim, 3, 1, 1)
        self.conv_init_edge = nn.Conv2d(embed_dim, 1, 3, 1, 1)

        # ------------------------- 2, Deep Feature Extraction ------------------------- #
        #self.num_layers = len(depth)
        self.use_chk = use_chk
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        #层归一化

        self.before_RG = nn.Sequential(
            Rearrange('b c h w -> b (h w) c'),
            nn.LayerNorm(embed_dim)
        )

        curr_dim = embed_dim

        self.layer = ResidualGroup(
            dim=embed_dim,
            num_heads=num_heads,
            reso=img_size,
            split_size=split_size,
            expansion_factor=expansion_factor,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_paths=[0.1, 0.08, 0.06, 0.04, 0.02, 0.0],#####################################################
            act_layer=act_layer,
            norm_layer=norm_layer,
            use_chk=use_chk,
            )

        self.norm = norm_layer(curr_dim)
        #残差连接:用成自己的,不是残差连接
        # build the last conv layer in deep feature extraction
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        self.conv_then = nn.Sequential(nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True))
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        # ------------------------- 3, Reconstruction delete------------------------- #

        self.apply(self._init_weights)
    #初始化模型的权重和偏置
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm, nn.InstanceNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    #让所有的处理都在residual里面就行
    def forward_features(self, msf, fu, pan):
        _, _, H, W = msf.shape
        x_size = [H, W]
        msf = self.before_RG(msf)#b c h w -> b (h w) c
        #把fu的四个通道拆开
        #print("here1", fu.shape)
        fu1 = fu[:, 0:180, :, :]
        fu2 = fu[:, 180:360, :, :]
        fu3 = fu[:, 360:540, :, :]
        fu4 = fu[:, 540:720, :, :]
        #print("here2", fu4.shape)
        #here1 torch.Size([1, 720, 64, 64])
        #here2 torch.Size([1, 180, 64, 64])
        fu1 = self.before_RG(fu1)#b c h w -> b (h w) c
        fu2 = self.before_RG(fu2)#b c h w -> b (h w) c
        fu3 = self.before_RG(fu3)#b c h w -> b (h w) c
        fu4 = self.before_RG(fu4)#b c h w -> b (h w) c
        fu = torch.cat((fu1, fu2, fu3, fu4), dim=2)
        pan = self.before_RG(pan)#b c h w -> b (h w) c
        ##核心关键！！
        x = self.layer(msf, fu, pan, x_size)
        #print("输出的原始图", x.shape)#输出的原始图 torch.Size([1, 1024, 180])
        x = self.norm(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=int(H/2), w=int(W/2))

        return x

    def forward(self, msf, fu, pan):
        """
        Input: msf fu: (B, 4, H, W) pan: (B, 1, H, W)
        output: x(B, 4, H/2, W/2), msf_init, pan_init(B, 1, H, W)
        
        """
        #print("msf,fu,pan的形状", msf.shape, fu.shape, pan.shape)
        #print("DAT:main中的msf和pan,后面是一个释放显存", msf.shape, pan.shape)
        torch.cuda.empty_cache()
        msf = self.conv_first_4(msf)
        msf_init = self.conv_init_edge(msf)#(B,1,H,W)
        fu1 = fu[:, 0:1, :, :]
        fu2 = fu[:, 1:2, :, :]
        fu3 = fu[:, 2:3, :, :]
        fu4 = fu[:, 3:4, :, :]
        fu1 = self.conv_first_1(fu1)
        fu2 = self.conv_first_1(fu2)  
        fu3 = self.conv_first_1(fu3)
        fu4 = self.conv_first_1(fu4)
        fu = torch.cat((fu1, fu2, fu3, fu4), dim=1)
        pan = self.conv_first_1(pan)
        pan_init = self.conv_init_edge(pan)#(B,1,H,W)


        x = self.conv_after_body(self.forward_features(msf, fu, pan))
        x = self.conv_then(x)
        x = self.conv_last(x)
        return x, msf_init, pan_init
    
"""
if __name__ == '__main__':
    upscale = 1
    height = 64
    width = 64
    model = DAT(
        upscale=2,
        in_chans_4=4,
        in_chans_1=1,
        img_size=64,
        img_range=1.,
        embed_dim=180,
        num_heads=6,
        expansion_factor=2,
        split_size=[8,16],
                ).cuda().eval()

    print(height, width)

    msf = torch.randn((32, 4, height, width)).cuda()
    fu = torch.randn((32, 4, height, width)).cuda()
    pan = torch.randn((32, 1, height, width)).cuda()

    Output, _, _, = model(msf, fu, pan)
    #torch.Size([1, 4, 32, 32])
    print(Output.shape)
"""