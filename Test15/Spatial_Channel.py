import torch
import torch.nn as nn
from torch.nn import functional as F
from timm.models.layers import DropPath
from einops.layers.torch import Rearrange
from einops import rearrange
from edge_sv import SqueezeBodyEdge
from Restnet import ResNet18

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


class Process_Spatial_Attention(nn.Module):
    """
    令MP的每个通道(已经被映射成180维)和PT进行交叉注意力, 最后形成以后再降维720->180
    
    """
    def __init__(self, dim, idx = 0, split_size=[8,16], dim_out = None , num_heads=6, attn_drop=0.):
        super().__init__()

        self.dim = dim
        self.dim_out = dim_out or dim
        self.split_size = split_size
        self.num_heads = num_heads
        self.idx = idx
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
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
        self.change_fu = nn.Conv2d(int((self.dim)*4), self.dim, 3, 1, 1)
        self.dwconv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups = dim),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )
        self.spatial_interaction = nn.Sequential(
            nn.Conv2d(dim, dim // 2, kernel_size=1),
            nn.BatchNorm2d(dim // 2),
            nn.GELU(),
            nn.Conv2d(dim // 2, 1, kernel_size=1)
        )
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(attn_drop)

    def im2win(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(-2,-1).contiguous().view(B, C, H, W)
        x = img2windows(x, self.H_sp, self.W_sp)
        x = x.reshape(-1, self.H_sp* self.W_sp, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        return x
    

    def forward(self, x, y, H, W):
        """
        x是fu(B,L,720), y是pan(B,L,180通道)
        Input: qkv: (B, 3*L, C),torch.Size([1, 4096, 180]),  H, W
        Output: x: (B, H*W, C)
        """
         # convolution output
        SM = rearrange(y, "b (h w) c -> b c h w", h=H, w=W)

        conv_x = self.dwconv(SM)#v1(B, C, H, W)
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
        k2, v2 = qkv2[1], qkv2[2]
        #把fu的四个通道拆开
        dim = self.dim
        qkv11 = qkv1[:, :, :, 0:dim] 
        q11 = qkv11[0]
        qkv12 = qkv1[:, :, :, dim:int(dim*2)]
        q12 = qkv12[0]
        qkv13 = qkv1[:, :, :, int(dim*2):int(dim*3)]
        q13 = qkv13[0]
        qkv14 = qkv1[:, :, :, int(dim*3):int(dim*4)]
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
        attn1 = (q11 @ k2.transpose(-2, -1))* self.scale  # B head N C @ B head C N --> B head N N
        attn2 = (q12 @ k2.transpose(-2, -1))* self.scale  # B head N C @ B head C N --> B head N N
        attn3 = (q13 @ k2.transpose(-2, -1))* self.scale  # B head N C @ B head C N --> B head N N
        attn4 = (q14 @ k2.transpose(-2, -1))* self.scale # B head N C @ B head C N --> B head N N


        #先q@k，再位置编码，@v
        # calculate drpe


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


        # S-Map (before sigmoid)
        spatial_map = self.spatial_interaction(conv_x).permute(0, 2, 3, 1).contiguous().view(B, L, 1)
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
        # S-I
        z = z * torch.sigmoid(spatial_map)
        #print("PSA中的z形状", z.shape)torch.Size([1, 4096, 180])
        z = self.proj(z)
        z = self.proj_drop(z)
        
        return z
    

#idx 管理窗口0还是1
class Spatial_Attention(nn.Module):
    """
    处理完第一个以后，其余普通的Spatial_Attention
    """
    def __init__(self, dim, idx, split_size=[8,8], dim_out=None, num_heads=6, attn_drop=0.):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out or dim
        self.split_size = split_size
        self.num_heads = num_heads
        self.idx = idx

        head_dim = dim // num_heads if dim >= num_heads else 1
        self.scale = head_dim ** -0.5

        if idx == 0:
            H_sp, W_sp = self.split_size[0], self.split_size[1]
        elif idx == 1:
            W_sp, H_sp = self.split_size[0], self.split_size[1]
        else:
            print ("ERROR MODE", idx)
            exit(0)
        self.H_sp = H_sp
        self.W_sp = W_sp
        self.attn_drop = nn.Dropout(attn_drop)

    def im2win(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(-2,-1).contiguous().view(B, C, H, W)
        x = img2windows(x, self.H_sp, self.W_sp)
        """
        Input: Image (B, C, H, W)
        Output: Window Partition (B', N, C)
        """
        x = x.reshape(-1, self.H_sp* self.W_sp, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        return x

    def forward(self, qkv1,qkv2, H, W):
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
        attn = (q1 @ k2.transpose(-2, -1))* self.scale  # B head N C @ B head C N --> B head N N

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
    def __init__(self, dim, num_heads=6, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        head_dim = dim // num_heads if dim >= num_heads else 1
        self.scale = head_dim ** -0.5

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



        attn = (q @ k.transpose(-2, -1)) * self.scale
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
    """ 
    没有swin
    """
    def __init__(self, dim, num_heads, split_size=[8,16], proj_drop=0., attn_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.split_size = split_size
        self.qkv = nn.Linear(dim, dim * 3)
     
        self.branch_num = 2

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.attns = nn.ModuleList([
                Spatial_Attention(
                    dim//2 if dim >= 4 else 1, idx = i,
                    split_size=split_size, num_heads=num_heads//2, dim_out=dim//2 if dim >= 4 else 1,
                    attn_drop=attn_drop)
                for i in range(self.branch_num)])
             
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

        z = attened_x 

        z = self.proj(z)
        z = self.proj_drop(z)

        return z
    

class FFN(nn.Module):
    """ 
    前向传播模块,会被整合到Process_Spatial_FFN和Spatial_Channel_FFN两个里面

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
    


class Process_Spatial_FFN(nn.Module):

    """
    输入输出都加了Lnorm
    """
    def __init__(self, dim, num_heads, split_size=[8,16],expansion_factor=2,
                 proj_drop= 0.1, attn_drop=0.1, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()

        self.norm0 = norm_layer(dim*4)
        self.norm1 = norm_layer(dim)
        #Proscess Spatial cross transformer block
        self.attn1 = Process_Spatial_Attention(
            dim, idx=0, split_size=split_size, num_heads=num_heads, attn_drop=attn_drop)
        
        self.proj_drop = nn.Dropout(proj_drop)
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
        z = self.proj_drop(self.attn1(self.norm0(x),self.norm1(y), H, W))
        z = z + self.proj_drop(self.ffn(self.norm2(z) , H, W))
        z = self.norm2(z)
        return z
    

class  Spatial_Channel_FFN(nn.Module):
    """
    输入输出都加了Lnorm
    """
    def __init__(self, dim, num_heads, split_size=[8,16], expansion_factor=2,
                 attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()

        self.norm = norm_layer(dim)
        #Spatial cross transformer block
        self.attn1 = Adaptive_Spatial_Attention(
            dim, num_heads=num_heads, split_size=split_size, 
            proj_drop=drop_path, attn_drop=attn_drop)

        #Channel cross transformer block
        self.attn2 = Adaptive_Channel_Attention(
            dim, num_heads=num_heads, attn_drop=attn_drop,
            proj_drop=drop_path)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        ffn_hidden_dim = int(dim * expansion_factor)
        self.ffn =FFN(in_features=dim, hidden_features=ffn_hidden_dim, out_features=dim, act_layer=act_layer)
    def forward(self, x, y, x_size):
        """
        Input: x: (B, H*W, C), x_size: (H, W)
        Output: x: (B, H*W, C)
        """        
        H , W = x_size
        z = self.drop_path(self.attn1(self.norm(x),self.norm(y), H, W))
        z = self.drop_path(self.attn2(self.norm(z), H, W))
        z = z + self.drop_path(self.ffn(z, H, W))#只在feedforward时候加残差快
        z = self.norm(z)
        return z
    

class ResidualGroup(nn.Module):
    """ 
    总核心
    """
    def __init__(   self,
                    dim,
                    num_heads,
                    split_size=[8,16],
                    expansion_factor=2,
                    proj_drop=0.1,
                    attn_drop=0.1,
                    drop_paths=[0.2, 0.16, 0.12, 0.08, 0.04, 0.0],
                    act_layer=nn.GELU,
                    norm_layer=nn.LayerNorm,
                    num_feat = 64
                    ):
        super().__init__()

        self.blocks1 = Process_Spatial_FFN(
            dim=dim,
            num_heads=num_heads,
            split_size = split_size,
            expansion_factor=expansion_factor,
            proj_drop=proj_drop,
            attn_drop=attn_drop,
            act_layer=act_layer,
            norm_layer=norm_layer,
            )


        self.blocks2 = nn.ModuleList([
        Spatial_Channel_FFN(
            dim=dim,
            num_heads=num_heads,
            split_size = split_size,
            expansion_factor=expansion_factor,
            attn_drop=attn_drop,
            drop_path=drop_paths[i],
            act_layer=act_layer,
            norm_layer=norm_layer,
            )for i in range(5)])
        
        self.blocks3 = nn.ModuleList([
        Spatial_Channel_FFN(
            dim=dim*2,
            num_heads=num_heads,
            split_size = split_size,
            expansion_factor=expansion_factor,
            attn_drop=attn_drop,
            drop_path=drop_paths[i],
            act_layer=act_layer,
            norm_layer=norm_layer,
            )for i in range(6)])        

        self.blocks4 = nn.ModuleList([
        Spatial_Channel_FFN(
            dim=dim*4,
            num_heads=num_heads,
            split_size = split_size,
            expansion_factor=expansion_factor,
            attn_drop=attn_drop,
            drop_path=drop_paths[i],
            act_layer=act_layer,
            norm_layer=norm_layer,
            )for i in range(3)])   


        self.change_fu = nn.Sequential(nn.Conv2d(dim*4, dim, 3, 1, 1), nn.BatchNorm2d(dim), nn.LeakyReLU(inplace=True))

        self.up_dim20 = nn.Sequential(
        nn.Conv2d(dim*2, dim*2, 3, 2, 1),
        nn.BatchNorm2d(dim*2),
        nn.LeakyReLU(inplace=True)
        )
        self.up_dim2 = nn.Sequential(
        nn.Conv2d(dim, dim*2, 3, 2, 1),
        nn.BatchNorm2d(dim*2),
        nn.LeakyReLU(inplace=True)
        )
        self.up_dim40 = nn.Sequential(
        nn.Conv2d(dim*4, dim*4, 3, 2, 1),
        nn.BatchNorm2d(dim*4),
        nn.LeakyReLU(inplace=True)
        )
        self.up_dim4 = nn.Sequential(
        nn.Conv2d(dim*2, dim*4, 3, 2, 1),
        nn.BatchNorm2d(dim*4),
        nn.LeakyReLU(inplace=True)
        )
        self.down_dim2 = nn.Sequential(
        nn.Conv2d(dim*4, dim*2, 3, 1, 1),
        nn.BatchNorm2d(dim*2),
        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        nn.LeakyReLU(inplace=True),
        nn.Conv2d(dim*2, dim*2, 3, 1, 1),
        nn.BatchNorm2d(dim*2),
        )
        self.down_dim4 = nn.Sequential(
        nn.Conv2d(dim*2, dim, 3, 1, 1),
        nn.BatchNorm2d(dim),
        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        nn.LeakyReLU(inplace=True),
        nn.Conv2d(dim, dim, 3, 1, 1),
        nn.BatchNorm2d(dim)
        )        
        self.edge_sv1 = SqueezeBodyEdge(dim*2)
        self.edge_sv2 = SqueezeBodyEdge(dim)
    def forward(self, msf, fu, pan, MT1, PT1, MT2, PT2, x_size):
        """
        Input: x: (B, H*W, C), x_size: (H, W), fu(B,H*W,720)
        Output: x: (B, H*W, C)在这里x是输出（没有完全改）
        """
        H, W = x_size
        B, L, C = msf.shape
        x1_size = int(H/2), int(W/2)
        x2_size = int(H/4), int(W/4)
        H1, W1 = x1_size
        H2, W2 = x2_size
        L1 = int(H * W / 4)
        L2 = int(H * W / 16)

        z1 = self.blocks1(fu, pan, x_size)
        blk1 = self.blocks2[0]
        blk2 = self.blocks2[1]

        blk3 = self.blocks3[0]
        blk4 = self.blocks3[1]
        blk5 = self.blocks3[2]


        blk6 = self.blocks4[0]
        blk7 = self.blocks4[1]
        blk8 = self.blocks4[2]

         
        fu = rearrange(fu, "b (h w) c -> b c h w", h=H, w=W)
        fu = self.change_fu(fu)
        fu = rearrange(fu, "b c h w -> b (h w) c", h=H, w=W)

        z2 = blk1(z1, msf, x_size)
        z3 = blk2(z2, fu, x_size)
#------------------------得到H*W*C--------------------------------------------------
        z3 = z3.transpose(-2,-1).contiguous().view(B, C, H, W)
        z30 = z3
        z3 = torch.concat([z3, MT1, PT1], dim=1).float()
        z3 = self.up_dim20(z3)#通道数翻倍,尺寸减倍
        z3 = z3.permute(0, 2, 3, 1).contiguous().view(B, L1, C*2)

        z2 = z2.transpose(-2,-1).contiguous().view(B, C, H, W)
        z2 = self.up_dim2(z2)#通道数翻倍,尺寸减倍
        z2 = z2.permute(0, 2, 3, 1).contiguous().view(B, L1, C*2)

        z1 = z1.transpose(-2,-1).contiguous().view(B, C, H, W)
        z1 = self.up_dim2(z1)#通道数翻倍,尺寸减倍
        z1 = z1.permute(0, 2, 3, 1).contiguous().view(B, L1, C*2)

        

        z4 = blk3(z3, z1, x1_size)
        z5 = blk4(z4, z2, x1_size)
        z6 = blk5(z5, z3, x1_size)
#-----------------得到H/2 * W/2 * 2C------------------------------------------------------       
        z4 = z4.transpose(-2,-1).contiguous().view(B, C*2, H1, W1)
        z4 = self.up_dim4(z4)#通道数翻倍,尺寸减倍
        z4 = z4.permute(0, 2, 3, 1).contiguous().view(B, L2, C*4)

        z5 = z5.transpose(-2,-1).contiguous().view(B, C*2, H1, W1)
        z5 = self.up_dim4(z5)#通道数翻倍,尺寸减倍
        z5 = z5.permute(0, 2, 3, 1).contiguous().view(B, L2, C*4)

        z6 = z6.transpose(-2,-1).contiguous().view(B, C*2, H1, W1)
        z60 = z6
        z6 = torch.concat([z6, MT2, PT2], dim=1).float()
        z6 = self.up_dim40(z6)#通道数翻倍,尺寸减倍
        z6 = z6.permute(0, 2, 3, 1).contiguous().view(B, L2, C*4)

        z7 = blk6(z6, z4, x2_size)
        z8 = blk7(z7, z5, x2_size)
        z9 = blk8(z8, z6, x2_size)
#-----------------得到H/4 * W/4 * 4C，开始边缘监督------------------------------------------------------       

        z9 = z9.transpose(-2,-1).contiguous().view(B, C*4, H2, W2)
        z9 = self.down_dim2(z9)
        z9 = self.edge_sv1(z9,z60)


#-----------------得到H/2 * W/2 * 2C------------------------------------------------------  

        z9 = self.down_dim4(z9)
        z9 = self.edge_sv2(z9,z30)

#-----------------得到H * W * C------------------------------------------------------          
        return z9

class DAT(nn.Module):
    """ 
    整体建构
    """
    def __init__(self,
                in_chans_4=4,
                in_chans_1=1,
                embed_dim = 48,
                split_size=[8,16],
                num_heads= 6,
                expansion_factor=2,
                proj_drop_rate = 0.,
                drop_paths = [0.2, 0.16, 0.12, 0.08, 0.04, 0.0],
                attn_drop_rate=0.,
                act_layer=nn.GELU,
                norm_layer=nn.LayerNorm,
                **kwargs):
        super().__init__()

        num_in_ch_4 = in_chans_4
        num_in_ch_1 = in_chans_1
        num_out_ch = in_chans_4
        num_feat = 64
        self.dim = embed_dim
        # ------------------------- 1, Shallow Feature Extraction ------------------------- #
        self.conv_first_4 = nn.Sequential(
            nn.Conv2d(num_in_ch_4, embed_dim, 3, 1, 1),
            nn.BatchNorm2d(embed_dim),
            nn.LeakyReLU(inplace=True)
        )
        self.conv_first_1 = nn.Sequential(
            nn.Conv2d(num_in_ch_1, embed_dim, 3, 1, 1),
            nn.BatchNorm2d(embed_dim),
            nn.LeakyReLU(inplace=True)
        )
        self.conv_init_edge40 = nn.Sequential(nn.Conv2d(4, embed_dim//2, 3, 1, 1),nn.BatchNorm2d(embed_dim//2),nn.LeakyReLU())
        self.conv_init_edge10 = nn.Sequential(nn.Conv2d(1, embed_dim//2, 3, 1, 1),nn.BatchNorm2d(embed_dim//2),nn.LeakyReLU())
        self.conv_init_edge41 = nn.Sequential(nn.Conv2d(4, embed_dim, 3, 2, 1),nn.BatchNorm2d(embed_dim),nn.LeakyReLU())
        self.conv_init_edge11 = nn.Sequential(nn.Conv2d(1, embed_dim, 3, 2, 1),nn.BatchNorm2d(embed_dim),nn.LeakyReLU())
        # ------------------------- 2, Deep Feature Extraction ------------------------- #
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        #层归一化

        self.before_RG = nn.Sequential(
            Rearrange('b c h w -> b (h w) c'),
            nn.LayerNorm(embed_dim)
        )

        self.layer = ResidualGroup(
            dim=embed_dim,
            num_heads=num_heads,
            split_size=split_size,
            expansion_factor=expansion_factor,
            proj_drop=proj_drop_rate,
            attn_drop=attn_drop_rate,
            drop_paths=drop_paths,#####################################################
            act_layer=act_layer,
            norm_layer=norm_layer,
            num_feat = num_feat
            )
        #残差连接:用成自己的,不是残差连接
        # build the last conv layer in deep feature extraction
        #self.conv_after_body6 = nn.Sequential(nn.Conv2d(embed_dim*2, embed_dim*2, 3, 1, 1), nn.BatchNorm2d(embed_dim*2), nn.LeakyReLU(inplace=True))
        #self.conv_after_body9 = nn.Sequential(nn.Conv2d(embed_dim*4, embed_dim*2, 3, 1, 1), nn.BatchNorm2d(embed_dim*2), nn.LeakyReLU(inplace=True))
        #self.conv_then = nn.Sequential(nn.Conv2d(embed_dim*2, num_feat, 3, 1, 1), nn.BatchNorm2d(num_feat), nn.LeakyReLU(inplace=True))
        #self.conv_last = nn.Sequential(nn.Conv2d(num_feat, num_feat, 3, 1, 1), nn.BatchNorm2d(num_feat), nn.Sigmoid())
        self.fc = ResNet18(embed_dim*2)
        # ------------------------- 3, Reconstruction delete------------------------- #
    
    #让所有的处理都在residual里面就行
    def forward_features(self, msf, fu, pan, MT1, PT1, MT2, PT2):
        _, _, H, W = msf.shape
        x_size = [H, W]
        msf = self.before_RG(msf)#b c h w -> b (h w) c
        dim = self.dim
        #把fu的四个通道拆开
        #print("here1", fu.shape)
        fu1 = fu[:, 0:dim, :, :]
        fu2 = fu[:, dim:int(dim*2), :, :]
        fu3 = fu[:, int(dim*2):int(dim*3), :, :]
        fu4 = fu[:, int(dim*3):int(dim*4), :, :]
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
        x = self.layer(msf, fu, pan,  MT1, PT1, MT2, PT2, x_size)


        return x#, reconstruction_loss

    def forward(self, msf, fu, pan):
        """
        Input: msf fu: (B, 4, H, W) pan: (B, 1, H, W)
        output: x(B, 64, H/2, W/2), msf_init, pan_init(B, 16, H, W)
        
        """
        #print("msf,fu,pan的形状", msf.shape, fu.shape, pan.shape)
        #print("DAT:main中的msf和pan,后面是一个释放显存", msf.shape, pan.shape)
        torch.cuda.empty_cache()
        msf_in = self.conv_first_4(msf)
        MT1 = self.conv_init_edge40(msf)#(B,C,H,W)
        MT2 = self.conv_init_edge41(msf)#(B,C,H,W)

        fu1 = fu[:, 0:1, :, :]
        fu2 = fu[:, 1:2, :, :]
        fu3 = fu[:, 2:3, :, :]
        fu4 = fu[:, 3:4, :, :]
        fu1 = self.conv_first_1(fu1)
        fu2 = self.conv_first_1(fu2)  
        fu3 = self.conv_first_1(fu3)
        fu4 = self.conv_first_1(fu4)
        fu = torch.cat((fu1, fu2, fu3, fu4), dim=1)

        pan_in = self.conv_first_1(pan)
        PT1 = self.conv_init_edge10(pan)#(B,C,H,W)
        PT2 = self.conv_init_edge11(pan)#(B,C,H,W)

        x = self.forward_features(msf_in, fu, pan_in, MT1, PT1, MT2, PT2)
        x = torch.concat([x, MT1, PT1], dim=1).float()
        x = self.fc(x)
        #conv_after_body9(x)
        #x = self.conv_then(x)
        #x = self.conv_last(x)
        return x
    

if __name__ == '__main__':
    height = 64
    width = 64
    model = DAT(
        in_chans_4=4,
        in_chans_1=1,
        embed_dim=48,
        num_heads=6,
        expansion_factor=2,
        split_size=[8,16],
                ).cuda().eval()

    print(height, width)

    msf = torch.randn((10, 4, height, width)).cuda()
    fu = torch.randn((10, 4, height, width)).cuda()
    pan = torch.randn((10, 1, height, width)).cuda()
    Output = model(msf, fu, pan)
    #torch.Size([1, 4, 32, 32])
    print(Output.shape)
