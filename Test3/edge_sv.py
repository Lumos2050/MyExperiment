import torch
import cv2
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
def normalize_images(images):
    """
    对多通道图像数据进行归一化操作。

    参数：
    - images: 形状为 (B, 4, H, W) 的图像数据，其中 B 是批量大小，H 和 W 分别是图像的高度和宽度。

    返回：
    - normalized_images: 归一化后的图像数据，形状与输入相同。
    """
    images_cpu = images#.detach().cpu().numpy() 
    normalized_images = np.zeros_like(images_cpu, dtype=np.float32)

    for i in range(images_cpu.shape[0]):  # 对每个批次的图像进行处理
        for j in range(images_cpu.shape[1]):
            channel_data = images_cpu[i, j]  # 获取当前通道的数据
            mean = np.mean(channel_data)  # 计算当前通道的均值
            std = np.std(channel_data)  # 计算当前通道的标准差
            normalized_channel_data = (channel_data - mean) / (std + 1e-8)  # 应用归一化
            min_value = np.min(normalized_channel_data)  # 计算归一化后数据的最小值
            max_value = np.max(normalized_channel_data)  # 计算归一化后数据的最大值
            scaled_data = (normalized_channel_data - min_value) / (max_value - min_value)  # 缩放数据到 [0, 1] 范围内
            normalized_images[i, j] = scaled_data # 更新归一化后的图像数据
    return normalized_images



def canny_weight(msf_init, pan_init, seg_edge, low = 50, high = 150):
    """
    input: msf_init(B, 1, H, W), pan_init(B, 1, H, W), seg(B, 4, H, W)
    output: weighted_image(B, 1, H, W), mse_msf, mse_pan
    """
    msf_init = msf_init.detach().cpu().numpy()
    pan_init = pan_init.detach().cpu().numpy()
    seg_edge = seg_edge.detach().cpu().numpy()


    #--------------------归一化-----------------
    normalize_msf_init = normalize_images(msf_init)
    normalize_pan_init = normalize_images(pan_init)    
    normalize_seg_edge = normalize_images(seg_edge)


    #---------------canny-----------------------
    msf_init_edge = np.zeros_like(msf_init)
    pan_init_edge = np.zeros_like(pan_init)
    seg_edge_edge = np.zeros_like(seg_edge)
    for i in range(normalize_msf_init.shape[0]):  # 对每张图像应用Canny算法
        single_image0 = (normalize_msf_init[i, 0]*255).astype(np.uint8)  # 将图像数据转换为8位无符号整数类型
        msf_init_edge[i, 0] = cv2.Canny(single_image0, low, high)
    for i in range(normalize_pan_init.shape[0]):  # 对每张图像应用Canny算法
        single_image1 = (normalize_pan_init[i, 0]*255).astype(np.uint8)  # 将图像数据转换为8位无符号整数类型
        pan_init_edge[i, 0] = cv2.Canny(single_image1, low, high)

    for i in range(normalize_seg_edge.shape[0]):  # 对每个批次的图像进行处理
        for j in range(normalize_seg_edge.shape[1]):  # 对每张图像的每个通道进行处理
            single_image2 = (normalize_seg_edge[i, j]*255).astype(np.uint8)  # 将图像数据转换为8位无符号整数类型
            seg_edge_edge[i, j] = cv2.Canny(single_image2, low, high)

    #print("msf_init_edge的形状", msf_init_edge.shape)
    #print("seg_edge_edge的形状", seg_edge_edge.shape)
    #print("pan_init_edge的形状", pan_init_edge.shape)
    msf_init_edge = msf_init_edge /255.0
    pan_init_edge = pan_init_edge / 255.0
    seg_edge_edge = seg_edge_edge/255.0

    batch_size = seg_edge_edge.shape[0]
    mse_msf = np.zeros(batch_size)
    mse_pan = np.zeros(batch_size)
    for j in range (batch_size):
        for i in range(seg_edge_edge.shape[1]):  # 对第二张图像的每个通道进行遍历
            mse_msf[j] =np.mean((msf_init_edge[j] - seg_edge_edge[j, i:i+1])**2)  # 计算当前通道的均方误差
            mse_pan[j] =np.mean((pan_init_edge[j] - seg_edge_edge[j, i:i+1])**2)  # 计算当前通道的均方误差


    # 使用 tanh 函数计算权重
    weight_msf = np.tanh(mse_msf)
    weight_pan = np.tanh(mse_pan)
    
    # 对权重进行归一化

    # 计算权重和
    weight_sum = weight_msf + weight_pan

    # 归一化权重
    for i in range(batch_size):
        if weight_sum[i].item() != 0:
            weight_msf[i] /= weight_sum[i].item()
            weight_pan[i] /= weight_sum[i].item()
        else:
            # 如果权重和为零，将权重设置为均匀分布
            print("除以零，错误")
    
    weight_msf_np = np.expand_dims(np.expand_dims(np.expand_dims(weight_msf, axis=-1), axis=-1), axis=-1)
    weight_pan_np = np.expand_dims(np.expand_dims(np.expand_dims(weight_pan, axis=-1), axis=-1), axis=-1)
    # 按权重相加
    weighted_image = msf_init * weight_msf_np + pan_init * weight_pan_np
    #print("mse_msf的形状", mse_msf.shape) 
    #print("mse_pan的形状", mse_pan.shape)
    #print("weighted_image的形状",  weighted_image.shape)
    return weighted_image, mse_msf, mse_pan

#mse_msf, mse_pan要加到loss里



def entropy(image):
    # 计算信息熵
    hist = torch.histc(image, bins=256, min=0, max=1)
    hist /= torch.sum(hist)
    entropy = -torch.sum(hist * torch.log2(hist + 1e-12))  # 添加一个小值避免 log(0) 的情况
    return entropy

def weighted_sum(msf_init, pan_init):
    """
    input: msf_init(B, 1, H, W), pan_init(B, 1, H, W)
    output: weighted_image(B, 1, H, W)
    """
    # 计算每张图像的熵值
    entropy_msf = entropy(msf_init)
    entropy_pan = entropy(pan_init)
    
    # 使用 tanh 函数计算权重
    weight_msf = torch.tanh(entropy_msf)
    weight_pan = torch.tanh(entropy_pan)
    
    # 对权重进行归一化
    weight_sum = weight_msf + weight_pan
    weight_msf /= weight_sum if weight_sum.item() != 0 else 1
    weight_pan /= weight_sum if weight_sum.item() != 0 else 1
    
    # 按权重相加
    weighted_image = msf_init * weight_msf + pan_init * weight_pan
    
    return weighted_image

# 示例用法
# 假设 msf_init 和 pan_init 分别是两个形状为 (B, 1, H, W) 的张量
# weighted_image = weighted_sum(msf_init, pan_init)

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

    def __init__(self):
        super(ResNet18, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64)
        )

        #followed 4 blocks
        #[b, 64, h, w] => [b, 64, h , w]
        self.blk1_1 = ResBlk(64, 128, stride=1)

        #[b, 64, h, w] =>[b , 128, h ,w]
        #self.blk2_1 = ResBlk(64, 128, stride=1)

        #[b, 128, h, w] =>[b , 256, h ,w]
        self.blk3_1 = ResBlk(128, 256, stride=1)

        # [b, 256, h, w] =>[b , 512, h ,w]
        self.blk4_1 = ResBlk(256, 512, stride=1)

        self.outlayer = nn.Linear(512, 11)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        :param x:#(B, 4, H, W)
        :return:
        """
        #print("input x", x.shape)
        x = F.relu(self.conv1(x))
        #[b, 64, h, w] => [b, 512, h , w]
        x = self.blk1_1(x)
        #x = self.blk2_1(x)
        x = self.blk3_1(x)
        x = self.blk4_1(x)
        x = F.adaptive_avg_pool2d(x, [1, 1])


        #print('x', x.shape)

        #print(x.size())
        x = x.view(x.size()[0],  -1)
        #print('ssss', x.shape)
        x = self.sigmoid(self.outlayer(x))
        #print('ssssssssss',x.shape)
        return x
    
class SqueezeBodyEdge(nn.Module):
    """
    input: x:torch.Size([1, 4, 32, 32]) inplane = 4
    output: seg_flow_warp(就是body), seg_edge, 都是torch.Size([1, 4, 32, 32]) 
    """


    def __init__(self, inplane):
        """
        implementation of body generation part
        :param inplane:
        :param norm_layer:
        """
        super(SqueezeBodyEdge, self).__init__()
        #四倍降采样，还用的是深度可分离卷积
        self.down = nn.Sequential(
            nn.Conv2d(inplane, inplane, kernel_size=3, groups=inplane, stride=2),
            # norm_layer(inplane),
            nn.BatchNorm2d(inplane),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplane, inplane, kernel_size=3, groups=inplane, stride=2),
            # norm_layer(inplane),
            nn.BatchNorm2d(inplane),
            nn.ReLU(inplace=True)
        )
        self.flow_make = nn.Conv2d(inplane * 2, 2, kernel_size=3, padding=1, bias=False)
        self.upsample2x_op = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv_5to4 = nn.Conv2d(inplane + 1, inplane, kernel_size=3, padding=1, bias=False)
        self.fc = ResNet18()
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

        output = F.grid_sample(input, grid)
        return output
    
    def forward(self, x, msf_init, pan_init):
        """
        input: x:torch.Size([1, 4, H/2, W/2])
        x(B, 4, H/2, W/2), msf_init, pan_init(B, 1, H, W)
        """
        #print("边缘监督中的释放显存1")
        torch.cuda.empty_cache()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = x.to(device)
        x = x.detach().cpu().numpy()
        x = normalize_images(x)
        x = torch.from_numpy(x).cuda()


        msf_init = msf_init.to(device)
        pan_init = pan_init.to(device)

#init的归一化在canny函数里面

        size = x.size()[2:]#H和W
        seg_down = self.down(x)
        seg_down = F.upsample(seg_down, size=size, mode="bilinear", align_corners=True)
        #concat成8通道，然后变成两通道
        flow = self.flow_make(torch.cat([x, seg_down], dim=1))#(B, 2, H/2, W/2)

        seg_flow_warp = self.flow_warp(x, flow, size)#(B, 4, H/2, W/2)
        seg_edge = x - seg_flow_warp


        #print(seg_edge.shape, seg_flow_warp.shape)
        #torch.Size([1, 4, 64, 64]) torch.Size([1, 4, 64, 64])

        #-----------------Enhancement--------------------------------------
        seg_edge =  self.upsample2x_op(seg_edge)#(B, 4, H/2, W/2)->(B, 4, H, W)

        #print("1seg_edge", seg_edge.shape)

        seg_flow_warp = self.upsample2x_op(seg_flow_warp)#(B, 4, H/2, W/2)->(B, 4, H, W)
        weighted_image, mse_msf, mse_pan = canny_weight(msf_init, pan_init, seg_edge, low = 0.4, high = 0.8)#(B, 1, H, W)

        #print("2seg_edge", seg_edge.shape, weighted_image.shape)
        weighted_image = torch.from_numpy(weighted_image).cuda()
        weighted_image = weighted_image.float()  # 将输入数据转换为 torch.cuda.FloatTensor 类型
        #print("seg_edge,weighted_iamge的数据类型", seg_edge.dtype, weighted_image.dtype)
        fine_edge = self.conv_5to4(torch.cat([seg_edge, weighted_image], dim = 1))#(B, 4, H, W)
        final = seg_flow_warp + fine_edge#(B, 4, H, W)
        #print("边缘监督中的释放显存2")
        torch.cuda.empty_cache()
        cls_prd = self.fc(final)
        return seg_flow_warp, seg_edge, mse_msf, mse_pan, cls_prd

"""
if __name__ == '__main__':
    height = 64
    width = 64
    model = SqueezeBodyEdge(
        inplane = 4
    ).cuda().eval()

    print(height, width)

    x = torch.randn((5, 4, int(height/2), int(width/2)),dtype=torch.float32).cuda()
    msf_init = torch.randn((5, 1, height, width),dtype=torch.float32).cuda()
    pan_init = torch.randn((5, 1, height, width),dtype=torch.float32).cuda()

    _, _, mse_msf, mse_pan, cls_prd= model(x, msf_init, pan_init)

    print(mse_msf, mse_pan, cls_prd.shape)
    #698.5107421875 1016.015625 torch.Size([1, 7])
    #[174.62768555 174.62768555 142.87719727 174.62768555 142.87719727] [174.62768555 174.62768555 142.87719727 174.62768555 142.87719727] torch.Size([5, 11])
"""