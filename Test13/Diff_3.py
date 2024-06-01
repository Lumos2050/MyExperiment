import numpy as np
import cv2
import torch
import torch.nn as nn
def to_tensor(image):
    max_i = np.max(image)
    min_i = np.min(image)
    image = (image - min_i) / (max_i - min_i)
    return image
def image_gradient(img):
    H, W = img.shape
    gx = np.pad(np.diff(img, axis=0), ((0,1),(0,0)), 'constant')
    gy = np.pad(np.diff(img, axis=1), ((0,0),(0,1)), 'constant')
    gradient = abs(gx) + abs(gy)
    return gradient
def edge_dect(img):
    nam=1e-9
    apx=1e-10
    return np.exp( -nam / ( (image_gradient(img)**4)+apx ) )

def edge_dect_batch1(imgs):
    batch_size, channels, H, W = imgs.shape
    edges = np.zeros((batch_size, channels, H, W), dtype=np.float32)
    for i in range(batch_size):
        for j in range(channels):
            img = imgs[i,j]  # 取出每个样本的单通道图像
            edges[i,j] = edge_dect(img)
    return edges
def diff3(ms4, pan):
    """
    torch.Size([128, 4, 16, 16]) torch.Size([128, 1, 64, 64])
    """
    #print(ms4.shape, pan.shape)
    alpha = np.array([0.18, 0.27, 0.11, 0.59]).astype(np.float32)
    upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
    ms4 = upsample(ms4)
    ms4 = ms4.detach().cpu().numpy()
    pan = pan.detach().cpu().numpy()

    batch_size, channels, H, W = ms4.shape

    #ms4 = np.transpose(ms4, (0, 3, 1, 2))
    #pan = np.transpose(pan, (0, 2, 3, 1))
    print(ms4.shape, pan.shape)
    # 使用广播机制对每个通道应用 alpha 值
    alpha = alpha.reshape(1, -1, 1, 1)  # 形状变为 (1, 4, 1, 1)
    MP = alpha * ms4
    I_m = np.mean(MP, axis=1, keepdims=True)
    PT = pan - I_m
    MT = (1 - alpha) * ms4

    #print(MP.shape, PT.shape, MP.shape)
    print("正在图像处理请稍候......")

    # Edge detection operator
    W_mi = edge_dect_batch1(ms4)#(B,4,H,W)
    W_pi = edge_dect_batch1(pan)#(B,1,H,W)
    m_sum = np.zeros_like(W_pi)#(5, 1, 64, 64) 
    m_sum = np.mean(ms4, axis=1, keepdims=True)

    gamma = 0.3  # 0 < gamma < 0.5
    W_m = np.zeros_like(W_mi)
    #print("1",W_m.shape)
    for i in range(batch_size):
        for j in range(channels):
            W_m[i,j] = ms4[i, j] / m_sum[i,0] * W_mi[i, j]

    W_m_av = np.mean(W_m, axis=1, keepdims=True)
    W_p = gamma * W_m_av + (1 - gamma) * W_pi
    #print(" W_m.shape, W_m_av.shape, W_p.shape", W_m.shape, W_m_av.shape, W_p.shape)
    # Fusion

    PT_1 = np.zeros_like(PT)    
    MT_1 = np.zeros_like(MT)
    MP_1 = np.zeros_like(MP)
    for i in range(batch_size):
            PT_1[i] = PT[i] * W_p[i]

    for i in range(batch_size):
        for j in range(channels):
            MT_1[i, j] = MT[i, j] * W_m[i, j]
            MP_1[i, j] = MP[i, j] * W_m[i, j]

    PT_1 = to_tensor(PT_1)
    MT_1 = to_tensor(MT_1)
    MP_1 = to_tensor(MP_1)

    PT_1 = torch.from_numpy(PT_1).cuda().float()
    MT_1 = torch.from_numpy(MT_1).cuda().float()
    MP_1 = torch.from_numpy(MP_1).cuda().float()
    return MT_1, MP_1, PT_1
"""
ms = torch.randn((5, 4, 16, 16), dtype=torch.float32).cuda()
pan = torch.randn((5, 1, 64, 64), dtype=torch.float32).cuda()  # 这里，必须让 init 长宽都是 ms 的四倍！
MT_1, MP_1, PT_1 = diff3(ms, pan)
print(MT_1.shape, MP_1.shape, PT_1.shape)
"""