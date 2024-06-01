import torch
import os
import torch.optim as optim
from Diff_process import train_loader
from Diff_process import test_loader
from model import Total_model

from utils import setup_seed, train_model, test_model

#from torch.utils.tensorboard import SummaryWriter
import time
from tqdm import tqdm
import logging
#-------------------定义超参数----------------------------------------------
EPOCH = 25  # 训练多少轮次
BATCH_SIZE = 25# 每次喂给的数据量实际应该在数据加载里面改
LR = 0.0001 # 学习率
seed_num = 3407
Train_Rate = 0.1  # 将训练集和测试集按比例分开
os.environ['CUDA_VISIBLE_DEVICES'] = '0'   # 是否用GPU环视cpu训练
torch.cuda.set_per_process_memory_fraction(0.8)
opt = {
    'DAT_options': {
        'upscale': 2,
        'in_chans_4': 4,
        'in_chans_1': 1,
        'img_size': 64,
        'img_range': 1.,
        'embed_dim': 48,
        'num_heads': 6,
        'expansion_factor': 2,
        'split_size': [8, 16],
        'drop_rate' : 0.1,#################
        'attn_drop_rate' : 0.1####################
    },
    'edge_sv_options': {
        'inplane': 64
    }
}


def main():
    # 设置随机种子
    setup_seed(seed_num)

    if_cuda = torch.cuda.is_available()
    print("if_cuda=", if_cuda)
    gpu_count = torch.cuda.device_count()
    print("gpu_count=", gpu_count)
    #logging.basicConfig(filename='testing.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    # 假设 model_list 是包含所有模型的列表
    """
    for model_idx in range(1,31):
        # 定义优化器
        model = Total_model(opt)
        model = model.cuda()

        # 加载模型
        model_path = f'D:\MyCo\MyCode\code\model_epoch_{model_idx}.pt'
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path))

            # 测试模型
            test_model(model, test_loader)
    """
    model = Total_model(opt)
    model = model.cuda()
    model_path = './model_epoch_8.pt'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))

        # 测试模型
        test_model(model, train_loader)
        #test_model(model, test_loader)

if __name__ == '__main__':
    main()


