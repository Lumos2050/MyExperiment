import torch
import os
import torch.optim as optim
from Diff_process import train_loader
from Diff_process import test_loader
from model import Total_model

from utils import setup_seed, train_model, test_model
import logging
#from torch.utils.tensorboard import SummaryWriter
import time
from tqdm import tqdm
#-------------------定义超参数----------------------------------------------
EPOCH = 25  # 训练多少轮次
BATCH_SIZE = 75# 每次喂给的数据量实际应该在数据加载里面改
LR = 0.001 # 学习率
seed_num = 3407
Train_Rate = 0.1  # 将训练集和测试集按比例分开
os.environ['CUDA_VISIBLE_DEVICES'] = '0'   # 是否用GPU环视cpu训练
torch.cuda.set_per_process_memory_fraction(0.8)
opt = {
    'DAT_options': {
        'in_chans_4': 4,
        'in_chans_1': 1,
        'img_size': 64,
        'embed_dim': 48,
        'num_heads': 6,
        'expansion_factor': 2,
        'split_size': [8, 16],
        'proj_drop_rate' : 0.16,
        'attn_drop_rate' : 0.16,
        'drop_paths' : [0.16, 0.12, 0.08, 0.16, 0.12, 0.08]
    },
    'edge_sv_options': {
        'inplane': 64
    }
}


def main():
    #设置随机种子
    setup_seed(seed_num)

    if_cuda = torch.cuda.is_available()
    print("if_cuda=",if_cuda)
    gpu_count = torch.cuda.device_count()
    print("gpu_count=",gpu_count)

    #定义优化器
    model = Total_model(opt)
    model = model.cuda()

    #if os.path.exists('D:\MyCo\MyCode\code\model_epoch_16.pt'):
        #model.load_state_dict(torch.load('D:\MyCo\MyCode\code\model_epoch_16.pt'))

    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    logging.basicConfig(level=logging.INFO, filename='training.log',
                    format='%(asctime)s - %(levelname)s - %(message)s')
    # 调用训练和测试
    for epoch in range(1, EPOCH+1):
        train_model(model, train_loader, optimizer, epoch, lamda1 = 0.0, lamda2 = 0.0, lamda3 = 0.0)
        #torch.save(model.state_dict(), f'./model_epoch_{epoch}.pt')
        test_model(model, test_loader)
    #保存模型
    #torch.save(model, './model.pkl')
    # 保存最终模型
    #torch.save(model.state_dict(), './model_final.pt')

if __name__ == '__main__':
    main()


