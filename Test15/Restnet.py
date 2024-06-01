import torch.nn as nn
import torch.nn.functional as F
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
        self.blk3_1 = ResBlk(128, 256, stride=1)
        self.blk4_1 = ResBlk(256, 512, stride=2)

        self.outlayer1 = nn.Linear(512, 11)
        #self.outlayer2 = nn.Linear(256, 128)
        #self.outlayer3 = nn.Linear(128, 11)
        #self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        :param x:#(B, 4, H, W)
        :return:
        """
        #print("input x", x.shape)
        x = F.relu(self.conv1(x))
        x = self.blk1_1(x)
        #print('x', x.shape)
        x = self.blk2_1(x)
        #print('x', x.shape)       
        x = self.blk3_1(x)
        #print('x', x.shape)
        x = self.blk4_1(x)
        #print('x', x.shape)
        x = F.adaptive_avg_pool2d(x, [1, 1])


        #print('x', x.shape)

        #print(x.size())
        x = x.view(x.size()[0],  -1)
        #print('ssss', x.shape)
        x = F.relu(self.outlayer1(x))
        #x = F.relu(self.outlayer2(x))
        #x = F.relu(self.outlayer3(x))
        x = F.softmax(x, dim=1)
        #print('ssssssssss',x.shape)
        return x
    

"""
if __name__ == '__main__':
    model = ResNet18(
        inplane = 64
    ).cuda().eval()

    x = torch.randn((5, 64, 32, 32),dtype=torch.float32).cuda()
    cls_prd= model(x)
    print(cls_prd.shape)
    #seg_edge,weighted_iamge的数据类型 torch.Size([5, 96, 32, 32]) torch.Size([5, 48, 32, 32])
#[0.33839926 0.28857422 0.32594809 0.33829752 0.35321045] [0.3386434  0.2882894  0.32545981 0.33880615 0.35239664] torch.Size([5, 11])
"""