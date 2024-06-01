import numpy as np
import torch
import cv2
from torch.utils.data import DataLoader
from torch.nn import functional as F
from tifffile import imread
from torch.utils.data import Dataset


TRAIN_BATCH_SIZE = 35   # 每次喂给的数据量
TEST_BATCH_SIZE = 35

ms4_np = imread('./dataset/ms4.tif').astype("float32")
pan_np =  imread('./dataset/pan.tif').astype("float32")
train_label_np = np.load("./dataset/train.npy")
test_label_np = np.load("./dataset/test.npy")
# ms4与pan图补零  (给图片加边框）
Ms4_patch_size = 16  # ms4截块的边长
Interpolation = cv2.BORDER_REFLECT_101
top_size, bottom_size, left_size, right_size = (int(Ms4_patch_size / 2 - 1), int(Ms4_patch_size / 2),
                                                int(Ms4_patch_size / 2 - 1), int(Ms4_patch_size / 2))
ms4_np = cv2.copyMakeBorder(ms4_np, top_size, bottom_size, left_size, right_size, Interpolation)
Pan_patch_size = Ms4_patch_size * 4  # pan截块的边长
top_size, bottom_size, left_size, right_size = (int(Pan_patch_size / 2 - 4), int(Pan_patch_size / 2),
                                                int(Pan_patch_size / 2 - 4), int(Pan_patch_size / 2))
pan_np = cv2.copyMakeBorder(pan_np, top_size, bottom_size, left_size, right_size, Interpolation)

#assert False
# 按类别比例拆分数据集
# label_np=label_np.astype(np.uint8)
train_label_np = train_label_np - 1# # 标签中0类标签是未标注的像素，通过减一后将类别归到0-N，而未标注类标签变为255
test_label_np = test_label_np - 1
label_element, element_count = np.unique(train_label_np, return_counts=True)  # 返回类别标签与各个类别所占的数量

Categories_Number = len(label_element) - 1  # 数据的类别数

label_row, label_column = np.shape(train_label_np)  # 获取标签图的行、列


'''归一化图片'''
def to_tensor(image):
    max_i = np.max(image)
    min_i = np.min(image)
    image = (image - min_i) / (max_i - min_i)
    return image

ground_xy_allData = np.arange(label_row * label_column * 2).reshape(label_row * label_column, 2)  # [800*830, 2] 二维数组

ground_xy_train = []
ground_xy_test = []
label_train = []
label_test = []

count = 0
for row in range(label_row):  # 行
    for column in range(label_column):
        ground_xy_allData[count] = [row, column]
        count = count + 1
        if int(train_label_np[row][column]) != 255:
            ground_xy_train.append([row, column])     # 记录属于每个类别的位置集合
            label_train.append(train_label_np[row][column])
        if int(test_label_np[row][column]) != 255:
            ground_xy_test.append([row, column])
            label_test.append(test_label_np[row][column])

label_train = np.array(label_train)
label_test = np.array(label_test)
ground_xy_train = np.array(ground_xy_train)
ground_xy_test = np.array(ground_xy_test)

# 训练数据与测试数据，数据集内打乱
shuffle_array = np.arange(0, len(label_test), 1)
np.random.shuffle(shuffle_array)
label_test = label_test[shuffle_array]
ground_xy_test = ground_xy_test[shuffle_array]

shuffle_array = np.arange(0, len(label_train), 1)
np.random.shuffle(shuffle_array)
label_train = label_train[shuffle_array]
ground_xy_train = ground_xy_train[shuffle_array]


label_train = torch.from_numpy(label_train).type(torch.LongTensor)
label_test = torch.from_numpy(label_test).type(torch.LongTensor)
ground_xy_train = torch.from_numpy(ground_xy_train).type(torch.LongTensor)
ground_xy_test = torch.from_numpy(ground_xy_test).type(torch.LongTensor)
ground_xy_allData = torch.from_numpy(ground_xy_allData).type(torch.LongTensor)


# 数据归一化
ms4 = to_tensor(ms4_np)
pan = to_tensor(pan_np)
pan = np.expand_dims(pan, axis=0)  # 二维数据进网络前要加一维
ms4 = np.array(ms4).transpose((2, 0, 1))  # 调整通道

# 转换类型
ms4 = torch.from_numpy(ms4).type(torch.FloatTensor)
pan = torch.from_numpy(pan).type(torch.FloatTensor)

class MyData(Dataset):
    def __init__(self, MS4, Pan, Label, xy, cut_size):
        self.train_data1 = MS4
        self.train_data2 = Pan
        self.train_labels = Label
        self.gt_xy = xy
        self.cut_ms_size = cut_size
        self.cut_pan_size = cut_size * 4
        # self.trans = transforms.Compose([transforms.Resize((64, 64)),])

    def __getitem__(self, index):
        x_ms, y_ms = self.gt_xy[index]
        x_pan = int(4 * x_ms)      # 计算不可以在切片过程中进行
        y_pan = int(4 * y_ms)
        image_ms = self.train_data1[:, x_ms:x_ms + self.cut_ms_size,
                   y_ms:y_ms + self.cut_ms_size]

        # image_ms = self.trans(image_ms)

        image_pan = self.train_data2[:, x_pan:x_pan + self.cut_pan_size,
                    y_pan:y_pan + self.cut_pan_size]
        # image_pan = self.trans(image_pan)
        locate_xy = self.gt_xy[index]

        target = self.train_labels[index]
        return image_ms, image_pan, target, locate_xy

    def __len__(self):
        return len(self.gt_xy)

class MyData1(Dataset):
    def __init__(self, MS4, Pan, xy, cut_size):
        self.train_data1 = MS4
        self.train_data2 = Pan

        self.gt_xy = xy
        self.cut_ms_size = cut_size
        self.cut_pan_size = cut_size * 4
        # self.trans = transforms.Compose([transforms.Resize((64, 64)), ])

    def __getitem__(self, index):
        x_ms, y_ms = self.gt_xy[index]
        x_pan = int(4 * x_ms)  # 计算不可以在切片过程中进行
        y_pan = int(4 * y_ms)
        image_ms = self.train_data1[:, x_ms:x_ms + self.cut_ms_size,
                   y_ms:y_ms + self.cut_ms_size]
        # image_ms = self.trans(image_ms)

        image_pan = self.train_data2[:, x_pan:x_pan + self.cut_pan_size,
                    y_pan:y_pan + self.cut_pan_size]

        #image_pan = self.trans(image_pan)  #适应轮廓波

        locate_xy = self.gt_xy[index]

        return image_ms, image_pan, locate_xy

    def __len__(self):
        return len(self.gt_xy)
train_data = MyData(ms4, pan, label_train, ground_xy_train, Ms4_patch_size)
test_data = MyData(ms4, pan, label_test, ground_xy_test, Ms4_patch_size)
#all_data = MyData1(ms4, pan, ground_xy_allData, Ms4_patch_size)
train_loader = DataLoader(dataset=train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=False, num_workers=16, drop_last=True)
test_loader = DataLoader(dataset=test_data, batch_size=TEST_BATCH_SIZE, shuffle=False, num_workers=16, drop_last=True)
#all_data_loader = DataLoader(dataset=all_data, batch_size=TEST_BATCH_SIZE,shuffle=False,num_workers=16)
#num_workers=0才能运行下面的
#train_loader中的形状 torch.Size([128, 4, 16, 16]) torch.Size([128, 1, 64, 64])
"""
for batch in train_loader:
    image_ms, image_pan, target, locate_xy = batch
    print("train_loader中的形状", image_ms.shape, image_pan.shape)
    break
"""