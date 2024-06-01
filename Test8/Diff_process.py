import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import random
import cv2
import os
from tifffile import imread
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
def kappa(confusion_matrix, k):
    dataMat = np.mat(confusion_matrix)
    P0 = 0.0
    for i in range(k):
        P0 += dataMat[i, i]*1.0
    xsum = np.sum(dataMat, axis=1)
    ysum = np.sum(dataMat, axis=0)
    Pe  = float(ysum*xsum)/np.sum(dataMat)**2
    OA = float(P0/np.sum(dataMat)*1.0)
    cohens_coefficient = float((OA-Pe)/(1-Pe))
    return cohens_coefficient
# 设置随机数种子


# 1.定义网络超参数
setup_seed(3407)
BATCH_SIZE = 75# 每次喂给的数据量
os.environ['CUDA_VISIBLE_DEVICES'] = '0'   # 是否用GPU环视cpu训练

if_cuda = torch.cuda.is_available()
#print("if_cuda=",if_cuda)
gpu_count = torch.cuda.device_count()
#print("gpu_count=",gpu_count)

#ms4_np = imread('/root/autodl-tmp/MyCode/dataset/ms4.tif').astype("float32")
ms4_np = imread('./dataset/ms4.tif').astype("float32")
ms4_np = cv2.resize(ms4_np, dsize=None, fx=4, fy=4, interpolation=cv2.INTER_LINEAR)
#print(ms4_np.shape, ms4_np.dtype, np.min(ms4_np), np.max(ms4_np))

#pan_np = imread('/root/autodl-tmp/MyCode/dataset/pan.tif').astype("float32")
pan_np = imread('./dataset/pan.tif').astype("float32")
#print(pan_np.shape, pan_np.dtype, np.min(pan_np), np.max(pan_np))

#label_np = np.load("/root/autodl-tmp/MyCode/dataset/label.npy")
label_np = np.load("./dataset/label.npy")
#print('label数组形状：', np.shape(label_np))

#train_label_np = np.load("/root/autodl-tmp/MyCode/dataset/train.npy")
train_label_np = np.load("./dataset/train.npy")
#print('train_label数组形状：', np.shape(train_label_np))

#test_label_np = np.load("/root/autodl-tmp/MyCode/dataset/test.npy")
test_label_np = np.load("./dataset/test.npy")
#print('test_label数组形状：', np.shape(test_label_np))

# ms4与pan图补零  (给图片加边框）
Ms4_patch_size = 64 # ms4截块的边长
Interpolation = cv2.BORDER_REFLECT_101
# cv2.BORDER_REPLICATE： 进行复制的补零操作;
# cv2.BORDER_REFLECT:  进行翻转的补零操作:gfedcba|abcdefgh|hgfedcb;
# cv2.BORDER_REFLECT_101： 进行翻转的补零操作:gfedcb|abcdefgh|gfedcb;
# cv2.BORDER_WRAP: 进行上下边缘调换的外包复制操作:bcdefgh|abcdefgh|abcdefg;

top_size, bottom_size, left_size, right_size = (int(Ms4_patch_size / 2 - 1), int(Ms4_patch_size / 2),
                                                int(Ms4_patch_size / 2 - 1), int(Ms4_patch_size / 2))
ms4_np = cv2.copyMakeBorder(ms4_np, top_size, bottom_size, left_size, right_size, Interpolation)
#print('补零后的ms4_np图的形状：', np.shape(ms4_np))

Pan_patch_size = Ms4_patch_size   # pan截块的边长
top_size, bottom_size, left_size, right_size = (int(Pan_patch_size / 2 - 1), int(Pan_patch_size / 2),
                                                int(Pan_patch_size / 2 - 1), int(Pan_patch_size / 2))
pan_np = cv2.copyMakeBorder(pan_np , top_size, bottom_size, left_size, right_size, Interpolation)
#print('补零后的pan_np 图的形状：', np.shape(pan_np ))#长和宽都比原先多了31，即Ms4_patch_size-1


label_np = label_np - 1  # 标签中0类标签是未标注的像素，通过减一后将类别归到0-N，而未标注类标签变为255
label_element, element_count = np.unique(label_np, return_counts=True)  # 返回类别标签与各个类别所占的数量
#print('类标：', label_element)
#print('各类样本数：', element_count)
Categories_Number = len(label_element) - 1  # 数据的类别数
#print('标注的类别数：', Categories_Number)
label_row, label_column = np.shape(label_np)  # 获取标签图的行、列


train_label_np = train_label_np - 1  # 标签中0类标签是未标注的像素，通过减一后将类别归到0-N，而未标注类标签变为255
train_label_element, train_element_count = np.unique(train_label_np, return_counts=True)  # 返回类别标签与各个类别所占的数量
#print('类标：', train_label_element)
#print('各类样本数：', train_element_count)
train_Categories_Number = len(train_label_element) - 1  # 数据的类别数
#print('标注的类别数：', train_Categories_Number)
train_label_row, train_label_column = np.shape(train_label_np)  # 获取标签图的行、列


test_label_np = test_label_np - 1  # 标签中0类标签是未标注的像素，通过减一后将类别归到0-N，而未标注类标签变为255
test_label_element, test_element_count = np.unique(test_label_np, return_counts=True)  # 返回类别标签与各个类别所占的数量
#print('类标：', test_label_element)
#print('各类样本数：', test_element_count)
test_Categories_Number = len(test_label_element) - 1  # 数据的类别数
#print('标注的类别数：', test_Categories_Number)
test_label_row, test_label_column = np.shape(test_label_np)  # 获取标签图的行、列


ground_xy_train = np.array([[]] * Categories_Number).tolist()   # [[],[],[],[],[],[],[]]  7个类别
ground_xy_test = np.array([[]] * Categories_Number).tolist()   # [[],[],[],[],[],[],[]]  7个类别

ground_xy_trainallData = np.arange(train_label_row * train_label_column * 2).reshape(train_label_row * train_label_column, 2)  # [800*830, 2] 二维数组
ground_xy_testallData = np.arange(test_label_row * test_label_column * 2).reshape(test_label_row * test_label_column, 2)  # [800*830, 2] 二维数组

count1 = 0
count2 = 0
print("正在图像处理请稍候......")
for row in range(train_label_row):  
    for column in range(train_label_column):
        #所有样本点的坐标[800*830, 2]
        ground_xy_trainallData[count1] = [row, column]
        count1 = count1 + 1
        if train_label_np[row][column] != 255:
            #在对应类别的列表里添加对应类别的坐标
            ground_xy_train[int(train_label_np[row][column])].append([row, column])    
            
for row in range(test_label_row):  
    for column in range(test_label_column):
        ground_xy_testallData[count2] = [row, column]
        count2 = count2 + 1
        if test_label_np[row][column] != 255:
            ground_xy_test[int(test_label_np[row][column])].append([row, column])  
print("正在图像处理请稍候......")
###train各类别内部打乱
for categories in range(Categories_Number):
    ########
    ground_xy_train[categories] = np.array(ground_xy_train[categories])
    shuffle_array = np.arange(0, len(ground_xy_train[categories]), 1)
    np.random.shuffle(shuffle_array)

    ground_xy_train[categories] = ground_xy_train[categories][shuffle_array]

###test各类别内部打乱
for categories in range(Categories_Number):
    ########
    ground_xy_test[categories] = np.array(ground_xy_test[categories])
    shuffle_array = np.arange(0, len(ground_xy_test[categories]), 1)
    np.random.shuffle(shuffle_array)

    ground_xy_test[categories] = ground_xy_test[categories][shuffle_array]

    
ground_xy_ltrain = []
ground_xy_ltest = []
label_train = []
label_test = []


for categories in range(Categories_Number):
    categories_number_train = len(ground_xy_train[categories])
    for i in range(categories_number_train):
        ground_xy_ltrain.append(ground_xy_train[categories][i])
    label_train = label_train + [categories for x in range(int(categories_number_train))]


for categories in range(Categories_Number):
    categories_number_test = len(ground_xy_test[categories])
    for i in range(int(categories_number_test * 0.05)):####################可在此调节test数据集大小#################################
        ground_xy_ltest.append(ground_xy_test[categories][i])
    label_test = label_test + [categories for x in range(int(categories_number_test * 0.05))]####################可在此调节test数据集大小#################################

label_train = np.array(label_train)
label_test = np.array(label_test)
ground_xy_ltrain = np.array(ground_xy_ltrain)
ground_xy_ltest = np.array(ground_xy_ltest)

#train整体打乱
shuffle_array = np.arange(0, len(label_train), 1)
np.random.shuffle(shuffle_array)
label_train = label_train[shuffle_array]
ground_xy_ltrain = ground_xy_ltrain[shuffle_array]
#test整体打乱
shuffle_array = np.arange(0, len(label_test), 1)
np.random.shuffle(shuffle_array)
label_test = label_test[shuffle_array]
ground_xy_ltest = ground_xy_ltest[shuffle_array]

label_train = torch.from_numpy(label_train).type(torch.LongTensor)
label_test = torch.from_numpy(label_test).type(torch.LongTensor)
ground_xy_ltrain = torch.from_numpy(ground_xy_ltrain).type(torch.LongTensor)
ground_xy_ltest = torch.from_numpy(ground_xy_ltest).type(torch.LongTensor)

#图像处理
def to_tensor(image):
    max_i = np.max(image)
    min_i = np.min(image)
    image = (image - min_i) / (max_i - min_i)
    return image


ms4 = to_tensor(ms4_np)
pan = to_tensor(pan_np)
alpha = np.array([0.18, 0.27, 0.11, 0.59]).astype(np.float32)
ms4 = np.transpose(ms4, (2, 0, 1))
alpha = alpha.reshape(-1, 1, 1)
MP = alpha * ms4
I_m = np.sum(MP, axis=0, keepdims=False)#通过设置 keepdims=True，保持了结果的维度为 (3200, 3320, 1)，而不会直接变成 (3200, 3320)
PT = pan - I_m
MT = (1 - alpha) * ms4
#print(MP.shape, PT.shape, MP.shape)
label_fu,label_pan,label_ms=[],[],[]

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

#print(MP.shape, PT.shape, MP.shape)
print("正在图像处理请稍候......")
# edge detection operator
W_mi = [ edge_dect(ms4[i]) for i in range(4) ]
W_pi = edge_dect(pan)
m_sum = (ms4[1] + ms4[2] + ms4[3] + ms4[0]) /4
 
gamma = 0.3 #0 < gamma < 0.5
W_m  = [ms4[i] / m_sum*( W_mi[i] ) for i in range(4)]
W_m_av = (ms4[1] + ms4[2] + ms4[3] + ms4[0]) / m_sum /4
W_m_ava =( W_m[1] + W_m[2] + W_m[3] + W_m[0] ) / 4
W_p = gamma * W_m_ava + (1 - gamma) * W_m_av * W_pi



# fusion
PT_1 = PT * W_p
MT_1 = np.zeros_like(MT)
MP_1 = np.zeros_like(MP)
for i in range(4):
    MT_1[i] = MT[i] * W_m[i]
    MP_1[i] = MP[i] * W_m[i]

PT_1 = to_tensor(PT_1)
MT_1 = to_tensor(MT_1)
MP_1 = to_tensor(MP_1)

class MyData(Dataset):
    def __init__(self, MT, MP, PT, Label, xy, cut_size):
        self.train_data1 = MT
        self.train_data2 = MP
        self.train_data3 = PT
        self.train_labels = Label
        self.gt_xy = xy
        self.cut_ms_size = cut_size
        self.cut_pan_size = cut_size
    def __getitem__(self, index):
        x_ms, y_ms = self.gt_xy[index]
        x_pan = int( x_ms)      
        y_pan = int( y_ms)
        image_MT = self.train_data1[:, x_ms:x_ms + self.cut_ms_size,
                   y_ms:y_ms + self.cut_ms_size]
        image_MP = self.train_data2[:, x_ms:x_ms + self.cut_ms_size,
                   y_ms:y_ms + self.cut_ms_size]        
        image_PT = self.train_data3[x_pan:x_pan + self.cut_pan_size,
                    y_pan:y_pan + self.cut_pan_size]
        

        image_PT = np.expand_dims(image_PT, axis=0)
        locate_xy = self.gt_xy[index]
        target = self.train_labels[index]
        # 打印数据形状
        #print("Shape of MP image:", image_MT.shape)
        #print("Shape of MP image:", image_MP.shape)
        #print("Shape of PT image:", image_PT.shape)
        #print("Target label:", target)
        #print("Location:", locate_xy)
        return image_MT, image_MP, image_PT, target, locate_xy

    def __len__(self):
        return len(self.gt_xy)
#只返回图像、位置   
class MyData1(Dataset):
    def __init__(self, MT, MP, PT, xy, cut_size):
        self.train_data1 = MT
        self.train_data2 = MP
        self.train_data3 = PT
        self.gt_xy = xy
        self.cut_ms_size = cut_size
        self.cut_pan_size = cut_size

    def __getitem__(self, index):
        x_ms, y_ms = self.gt_xy[index]
        x_pan = int(x_ms)  
        y_pan = int(y_ms)

        image_MT = self.train_data1[:, x_ms:x_ms + self.cut_ms_size,
                   y_ms:y_ms + self.cut_ms_size]
        image_MP = self.train_data2[:, x_ms:x_ms + self.cut_ms_size,
                   y_ms:y_ms + self.cut_ms_size]        
        image_PT = self.train_data3[x_pan:x_pan + self.cut_pan_size,
                    y_pan:y_pan + self.cut_pan_size]        
        image_PT = np.expand_dims(image_PT, axis=0)

        locate_xy = self.gt_xy[index]

        return image_MT, image_MP, image_PT, locate_xy

    def __len__(self):
        return len(self.gt_xy)
train_data = MyData(MT_1, MP_1, PT_1, label_train,ground_xy_ltrain, Ms4_patch_size)
test_data = MyData(MT_1, MP_1, PT_1, label_test ,ground_xy_ltest, Ms4_patch_size)########
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

