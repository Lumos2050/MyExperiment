import cv2
import numpy as np
from scipy.io import savemat
from scipy.optimize import minimize, LinearConstraint, Bounds
from tifffile import imread
def to_tensor(image):
    max_i = np.max(image)
    min_i = np.min(image)
    image = (image - min_i) / (max_i - min_i)
    return image

'''def split(pan, size):
    st = []
    for i in range(size):
        for j in range(size):
            st.append(pan[i::size,j::size])
    return np.stack(st, axis=-1)'''

#当前文件夹指资源管理器打开的文件夹，不是当前程序所在文件夹，或者可以直接查看终端PS D:\MyCo\test\IHS\TSMF-Net-main>
#当前文件夹. 上一级..
ms4 = imread('./dataset/ms4.tif').astype("float32")
msf = cv2.resize(ms4, dsize=None, fx=4, fy=4, interpolation=cv2.INTER_LINEAR)
print(msf.shape, msf.dtype, np.min(msf), np.max(msf))
msf = to_tensor( msf )  
#msf = msf.reshape((-1, 4))
print(msf.shape, msf.dtype, np.min(msf), np.max(msf))
#savemat('./dataset/msf.mat', {'msf': msf})


pan = imread('./dataset/pan.tif').astype("float32")
#pan = split(pan, 2)
print(pan.shape, pan.dtype, np.min(pan), np.max(pan))
pan = to_tensor( pan )   
#pan = pan.reshape((-1, 1))
print(pan.shape, pan.dtype, np.min(pan), np.max(pan))
#savemat('./dataset/pan.mat', {'pan': pan})


msf = np.transpose(msf, (2, 0, 1))
def objective(alpha, msf, pan):
    
    alpha = alpha.reshape(-1, 1, 1)
    ms_jiaquan = alpha * msf
    I_m = np.sum(ms_jiaquan, axis=0, keepdims=False)#通过设置 keepdims=True，保持了结果的维度为 (3200, 3320, 1)，而不会直接变成 (3200, 3320)。
    PT = pan - I_m
   # MT = (1 - alpha) * msf
    target = np.linalg.norm(PT)**2
    return target

# 定义初始解
#alpha_init = np.array([0.37, 0.10, 0.10, 0.43]),俩种结果后两位都相同
alpha_init = np.array([0.25, 0.25, 0.25, 0.25])

#constraint = LinearConstraint([[1, 1, 1, 1]], [1], [1])#系数，下界，上界
bounds = Bounds([0, 0, 0, 0], [1, 1, 1, 1])
# 进行优化
result = minimize(objective, alpha_init, args=(msf, pan), bounds=bounds)

# 输出结果
print("Optimal alpha:", result.x)
print("Optimal value:", result.fun)
#Optimal alpha: [0.67412343 0.21338002 0.         0.41370288]
#Optimal value: 13706.533118743053
#Optimal alpha: [0.18425753 0.26668243 0.10885119 0.58744596]
#Optimal value: 148134.86357599456


