# 6.1 大突破！

更改网络结构，改变融合和边缘监督的前后关系，让边缘监督嵌入融合（也相当于是一种变相的残差连接），监督的对象是同一尺度的浅层特征和深层特征

删掉边缘监督中计算边缘图、用numpy方法归一化的部分（网络中乘法太多会导致信息损失过多）

在最后送入全连接之前concat初始特有特征MT、PT（该动作十分有用）

最好OA从36%提升到64%
