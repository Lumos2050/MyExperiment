## 5.29 小突破：更改数据构建
由于经过众多尝试，发现效果依然很差，决定更改数据构建。之前的数据构建我把ms的上采样放在导入部分，然后根据位置信息对准每一组ms和pan。而经典的demo在把ms和pan导入后没有尺度变化操作，我可能之前在对准部分出现问题。虽然这个问题在我先前检查的时候并没有发现。

应该先用经典demo构建好，在再每个批次的训练中变换尺度。这样数据构建一定不会出错。

更改后，OA由10%~20%变为30%，虽然也不怎么样，但说明数据曾经一定有问题。现在数据不会有问题了。
