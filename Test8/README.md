# 改变边缘监督中的f(x).从tanh改成sigmoid
发现f(x)在代码里用的一直是tanh！这会造成细节图添加的权重变成负数！
改成sigmoid以后发现仍然效果不好
