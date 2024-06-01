# 改变边缘监督中的f(x).从tanh改成sigmoid
# 加深cross-transformer的深度，（H，W，C）->(H/2, W/2, 2C)->(H/4, W/4, 4C)
发现f(x)在代码里用的一直是tanh！这会造成细节图添加的权重变成负数！

效果不好
