# Change edge
在数据加强部分，我原本是将原始边缘细节与网络解耦得到的边缘细节相加，现在改成了concat。因为他们是不同来源的边缘细节，直接相加可能缺乏意义，不如让网络自行学习。