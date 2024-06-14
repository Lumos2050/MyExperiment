# MyExperiment
This repository is one record for the experiment of my research：**An Edge Supervision Aided Dual Attention Transformer for MS and PAN Classification**. 
There were problems with the initial structure design and code architecture, and I have been modifying the network structure and adjusting hyperparameters for a month now.

Each version containing the network structure changes is placed in a "Test x" folder.
# Explanation
My current dataset is MS and PAN images of Hohhot. In the beginning my network's test set OA was only 20%-30%. I have continuously improved my code from the aspects of data construction, network construction, and details such as normalization. 

The changes and problems found in each version are marked in its folder. Found hard defects (such as forgetting to add normalization) are retained in all subsequent versions after modification. Other changes will be retained or deleted according to the effect.

**6.1 Test14 version raised the OA from 30% to 64%, and 6.3 Test15 version raised it from 64% to 67%.** My current task is to raise the project OA to more than 70% (because the current SOTA calculation methods are around 70%). Then I will use multiple data sets to calculate the accuracy of various categories, AA and Kappa.

The statement of my work is displayed in **"科研实习说明.docx"**. 

# Ps
My thesis supervisor is Zhu Hao, associate professor of Xidian University. He has some achievements in deep learning and remote sensing image processing. He supervises undergraduate research and publishes papers in IEEE TGRS journals(IF:8.2, 中科院一区). This is his personal homepage at https://faculty.xidian.edu.cn/ZHUHAO/zh_CN/index.htm
