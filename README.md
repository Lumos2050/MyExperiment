# MyExperiment
This repository is one record for the experiment of my research：**An Edge Supervision Aided Dual Attention Transformer for MS and PAN Classification**. 
There were problems with the initial structure design and code architecture, and I have been modifying the network structure and adjusting hyperparameters for more than a month now.

Each version containing the network structure changes is placed in a "Test x" folder.
# Explanation
My current dataset is MS and PAN images of Hohhot. In the beginning my network's test set OA was only 20%-30%. I have continuously improved my code from the aspects of data construction, network construction, and details such as normalization. 

The changes and problems found in each version are marked in its folder. Found hard defects (such as forgetting to add normalization) are retained in all subsequent versions after modification. Other changes will be retained or deleted according to the effect.

**6.1 Test14 version raised the OA from 30% to 64%**, and I have not calculated the correct rate of other methods. My current task is to raise the project OA to more than 70% (because the current SOTA calculation methods are around 70%), and then use multiple data sets to calculate the accuracy of various categories, AA and Kappa.
