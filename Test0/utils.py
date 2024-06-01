import numpy as np
import torch
from torch.nn import functional as F
import random
from tqdm import tqdm
import logging
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



#定义训练方法
def train_model(model, train_loader, optimizer, epoch, lamda0 = 1, lamda1 = 0.0001, lamda2 = 0.0001):
    loop = tqdm(train_loader, leave=True)
    model.train()
    correct = 0.0
    for step, (ms, fu, pan, label, _) in enumerate(loop):
        ms, fu, pan, label = ms.cuda(), fu.cuda(), pan.cuda(), label.cuda()
        optimizer.zero_grad()
        output, mse_msf, mse_pan = model(ms, fu, pan)
        pred_train = output.max(1, keepdim=True)[1]
        correct += pred_train.eq(label.view_as(pred_train).long()).sum().item()
        mse_msf = torch.tensor(mse_msf).to(torch.float32).cuda()
        mse_pan = torch.tensor(mse_pan).to(torch.float32).cuda()
        loss = lamda0 * F.cross_entropy(output, label.long()) + lamda1 * torch.sum(mse_msf) + lamda2 * torch.sum(mse_pan)
        loss.backward()
            #定义优化
        optimizer.step()
        loop.set_postfix(loss=loss, epoch=epoch, accuracy=correct * 100.0 / len(train_loader.dataset), mode='train')
        if step % 100 == 0:
            print("Train Epoch: {} \t Loss : {:.6f} \t step: {} \t Train Accuracy: {:.6f} ".format(epoch, loss.item(), step, correct * 100.0 / len(train_loader.dataset)))
            logging.info("Train Epoch: {} \t Loss : {:.6f} \t step: {} \t Train Accuracy: {:.6f} ".format(epoch, loss.item(), step, correct * 100.0 / len(train_loader.dataset)))
    loop.close()
    print("Train Accuracy: {:.6f}".format(correct * 100.0 / len(train_loader.dataset)))
    logging.info("Train Accuracy: {:.6f}".format(correct * 100.0 / len(train_loader.dataset)))

#定义测试方法
def test_model(model, test_loader):
    loop = tqdm(test_loader, leave=True)
    model.eval()
    correct = 0.0
    test_loss = 0.0
    #_, criterion_val = get_loss(loss_mode)
    with torch.no_grad():
        for ms, fu, pan, target, _ in loop:
            ms, fu, pan, target = ms.cuda(), fu.cuda(), pan.cuda(), target.cuda()
            output, _, _ = model(ms, fu, pan)
            test_loss += F.cross_entropy(output, target.long()).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred).long()).sum().item()
        test_loss = test_loss / len(test_loader.dataset)
        
        loop.close()        
        print("test-average loss: {:.4f}, Accuracy:{:.3f} \n".format(
            test_loss, 100.0 * correct / len(test_loader.dataset)))
        logging.info("test-average loss: {:.4f}, Accuracy:{:.3f} \n".format(test_loss, 100.0 * correct / len(test_loader.dataset)))
        with open('output.txt', 'w') as f:
            print("test-average loss: {:.4f}, Accuracy:{:.3f} \n".format(
            test_loss, 100.0 * correct / len(test_loader.dataset)), file=f)