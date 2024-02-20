import torch
import torch.nn as nn
import cv2
import numpy as np
import torchvision.transforms as transforms
import os
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR
from sklearn.metrics import roc_auc_score
import time
from tqdm import tqdm
from model import *
from dataloder import *
from config import cfg


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_dataset = CancerDataset(base_dir=cfg['train_dir'], label_dir=cfg['train_label_dir'],transform=data_transforms)
train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True)

valid_dataset =CancerDataset(base_dir=cfg['valid_dir'],label_dir=cfg['valid_label_dir'],transform=data_transforms)
val_loader = DataLoader(valid_dataset,batch_size=cfg['batch_size'], shuffle=True)

test_dataset =CancerDataset(base_dir=cfg['test_dir'],label_dir=cfg['test_label_dir'],transform=data_transforms)
test_loader =DataLoader(test_dataset,batch_size=cfg['batch_size'])

model_conv =CSAM_ResNet(pretrained=False)
model_conv.fc =nn.Sequential(
    nn.Dropout(0.6),
    nn.Linear(2048,1024,bias=True),
    nn.SELU(),
    nn.Dropout(0.8),
    nn.Linear(1024,2,bias=True),
)
model_conv = model_conv.to(device)

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

loss_fn =nn.CrossEntropyLoss()

loss_fn = loss_fn.to(device)
optimizer = torch.optim.Adam(model_conv.parameters(), lr=0.0004)  #1e-2 =0.01
lr_scheduler = ReduceLROnPlateau(optimizer, mode='min',factor=0.2, patience=5,verbose=1)


def fit(epoch, model):
    correct = 0
    total = 0
    running_loss = 0
    model.train()
    train_auc =[]
    val_auc=[]
    global val_acc_max
    global val_auc_max
    loop = tqdm(enumerate(train_loader), total=len(train_loader),ncols=80)
    for i,(x, y) in loop:
        x, y = x.to(device), y.to(device)
        y_pred = model(x)
        y_pred = y_pred.to(device)
        loss = loss_fn(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        a =y.data.cpu().numpy()
        try:
            b =y_pred[:,1].detach().cpu().numpy()
            train_auc.append(roc_auc_score(a ,b))
        except:
            pass

        with torch.no_grad():
            y_pred = torch.argmax(y_pred, dim=1)
            correct += (y_pred == y).sum().item()
            total += y.size(0)
            running_loss += loss.item()
        acc =correct/len(x)
        loop.set_description(f'Epoch [{epoch}/{cfg["num_epochs"]}]')
        loop.set_postfix(loss=running_loss / (i + 1), acc=float(correct) / float(cfg["batch_size"]* i + len(x))) #因为i是从0开始的 所以额外加一个batch_size

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total

        if (i+1)%600 == 0:
            test_correct = 0
            test_total = 0
            test_running_loss = 0
            print("eval")
            model.eval() 
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    y_pred = model(x)
                    y_pred = y_pred.to(device)
                    loss = loss_fn(y_pred, y)

                    a = y.data.cpu().numpy()
                    try:
                        b = y_pred[:, 1].detach().cpu().numpy()
                        val_auc.append(roc_auc_score(a, b))
                    except:
                        pass
                    y_pred = torch.argmax(y_pred, dim=1)
                    test_correct += (y_pred == y).sum().item()
                    test_total += y.size(0)
                    test_running_loss += loss.item()


            epoch_test_loss = test_running_loss / len(val_loader)
            lr_scheduler.step(epoch_test_loss)
            epoch_test_acc = test_correct / test_total

            if round(epoch_test_acc,4) > val_acc_max:
                print("max:{}-->epoch_test_acc:{}".format(val_acc_max, epoch_test_acc))
                torch.save(model_conv.state_dict(), cfg['save_path'])
                val_acc_max = round(epoch_test_acc,4)

            if round(np.mean(val_auc),4) >val_auc_max:
                print("max:{}-->epoch_test_auc:{}".format(val_auc_max, np.mean(val_auc)))
                torch.save(model_conv.state_dict(), cfg['save_path2'])
                val_auc_max = round(np.mean(val_auc),4)


    print(
    'val_auc:',np.mean(val_auc),
    'val_loss:', round(epoch_test_loss, 4),
    'val_accuracy:', round(epoch_test_acc, 4)
    )


    print('epoch: ', epoch,
          'train_loss:', round(epoch_loss, 4), #四舍五入 且保留小数点后3位
          'train_accuracy:', round(epoch_acc, 4),
          'train_auc:',np.mean(train_auc)
          )


epoch_num =[i+1 for i in range(cfg['num_epochs'])]
for epoch in range(cfg['num_epochs']):
    fit(epoch, model_conv)
    torch.save(model_conv.state_dict(), cfg["model_path"])


