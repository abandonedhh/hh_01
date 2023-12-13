import os
import random

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from torch import nn
from tqdm import tqdm


def eval(Epoch, EPOCHS, model, test, data_loader, criterion, optimizer, device):

    running_loss = 0.0
    pred, y = [], []
    if test:
        model.eval()
        with torch.no_grad():
            for step, data in enumerate(data_loader):
                data = list(data)
                to_device(data, device)
                out = model(data)
                loss = criterion(out, data[0])
                running_loss += loss.cpu().detach().numpy().item()

                pred += out.argmax(dim=1).cpu().numpy().tolist()
                y += data[0].cpu().numpy().tolist()

                return (accuracy_score(y_true=y, y_pred=pred),
                        f1_score(y_true=y, y_pred=pred, average='macro'),
                        recall_score(y_true=y, y_pred=pred, average='macro'),
                        precision_score(y_true=y, y_pred=pred, average='macro'),
                        running_loss / len(data_loader)
                        )
    else:
        loop = tqdm(enumerate(data_loader), total=len(data_loader))
        model.train()
        for step, data in loop:
            data = list(data)
            to_device(data, device)

            out = model(data)

            loss = criterion(out, data[0])
            running_loss += loss.cpu().detach().numpy().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.cpu().detach().numpy().item()
            # 累加识别正确的样本数
            pred += out.argmax(dim=1).cpu().numpy().tolist()
            y += data[0].cpu().numpy().tolist()

            # 更新信息
            loop.set_description(f'train Epoch [{Epoch}/{EPOCHS}]')
            loop.set_postfix(loss=running_loss / (step + 1),
                             acc=accuracy_score(y_true=y, y_pred=pred),
                             f1=f1_score(y_true=y, y_pred=pred, average='macro'),
                             recall=recall_score(y_true=y, y_pred=pred, average='macro'),
                             precision=precision_score(y_true=y, y_pred=pred, average='macro')
                             )
        return running_loss / len(data_loader)


def test(model, data_loader, device):
    pred, y = [], []
    model.eval()

    for data in data_loader:
        data = list(data)
        to_device(data,device)
        out = model(data)

        pred += out.argmax(dim=1).cpu().numpy().tolist()
        y += data[0].cpu().numpy().tolist()
    return (accuracy_score(y_true=y, y_pred=pred),
            f1_score(y_true=y, y_pred=pred, average='macro'),
            recall_score(y_true=y, y_pred=pred, average='macro'),
            precision_score(y_true=y, y_pred=pred, average='macro'),
            )


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


def to_device(data, device):
    data[0] = data[0].to(device)
    for i, it in enumerate(data[1:]):
        for j, jt in enumerate(it):
            data[i + 1][j] = jt.to(device)

def init_network(model, method='xavier', exclude='pretrained'):
    for name, w in model.named_parameters():
        if exclude not in name:
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                elif method == 'normal':
                    nn.init.normal_(w)
                elif method == 'random':
                    return
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass