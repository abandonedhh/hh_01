import argparse
import json
import os
import time
import jieba
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
import warnings
from myDataset import MyDataset, MyCollate
from util import *

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Irony Detection')
parser.add_argument('--model', default='smc_tc', type=str, help='model name:smc_tc,smc_tc_bga,smc_tc_aoa,smc_tc_lsa,smc_tc_sa,smc_tc_l')
parser.add_argument('--dataset', default='data', type=str)
parser.add_argument('--max_length', default=324, type=int,help='max length')
parser.add_argument('--num_labels', default=2, type=int, help='num labels')
parser.add_argument('--pretrained', default='pretrain/chinese_wwm_ext_pytorch', type=str, help='pretrained model path')
parser.add_argument('--hidden_channels', default=256, type=int, help='hidden channels')
parser.add_argument('--lr', default=3e-5, type=float, help='learning rate')
parser.add_argument('--weight_decay', default=5e-5, type=float, help='weight decay')
parser.add_argument('--batch_size', default=64, type=int, help='batch size')
parser.add_argument('--dropout', default=0, type=float, help='dropout')
parser.add_argument('--init', default='random', type=str, help='xavier,kaiming,normal,random init weight')
parser.add_argument('--epochs', default=100, type=int, help='epochs')
parser.add_argument('--patience', default=10, type=int, help='patience')
parser.add_argument('--seed', default=114514, type=int, help='seed')
parser.add_argument('--device', default='None', type=str, help='device')
args = parser.parse_args()


def get_model(config):
    if config.model == 'smc_tc':
        from model.SMC_TC import SMC_TC
        return SMC_TC(config)
    elif config.model == 'smc_tc_bga':
        from model.SMC_TC_BGA import SMC_TC
        return SMC_TC(config)
    elif config.model == 'smc_tc_aoa':
        from model.SMC_TC_AOA import SMC_TC
        return SMC_TC(config)
    elif config.model == 'smc_tc_lsa':
        from model.SMC_TC_LSA import SMC_TC
        return SMC_TC(config)
    elif config.model == 'smc_tc_sa':
        from model.SMC_TC_SA import SMC_TC
        return SMC_TC(config)
    elif config.model == 'smc_tc_l':
        from model.SMC_TC_L import SMC_TC
        return SMC_TC(config)

class Config():
    def __init__(self):
        self.bert_h = 768
        self.hidden_size = args.hidden_channels
        self.model = args.model
        self.num_labels = args.num_labels
        self.pretrained = args.pretrained
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dropout = args.dropout
        self.EPOCH = args.epochs
        self.max_length = args.max_length
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.seed = args.seed
        self.init = args.init
        self.dataset = args.dataset
        self.patience = args.patience
        self.batch_size = args.batch_size
    def log(self):
        return {'model': self.model,
                'bert_h': self.bert_h,
                'hidden_size': self.hidden_size,
                'num_labels': self.num_labels,
                'pretrained': self.pretrained,
                'device': self.device,
                'dropout': self.dropout,
                'max_length': self.max_length,
                'EPOCH': self.EPOCH,
                'lr': self.lr,
                'weight_decay': self.weight_decay,
                'init': self.init,
                'seed': self.seed,
                'dataset': self.dataset,
                'patience': self.patience,
                'batch_size': self.batch_size
                }


config = Config()  # 配置参数


if __name__ == '__main__':

    start_time = time.time()  # 起始时间

    set_seed(config.seed)  # 固定随机种子

    save_dir = 'log/' + config.dataset+'/'+config.model

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    model_save_path = os.path.join(save_dir, 'best.pth')

    # 数据集加载
    train_dataset = MyDataset('datasets/' + config.dataset + '/train.json')
    dev_dataset = MyDataset('datasets/' + config.dataset + '/dev.json')
    test_dataset = MyDataset('datasets/' + config.dataset + '/test.json')

    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=MyCollate(config).collate_fn,
                                  drop_last=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=MyCollate(config).collate_fn,
                                drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=MyCollate(config).collate_fn,
                                 drop_last=True)

    print(config.device)

    # 模型加载
    model = get_model(config)
    model.to(config.device)

    # 权重初始化
    init_network(model=model, method=config.init)
    # 损失函数
    criterion = nn.CrossEntropyLoss()
    criterion.to(config.device)
    # 优化器
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    best = 0
    count = 0
    end_epoch = config.EPOCH
    for epoch in range(config.EPOCH):
        eval(Epoch=epoch, EPOCHS=config.EPOCH, model=model, data_loader=train_dataloader, device=config.device,
             criterion=criterion, optimizer=optimizer, test=False)
        acc, f1, r, p, dev_ls = eval(Epoch=epoch, EPOCHS=config.EPOCH, model=model, data_loader=dev_dataloader,
                                     device=config.device, criterion=criterion, optimizer=optimizer, test=True)
        tqdm.write('dev acc:{},f1:{},r:{},p:{},loss:{}'.format(acc, f1, r, p, dev_ls))
        if acc > best:
            count = 0
            best = acc
            torch.save(model.state_dict(), model_save_path)
        elif count < config.patience:
            count += 1
        else:
            end_epoch = epoch+1
            break
    model.load_state_dict(torch.load(model_save_path))
    acc, f1, r, p = test(model=model, data_loader=dev_dataloader, device=config.device)
    print('test acc:{},f1:{},r:{},p:{}'.format(acc, f1, r, p))

    end_time = time.time()
    run_time = round(end_time - start_time)
    # 计算时分秒
    hour = run_time // 3600
    minute = (run_time - 3600 * hour) // 60
    second = run_time - 3600 * hour - 60 * minute
    # 输出
    print('运行时间：{}时{}分{}秒'.format(hour, minute, second))
    with open(save_dir + '/{}.json'.format(end_time), 'w', encoding='utf-8') as f:
        json.dump({'performance': {'acc': acc, 'f1': f1, 'r': r, 'p': p},
                   'end_epoch': end_epoch,
                   'log': config.log()
                   }, f,
                  ensure_ascii=False,
                  indent=4)
    save_path = save_dir+'/0best.json'
    if os.path.exists(save_path):
        with open(save_path, 'r', encoding='utf-8') as f:
            best_p = json.load(f)
    else:
        best_p = {'performance': {'acc': 0, 'f1': 0, 'r': 0, 'p': 0}, 'config': config.log()}
    if acc > best_p['performance']['acc']:
        best_p = {'performance': {'acc': acc, 'f1': f1, 'r': r, 'p': p}, 'config': config.log()}
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(best_p, f, ensure_ascii=False, indent=4)
