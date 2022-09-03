import os
import random
import shutil
import yaml
from attrdict import AttrMap
import time
from tqdm import tqdm

import torch
from torch import nn
from torch.backends import cudnn
from torch import optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn import functional as F

from data_manager import TrainDataset
from models.gen.SPANet import Generator
from models.dis.dis import Discriminator
import utils
from utils import gpu_manage, save_image, checkpoint
from eval import test
from log_report import LogReport
from log_report import TestReport
import argparse
# 通过jobnum指定测试模型文件夹
# 通过epoch指定模型
# 输出在对应jobnum下

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--jobnum', type=int, default=0)
    parser.add_argument('--epoch', type=int, default=200)
    args = parser.parse_args()

    filepath = "./results/%06d/" % args.jobnum
    gen_modelname = filepath + "models/gen_model_epoch_%d.pth" % args.epoch
    dis_modelname = filepath + "models/dis_model_epoch_%d.pth" % args.epoch
    
    with open(filepath+'config.yml', 'r', encoding='UTF-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = AttrMap(config)

    gpu_manage(config)

    ### DATASET LOAD ###
    print('===> Loading datasets')

    dataset = TrainDataset(config)
    print('dataset:', len(dataset))
    train_size = int(config.train_size * len(dataset))
    validation_size = int(config.validation_size * len(dataset))
    test_size = len(dataset)-train_size-validation_size  # int(config.test_size * len(dataset))

    train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, validation_size, test_size], generator=torch.Generator().manual_seed(config.manualSeed))
    print('train dataset:', len(train_dataset))
    print('validation dataset:', len(validation_dataset))
    print('test dataset:', len(test_dataset))
    
    test_data_loader = DataLoader(dataset=test_dataset, num_workers=config.threads, batch_size=config.batchsize, shuffle=True)
    
    ### MODELS LOAD ###
    print('===> Loading models')

    gen = Generator(gpu_ids=config.gpu_ids)

    param = torch.load(gen_modelname)
    gen.load_state_dict(param)
    print('load {} as gen model'.format(gen_modelname))

    dis = Discriminator(in_ch=config.in_ch, out_ch=config.out_ch, gpu_ids=config.gpu_ids)

    param = torch.load(dis_modelname)
    dis.load_state_dict(param)
    print('load {} as dis model'.format(dis_modelname))

    criterionMSE = nn.MSELoss()

    if config.cuda:
        gen = gen.cuda(0)
        criterionMSE = criterionMSE.cuda()

    ### START TEST ###
    print('===> Test starts')
    testreport = TestReport(log_dir=filepath)
    with torch.no_grad():
        log_test = test(config, test_data_loader, gen, criterionMSE, config.epoch)
        testreport(log_test)
    print("===> Done")