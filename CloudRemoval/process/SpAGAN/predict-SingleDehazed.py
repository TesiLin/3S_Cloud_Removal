import numpy as np
import argparse
import yaml
from attrdict import AttrMap
import os
import cv2

import torch
from torch.autograd import Variable

from utils import gpu_manage, save_image
from models.gen.SPANet import Generator
from settingObj import settingObj

def predict(config, args, gen):
    filename = os.path.basename(args.test_file)
    x = cv2.imread(args.test_file, 1).astype(np.float32)
    x = x / 255   
    x = x.transpose(2, 0, 1)
    x = Variable(torch.from_numpy(x).unsqueeze(0))
    with torch.no_grad():
        if args.cuda:
            x = x.cuda()
        
        att, out = gen(x)

        c = 3
        w = config.width
        h = config.height

        # version2
        allim = np.zeros((c, h, w))
        out_ = out.cpu().numpy()[0]
        out_rgb = np.clip(out_[:3], 0, 1)
        
        allim[:] = out_rgb * 255
        allim = allim.transpose(1, 2, 0)
        allim = allim.reshape((h, w, c))

        save_image(args.out_dir, allim, 0, 4, filename=filename)
        print("Img saved.")

def pretrain_load():
    config='./pretrained_models/RICE2/config.yml'
    pretrained = './pretrained_models/RICE2/gen_model_epoch_200.pth'
    gpu_ids = [0]
    manualSeed = 0
    cuda = True

    with open(config, 'r', encoding='UTF-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = AttrMap(config)

    args = settingObj(config, pretrained,gpu_ids,manualSeed,cuda)
    gpu_manage(args)

    ### MODELS LOAD ###
    print('===> Loading models')

    gen = Generator(gpu_ids=config.gpu_ids)

    param = torch.load(args.pretrained)
    gen.load_state_dict(param)

    if args.cuda:
        gen = gen.cuda(0)
    
    return gen, config, args


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_file', type=str, default=None, required=False)
    parser.add_argument('--out_dir', type=str, required=True)

    input = parser.parse_args()

    gen, config, args = pretrain_load()

    args.append(test_file=input.test_file, out_dir=input.out_dir)

    predict(config, args, gen)

    print("Done.")
