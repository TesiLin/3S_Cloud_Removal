import numpy as np
import argparse
from tqdm import tqdm
import yaml
from attrdict import AttrMap

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from data_manager import TestDataset
from utils import gpu_manage, save_image, heatmap
from models.gen.SPANet import Generator


def predict(config, args):
    gpu_manage(args)
    dataset = TestDataset(args.test_dir, args.test_file, config.in_ch, config.out_ch)
    data_loader = DataLoader(dataset=dataset, num_workers=config.threads, batch_size=1, shuffle=False)

    ### MODELS LOAD ###
    print('===> Loading models')

    gen = Generator(gpu_ids=config.gpu_ids)

    param = torch.load(args.pretrained)
    gen.load_state_dict(param)

    if args.cuda:
        gen = gen.cuda(0)

    with torch.no_grad():
        for i, batch in enumerate(tqdm(data_loader)):
            x = Variable(batch[0])
            filename = batch[1][0]
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

            save_image(args.out_dir, allim, i, 4, filename=filename)


if __name__ == '__main__':
    # python predict.py --config pretrained_models/RICE2/config.yml --test_dir ./data/RICE_DATASET/RICE2 --out_dir ./results/test --pretrained ./pretrained_models/RICE2/gen_model_epoch_200.pth --cuda
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--test_dir', type=str, default=None, required=False)
    parser.add_argument('--test_file', type=str, default=None, required=False)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--pretrained', type=str, required=True)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--gpu_ids', type=int, default=[0])
    parser.add_argument('--manualSeed', type=int, default=0)

    args = parser.parse_args()
    
    if (args.test_dir and args.test_file) or (args.test_dir is None and args.test_file is None):
        print("[Error] Please input either test_dir or test_file.")
        exit()

    with open(args.config, 'r', encoding='UTF-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = AttrMap(config)

    predict(config, args)
