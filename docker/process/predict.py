import numpy as np
import argparse

from tqdm import tqdm
import yaml
from attrdict import AttrMap

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from data_manager import TestDataset
from utils import gpu_manage, save_image
from models.gen.SPANet import Generator
from settingObj import settingObj

def predict(config, args, gen):
    dataset = TestDataset(args.test_dir, args.test_file, config.in_ch, config.out_ch)
    dataset_num = len(dataset)
    data_loader = DataLoader(dataset=dataset, num_workers=config.threads, batch_size=1, shuffle=False)
    
    with torch.no_grad():
        with tqdm(total = dataset_num) as pbar:
            for i, batch in enumerate(data_loader):
                # print(i)
                # print(batch)
                # exit()
                # print(type(enumerate(tqdm(data_loader))))
                # print(type(tqdm(data_loader)))
                # print(type(data_loader))
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
                pbar.update(1)
                # print("Img saved.")


def pretrain_load():
    config='./pretrained_models/85_06/config.yml'
    pretrained = './pretrained_models/85_06/gen_model_epoch_200.pth'
    manualSeed = 0

    with open(config, 'r', encoding='UTF-8') as f:
        # config = yaml.load(f, Loader=yaml.FullLoader)
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = AttrMap(config)

    config.cuda = torch.cuda.is_available()
    config.gpu_ids = [0] if torch.cuda.is_available() else None

    args = settingObj(config, pretrained,config.gpu_ids,manualSeed,config.cuda)
    gpu_manage(args)

    ### MODELS LOAD ###
    print('===> Loading models')

    # gen = Generator()
    gen = Generator(gpu_ids=config.gpu_ids)

    # param = torch.load(args.pretrained)
    param = torch.load(args.pretrained, map_location = torch.device("cuda") if config.cuda else torch.device('cpu')) 

    gen.load_state_dict(param)

    if args.cuda:
        gen = gen.cuda(0)
    
    return gen, config, args


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    temp_path = "./temp"    # 图像划分临时文件保存路径
    temp_path_2 = "./temp2" # 图像去云临时文件保存路径
    parser.add_argument('--test_dir', type=str, default=temp_path, required=False)
    parser.add_argument('--test_file', type=str, default=None, required=False)
    parser.add_argument('--out_dir', type=str, default=temp_path_2, required=False)

    input = parser.parse_args()

    gen, config, args = pretrain_load()

    # input_path = Path("./data")

    args.append(input.test_dir, input.test_file, input.out_dir)

    predict(config, args, gen)
    print("Done.")
