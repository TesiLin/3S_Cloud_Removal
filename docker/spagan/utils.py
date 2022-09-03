import os
import cv2
import random
import numpy as np

import torch
from torch.backends import cudnn
import matplotlib.pyplot as plt

def gpu_manage(config):
    if config.cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, config.gpu_ids))
        config.gpu_ids = list(range(len(config.gpu_ids)))

    # print(os.environ['CUDA_VISIBLE_DEVICES'])

    if config.manualSeed is None:
        config.manualSeed = random.randint(1, 10000)
    print('Random Seed: ', config.manualSeed)
    random.seed(config.manualSeed)
    torch.manual_seed(config.manualSeed)
    if config.cuda:
        torch.cuda.manual_seed_all(config.manualSeed)

    cudnn.benchmark = True

    if torch.cuda.is_available() and not config.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")


def save_image(out_dir, x, num, epoch, filename=None):
    test_dir = os.path.join(out_dir, 'epoch_{0:04d}'.format(epoch))
    if filename is not None:
        test_path = os.path.join(test_dir, filename)
    else:
        test_path = os.path.join(test_dir, 'test_{0:04d}.png'.format(num))

    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    cv2.imwrite(test_path, x)


def checkpoint(config, epoch, gen, dis):
    model_dir = os.path.join(config.out_dir, 'models')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    net_gen_model_out_path = os.path.join(model_dir, 'gen_model_epoch_{}.pth'.format(epoch))
    net_dis_model_out_path = os.path.join(model_dir, 'dis_model_epoch_{}.pth'.format(epoch))
    torch.save(gen.state_dict(), net_gen_model_out_path)
    torch.save(dis.state_dict(), net_dis_model_out_path)
    print("Checkpoint saved to {}".format(model_dir))


def make_manager():
    if not os.path.exists('.job'):
        os.makedirs('.job')
        with open('.job/job.txt', 'w', encoding='UTF-8') as f:
            f.write('0')


def job_increment():
    with open('.job/job.txt', 'r', encoding='UTF-8') as f:
        n_job = f.read()
        n_job = int(n_job)
    with open('.job/job.txt', 'w', encoding='UTF-8') as f:
        f.write(str(n_job + 1))
    
    return n_job

def heatmap(img):
    if len(img.shape) == 3:
        b,h,w = img.shape
        heat = np.zeros((b,3,h,w)).astype('uint8')
        for i in range(b):
            heat[i,:,:,:] = np.transpose(cv2.applyColorMap(img[i,:,:],cv2.COLORMAP_JET),(2,0,1))
    else:
        b,c,h,w = img.shape
        heat = np.zeros((b,3,h,w)).astype('uint8')
        for i in range(b):
            heat[i,:,:,:] = np.transpose(cv2.applyColorMap(img[i,0,:,:],cv2.COLORMAP_JET),(2,0,1))
    return heat

def save_attention_as_heatmap(filename, att):
    att_heat = heatmap(att)
    cv2.imwrite(filename, att_heat)
    print(filename, 'saved')


def save_criterion_graph(log_dir, train_log, valid_log):
    train_epoch = []
    train_mse = []
    train_psnr = []
    train_ssim = []
    train_lpips = []
    
    for l in train_log.log_:
        train_epoch.append(l['epoch'])
        train_mse.append(l['mse'])
        train_psnr.append(l['psnr'])
        train_ssim.append(l['ssim'])
        train_lpips.append(l['lpips'])

    train_epoch = np.asarray(train_epoch)
    train_mse = np.asarray(train_mse)
    train_psnr = np.asarray(train_psnr)
    train_ssim = np.asarray(train_ssim)
    train_lpips = np.asarray(train_lpips)


    valid_epoch = []
    valid_mse = []
    valid_psnr = []
    valid_ssim = []
    valid_lpips = []
    
    for l in valid_log.log_:
        valid_epoch.append(l['epoch'])
        valid_mse.append(l['mse'])
        valid_psnr.append(l['psnr'])
        valid_ssim.append(l['ssim'])
        valid_lpips.append(l['lpips'])

    valid_epoch = np.asarray(valid_epoch)
    valid_mse = np.asarray(valid_mse)
    valid_psnr = np.asarray(valid_psnr)
    valid_ssim = np.asarray(valid_ssim)
    valid_lpips = np.asarray(valid_lpips)

    plt.subplot(411)
    plt.plot(train_epoch, train_mse, 'b', label='train_mse')
    plt.plot(train_epoch, valid_mse, 'r', label='valid_mse')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()

    plt.subplot(412)
    plt.plot(train_epoch, train_psnr, 'b', label='train_psnr')
    plt.plot(train_epoch, valid_psnr, 'r', label='valid_psnr')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR')
    plt.legend()

    plt.subplot(413)
    plt.plot(train_epoch, train_ssim, 'b', label='train_ssim')
    plt.plot(train_epoch, valid_ssim, 'r', label='valid_ssim')
    plt.xlabel('Epoch')
    plt.ylabel('SSIM')
    plt.legend()

    plt.subplot(414)
    plt.plot(train_epoch, train_lpips, 'b', label='train_lpips')
    plt.plot(train_epoch, valid_lpips, 'r', label='valid_lpips')
    plt.xlabel('Epoch')
    plt.ylabel('LPIPS')
    plt.legend()

    plt.savefig(os.path.join(log_dir, 'train_valid_log.pdf'))
    plt.close()