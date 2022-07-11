#改写原论文test文件
#利用PSNR SSIM等指数量化test结果

'''
    测试代码示例:
    CPU:
    python test_quantification.py --dataroot ../divide_multi/combined  --name stgan --model temporal_branched --netG unet_256_independent --dataset_mode temporal --input_nc 3 --gpu_ids -1
    GPU:
    python test_quantification.py --dataroot ../divide_multi/combined  --name stgan --model temporal_branched --netG unet_256_independent --dataset_mode temporal --input_nc 3
'''
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html

from util import util
from util import quantification


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.epoch))  # define the website directory
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    if opt.eval:
        model.eval()

    img_list = []  #储存生成图及原图

    for i, data in enumerate(dataset):
        # print(data.keys())
        # print(data.values())
        # exit(0)

        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths

        # 提取生成图和原图
        fake_img = util.tensor2im(visuals['fake_B'])
        real_img = util.tensor2im(visuals['real_B'])

        img_list.append({'fake': fake_img, 'real': real_img})

        if i % 5 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)


    # 计算平均PSNR峰值信噪比
    psnr = quantification.avg_psnr(img_list)
    # 计算平均SSIM结构相似指数
    ssim = quantification.avg_ssim(img_list)
    # 计算平均LPIPS图像感知相似度指标
    lpips = quantification.avg_lpips(img_list)

    print("final psnr: %f" % psnr)
    print("final ssim: %f" % ssim)
    print("final lpips: %f" % lpips)

    webpage.save()  # save the HTML





