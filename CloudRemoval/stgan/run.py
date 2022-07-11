#改写原论文test文件
#运行模型，输入一张有云图，返回一张无云图


import os
from test.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html

from util import util
from util import quantification

#   去云函数
#   单图像 mode为0 image输入一张云图
#   多时像 mode为1 image输入一个三张云图的数组
def removeCloud(image, mode = 1):
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test

    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    
    # fix the model

    opt.dataroot = "/"

    input = dict()
    if(mode == 1):
        opt.name = "stgan"
        opt.netG = "unet_256_independent"
        opt.model = "temporal_branched"
        opt.dataset_mode = "temporal"
        # opt.gpu_ids = -1
        if(opt.gpu_ids):
            print("Aaa")
        else:
            print("Bbb")
        # exit()
        input = {"A_0": image[0], "A_1": image[1], "A_2": image[2]}


    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)  
    # print(input)
    # exit()
    
    if opt.eval:
        model.eval()
    
    

    model.set_input(image)  # unpack data from data loader
    model.test()           # run inference
    result = model.predict()
    
    visuals = model.get_current_visuals()  # get image results
    img_path = model.get_image_paths()     # get image paths

    return visuals

    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break

        


        # 提取生成图和原图
        fake_img = util.tensor2im(visuals['fake_B'])
        real_img = util.tensor2im(visuals['real_B'])

        img_list.append({'fake': fake_img, 'real': real_img})

        if i % 5 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)





if __name__ == '__main__':
    import cv2
    input_path = "./test"
    output_path = "./test"

    img0 =cv2.imread(input_path + "/0.jpg")
    img1 =cv2.imread(input_path + "/1.jpg")
    img2 =cv2.imread(input_path + "/2.jpg")
    
    image = [img0, img1, img2]

    
    print(removeCloud(image, mode = 1))
    print('done')





