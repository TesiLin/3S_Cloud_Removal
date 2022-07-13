from util import image_processing
from util import quantification
import CloudRemoval
from PIL import Image


import os
import datetime
import time
import cv2
# import matplotlib.pyplot as plt

if __name__ == '__main__':

    # 参数部分
    input_path = "./data"
    output_path = "./result"
    file_name = "Hurricane_Dorian_pair2_pre.png"
    # file_name = "Hurricane_Laura_pair6_post.png"
    # file_name = "Hurricane_Laura_pair7_post.png"
    list_name = file_name[:-4] + "_list"

    sub_image_size = 256    #子图边长
    min_overlap = 75 # 两张子图重叠部分的最小宽度


    # 读取图片
    img0 = Image.open(input_path + "/" + file_name)
    result = CloudRemoval.CloudRemoval(img0)
    print(result)
    exit(0)


    # 测试
    # img0 =cv2.imread(input_path + "/Hurricane_Dorian_pair2_pre.png")
    # img1 =cv2.imread(output_path + "/Hurricane_Dorian_pair2_pre.png")
    # img_list = [{'fake' : img0, 'real' : img1}]
    # print("psnr: %f" % quantification.avg_psnr(img_list))
    # exit(0)
    # print("ssim: %f" % quantification.avg_ssim(img_list))
    # print("lpips: %f" % quantification.avg_lpips(img_list))




    # img2 =cv2.imread(input_path + "/2.jpg")    
    # img0 = Image.open(input_path + "/Hurricane_Dorian_pair2_pre.png")
    # img0 = tif2jpg.readTif(input_path + "/Hurricane_Dorian_pair2_pre.png")
    # img0 = Image.open(input_path + "/Hurricane_Laura_pair7_post.jpg")
    
    # 创建子图保存文件夹
    if os.path.exists(output_path + '/' + list_name):
        os.removedirs(output_path + '/' + list_name)
    os.mkdir(output_path + '/' + list_name)

    # 读取图片
    img0 = Image.open(input_path + "/" + file_name)

    # 对图片进行预判断
    if(img0.size[0] < sub_image_size or img0.size[1] < sub_image_size):
        print("输入图片size过小")
        exit(0)











# 暴力图像划分
    # print("brute force part")

    # start = time.perf_counter()
    # print("splitting image")

    # out = image_processing.image_split_violent(img0, sub_image_size)
    
    # print("splitting image finished")
    # end1 = time.perf_counter()
    # print('splitting image Running time: %s Seconds'%(end1-start))


    # print("size of origin image: %s" % str(img0.size))
    # print("size of sub-image : %s" % str(out['images'][0][0].size))
    # print("size of sub-image list: %s" % str((len(out['images']), len(out['images'][0]))))
    
    # 保存图像集
    # print("saving sub-images")

    # for i in range(len(out['images'])):
    #     for j in range(len(out['images'][0])):
    #         out['images'][i][j].save(output_path + '/' + list_name + '/' + str(i) + '_' + str(j) + '.png')
    # print("saving finished")


#



# 暴力图像合并
    # out1 = image_processing.image_merge_violent(out['images'], out['origin_size'])
    # print("size of merge image: %s" % str(out1.size))

    # # 保存图像
    # out1.save(output_path + '/' + file_name)

    
    
    
    # end = time.perf_counter()
    # print('Running time: %s Seconds'%(end-start))



#     print("\n\nnon-brute part")
# # 图像划分
#     out2 = image_processing.image_split(img0, sub_image_size)
#     print("size of origin image: %s" % str(img0.size))
#     print("size of sub-image : %s" % str(out2['images'][0][0].size))
#     print("size of sub-image list: %s" % str((len(out2['images']), len(out2['images'][0]))))
    
#     # 保存图像集
#     for i in range(len(out2['images'])):
#         for j in range(len(out2['images'][0])):
#             out2['images'][i][j].save(output_path + '/list2/' + str(i) + '_' + str(j) + '.jpg')


# # 图像合并
#     out3 = image_processing.image_merge(out2['images'], out2['locations'], out2['origin_size'])
#     print("size of merge image: %s" % str(out3.size))

#     # 保存图像
#     out3.save(output_path + '/twitter2.jpg')

    

#     print("done")