# 划分数据集单图像或多时像数据集
import os
from shutil import copy
import random


def mkfile(file):
    if not os.path.exists(file):
        os.makedirs(file)

is_multi = 0    # 划分数据类型 0为单图像，1为多时像

file_path = './multipleImage'       #多时像源文件路径
if(not is_multi):
    file_path = './singleImage'     #单图像源文件路径

new_file_path = './divide_multi'    #多时像新文件路径
if(not is_multi):
    new_file_path = './divide_single'    #单图像新文件路径




# 划分比例，训练集 : 验证集 : 测试集 = 8 : 1 : 1
train_rate = 0.8                            #训练集比例
val_rate = 0.1                              #验证集比例
test_rate = 1 - train_rate - val_rate       #测试集比例


#源文件夹
file_path_clear = file_path + '/clear/'
file_path_cloudy = file_path + '/cloudy/'

#新文件夹
train_path_clear = new_file_path + '/clear/train/'
val_path_clear = new_file_path + '/clear/val/'
test_path_clear = new_file_path + '/clear/test/'

train_path_cloudy = new_file_path + '/cloudy/train/'
val_path_cloudy = new_file_path + '/cloudy/val/'
test_path_cloudy = new_file_path + '/cloudy/test/'

#创建所需文件夹
mkfile(train_path_clear)
mkfile(val_path_clear)
mkfile(test_path_clear)
mkfile(train_path_cloudy)
mkfile(val_path_cloudy)
mkfile(test_path_cloudy)


# 开始划分
image_clear = [cla for cla in os.listdir(file_path + '/clear')]     #获取全部clear图像
total_num = len(image_clear)

# 划分数据集index

total_list = range(len(image_clear))
total_num = len(image_clear)

val_index = random.sample(total_list, k=int(total_num * val_rate))  # 从total_list列表中随机抽取k个
new_total_list = [n for i, n in enumerate(total_list) if i not in val_index] #从total_list中剔除val_index
test_index = random.sample(new_total_list, k=int(total_num * test_rate)) # 从new_total_list列表中随机抽取k个
train_index = [n for i, n in enumerate(new_total_list) if n not in test_index]  #从new_total_list中剔除test_index后，剩余部分即为train_index


#划分数据集file
#多时像数据划分
if (is_multi):
    for index, image in enumerate(image_clear):
        #划分验证集
        if index in val_index:
            clear_image_path = file_path_clear + image
            copy(clear_image_path, val_path_clear)  # 将选中的图像复制到新路径
            for i in range(3):
                cloudy_image_path = file_path_cloudy + image[:-4] + '_' + str(i) + '.jpg'
                copy(cloudy_image_path, val_path_cloudy)
        #划分测试集
        elif index in test_index:
            clear_image_path = file_path_clear + image
            copy(clear_image_path, test_path_clear)
            for i in range(3):
                cloudy_image_path = file_path_cloudy + image[:-4] + '_' + str(i) + '.jpg'
                copy(cloudy_image_path, test_path_cloudy)
        #划分训练集
        else:
            clear_image_path = file_path_clear + image
            copy(clear_image_path, train_path_clear)
            for i in range(3):
                cloudy_image_path = file_path_cloudy + image[:-4] + '_' + str(i) + '.jpg'
                copy(cloudy_image_path, train_path_cloudy)
#单图像数据划分
else:
    for index, image in enumerate(image_clear):
        #划分验证集
        if index in val_index:
            clear_image_path = file_path_clear + image
            copy(clear_image_path, val_path_clear)

            cloudy_image_path = file_path_cloudy + image
            copy(cloudy_image_path, val_path_cloudy)
        #划分测试集
        elif index in test_index:
            clear_image_path = file_path_clear + image
            copy(clear_image_path, test_path_clear)

            cloudy_image_path = file_path_cloudy + image
            copy(cloudy_image_path, test_path_cloudy)
        #划分训练集
        else:
            clear_image_path = file_path_clear + image
            copy(clear_image_path, train_path_clear)

            cloudy_image_path = file_path_cloudy + image
            copy(cloudy_image_path, train_path_cloudy)

# 划分测试集


print("processing done!")