import math
from PIL import Image

# 暴力划分图像
# 按从左到右，从上到下的顺序进行划分
# 每次取256*256图像，取至不足256像素时，从原图像末尾（最右或最下）按反方向取一张256*256图像
# example1: 一张500 * 500 的图像会被划分为4张子图，每两张子图间会有12像素的重叠
# example2：一张1000 * 1000 的图像会被划分为 
def image_split_violent (img, size = 256):
    """
    暴力划分图像

    Parameters:
        img - the origin image
        size - the side length of sub-image, default 256

    Returns:
	    Return a dict, which include the split sub-images and the size of origin image
        example: {'images': [img0, img1, ...], 'size': [10000, 10000]}
    """
    origin_size = img.size  #原图大小
    height = origin_size[1] #原图高
    width = origin_size[0]  #原图宽
    row_num = math.ceil(height / size)
    column_num = math.ceil(width / size)


    # img_list = list(row_num, column_num)
    img_list = [[0 for i in range(column_num)] for i in range(row_num)]
    y = 0   #初始化y值
    for i in range(row_num):
        x = 0   #初始化x值
        for j in range(column_num):
            x0 = x
            x1 = x + size
            y0 = y
            y1 = y + size
            sub_img = img.crop((x0, y0, x1, y1))
            img_list[i][j] = sub_img
            #最后一列划分考虑重叠情况
            if(j == column_num - 1):
                x = width - size
            else:
                x += size
        #最后一行划分考虑重叠情况
        if(i == row_num - 1):
            y = height - size
        else:
            y += size


    result = {'images': img_list, 'origin_size': origin_size}
    return result

def image_split():
    return 0


def image_merge_violent(img_list, origin_size):
    size = img_list[0][0].size[0]
    img = Image.new('RGB', origin_size)

    height = origin_size[1] #原图高
    width = origin_size[0]  #原图宽
    row_num = math.ceil(height / size)
    column_num = math.ceil(width / size)

    y = 0   #初始化y值
    for i in range(row_num):
        x = 0   #初始化x值
        for j in range(column_num):
            
            img.paste(img_list[i][j], (x, y))
            #最后一列划分考虑重叠情况
            if(j == column_num - 1):
                x = width - size
            else:
                x += size
        #最后一行划分考虑重叠情况
        if(i == row_num - 1):
            y = height - size
        else:
            y += size

    return img


def image_merge(img_list):
    return 0
