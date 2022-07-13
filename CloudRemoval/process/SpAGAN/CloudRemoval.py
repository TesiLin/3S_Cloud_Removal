from PIL import Image
from util import image_processing
from util import files
import predict


import time
import os

def CloudRemoval(img):
    """
    图像去云

    Parameters:
        img - the origin image

    Returns:
	    Return the image without cloud
    """
    # 参数部分
    sub_image_size = 512    # 子图边长
    split_mode = 0          # 图像划分模式，0为暴力划分，1为重叠划分
    min_overlap = 75        # 两张子图重叠部分的最小宽

    #注意 此处文件路径尽量不要更改
    temp_path = "./temp"    # 图像划分临时文件保存路径
    temp_path_2 = "./temp2" # 图像去云临时文件保存路径


    
    # 预处理临时文件保存路径
    # files.del_files(temp_path)
    # files.del_files(temp_path_2)
    # os.mkdir(temp_path + "/cloudy_image")

    files.del_files(temp_path)
    files.del_files(temp_path_2)
    # if os.path.exists(temp_path):
    #     os.removedirs(temp_path) 
    # if os.path.exists(temp_path_2):
    #     os.removedirs(temp_path_2)
    # os.mkdir(temp_path)
    # os.mkdir(temp_path_2)
    # os.mkdir(temp_path + "/cloudy_image")

    # 图像划分部分
    
    start = time.perf_counter()
    print("splitting image")

    if(split_mode == 0):
        print("brute force mode")
        split_result = image_processing.image_split_violent(img, sub_image_size)
    elif (split_mode == 1):
        print("non-brute force mode")
        split_result = image_processing.image_split(img, sub_image_size, min_overlap)
    
    print("splitting image finished")
    end1 = time.perf_counter()
    print('splitting image Running time: %s Seconds'%(end1-start))

    
    print("saving sub-images")

    for i in range(len(split_result['images'])):
        for j in range(len(split_result['images'][0])):
            split_result['images'][i][j].save(temp_path + '/cloudy_image/' + str(i) + '_' + str(j) + '.png')
    
    print("saving sub-images finished")
    end2 = time.perf_counter()
    print('saving sub-images Running time: %s Seconds'%(end2- end1))



    # 图像去云部分

    parser = predict.argparse.ArgumentParser()
    parser.add_argument('--test_dir', type=str, default=temp_path, required=False)
    parser.add_argument('--test_file', type=str, default=None, required=False)
    parser.add_argument('--out_dir', type=str, default=temp_path_2, required=False)
    # parser.add_argument('--gpu_ids', type=str, default='-1', required=False)
    

    input = parser.parse_args()

    gen, config, args = predict.pretrain_load()

    args.append(input.test_dir, input.test_file, input.out_dir)

    predict.predict(config, args, gen)
    print("Done.")


    






    #图像融合与合并部分
    print("merging image")
    end3 = time.perf_counter()
    #读取去云后的文件
    to_merge_list = [[0 for i in range(len(split_result['images'][0]))] for i in range(len(split_result['images']))]
    for i in range(len(split_result['images'])):
        for j in range(len(split_result['images'][0])):
            image = Image.open(temp_path_2 + "/epoch_0004/" + str(i) + '_' + str(j) + ".png")
            to_merge_list[i][j] = image

    #合并图像
    if(split_mode == 0):
        merge_result = image_processing.image_merge_violent(to_merge_list, split_result['origin_size'])
    elif (split_mode == 1):
        print("non-brute force mode")
        merge_result = image_processing.image_merge(to_merge_list, split_result['locations'], split_result['origin_size'])
    
    print("merging image finished")
    end4 = time.perf_counter()
    print('merging image Running time: %s Seconds'%(end4 - end3))

    print("size of merge image: %s" % str(merge_result.size))
    print('total Running time: %s Seconds'%(end4 - start))
    return merge_result


# if __name__ == '__main__':
#     #测试
#     temp_path = "./temp"    # 图像划分临时文件保存路径
#     temp_path_2 = "./temp2" # 图像去云临时文件保存路径
#     files.del_files(temp_path)
#     files.del_files(temp_path_2)
#     print(1)











