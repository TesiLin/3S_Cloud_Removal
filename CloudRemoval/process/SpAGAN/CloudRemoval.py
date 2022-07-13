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
    min_overlap = 75        # 两张子图重叠部分的最小宽度

    temp_path = "./temp"    # 图像划分临时文件保存路径
    temp_path_2 = "./temp2" # 图像去云临时文件保存路径
    
    # 预处理临时文件保存路径
    files.del_files(temp_path)
    files.del_files(temp_path_2)

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
            split_result['images'][i][j].save(temp_path + '/' + str(i) + '_' + str(j) + '.png')
    
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

    print(1)
    predict.predict(config, args, gen)
    print("Done.")


    exit(0)






    #图像融合与合并部分
    print("splitting image")
    
    #读取去云后的文件


    if(split_mode == 0):
        merge_result = image_processing.image_split_violent(img, sub_image_size)
    elif (split_mode == 1):
        print("non-brute force mode")
        merge_result = image_processing.image_split(img, sub_image_size, min_overlap)
    
    print("splitting image finished")
    end1 = time.perf_counter()
    print('splitting image Running time: %s Seconds'%(end1-start))




    # 暴力图像合并
    out1 = image_processing.image_merge_violent(out['images'], out['origin_size'])
    print("size of merge image: %s" % str(out1.size))

    # 保存图像
    out1.save(output_path + '/' + file_name)

    
    
    
    end = time.perf_counter()
    print('Running time: %s Seconds'%(end-start))

if __name__ == '__main__':
    #测试
    temp_path = "./temp"    # 图像划分临时文件保存路径
    temp_path_2 = "./temp2" # 图像去云临时文件保存路径
    files.del_files(temp_path)
    files.del_files(temp_path_2)
    print(1)











