from util import image_processing



if __name__ == '__main__':
    import cv2
    from PIL import Image
    import matplotlib.pyplot as plt
    input_path = "./data"
    output_path = "./result"

    # img0 =cv2.imread(input_path + "/0.jpg")
    # img1 =cv2.imread(input_path + "/1.jpg")
    # img2 =cv2.imread(input_path + "/2.jpg")    

    # img0 = Image.open(input_path + "/0.jpg")
    img0 = Image.open(input_path + "/twitter.jpg")
    # img2 = Image.open(input_path + "/2.jpg")
    
    sub_image_size = 256    #子图边长
    
    # 对图片进行预判断
    if(img0.size[0] < sub_image_size or img0.size[1] < sub_image_size):
        print("输入图片size过小")
        exit(0)

    # 暴力划分
    out = image_processing.image_split_violent(img0, sub_image_size)
    print("size of origin image: %s" % str(img0.size))
    print("size of sub-image : %s" % str(out['images'][0][0].size))
    print("size of sub-image list: %s" % str((len(out['images']), len(out['images'][0]))))
    count = 0
    # 保存图像集
    for i in out['images']:
        count0 = 0
        for j in i:
            j.save(output_path + '/list/' + str(count) + '_' + str(count0) + '.jpg')
            count0 += 1
        count += 1


    # 暴力合并
    out1 = image_processing.image_merge_violent(out['images'], out['origin_size'])
    print("size of merge image: %s" % str(out1.size))

    # 保存图像
    out1.save(output_path + '/twitter.jpg')


    

    print("done")