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