import math
import numpy as np

import tensorflow as tf
import lpips


# 使用前需pip install lpips 以及其他所需库

# 计算平均PSNR的值
def avg_psnr(img_list: list):
    """
    计算平均PSNR峰值信噪比的值

    Parameters:
        img_list - a dict list stores fake image and real image pair

    Returns:
	    Return the average psnr of a group of image pairs

    """
    total = 0.0
    count = 0
    for img in img_list:
        count += 1
        total += psnr(img['fake'], img['real'])
    return total / count

# 计算平均SSIM的值
def avg_ssim(img_list: list):
    """
    计算平均SSIM结构相似指数的值

    Parameters:
        img_list - a dict list stores fake image and real image pair

    Returns:
	    Return the average SSIM of a group of image pairs

    """
    total = 0.0
    count = 0
    for img in img_list:
        count += 1
        total += tf.image.ssim(img['fake'], img['real'], 255)

    total = total.numpy()
    return total / count


# 计算平均LPIPS图像感知相似度指标的值
# LPIPS( Learned Perceptual Image Patch Similarity)(通常效果越小越好)
def avg_lpips(img_list: list):
    """
    计算平均LPIPS图像感知相似度指标的值

    Parameters:
        img_list - a dict list stores fake image and real image pair

    Returns:
	    Return the average LPIPS of a group of image pairs

    """
    total = 0.0
    count = 0
    model = util_of_lpips("alex")
    for img in img_list:
        count += 1
        total += model.calc_lpips(img['fake'], img['real'])
    return total / count




# PSNR部分
# 参考 https://blog.csdn.net/weixin_44825185/article/details/107176738

# 计算PSNR的值
def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    # PIXEL_MAX = 1.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))



# lpips部分
# 参考 https://blog.csdn.net/weixin_43466026/article/details/119898304
class util_of_lpips():
    def __init__(self, net, use_gpu=False):
        '''
        Parameters
        ----------
        net: str
            抽取特征的网络，['alex', 'vgg']
        use_gpu: bool
            是否使用GPU，默认不使用
        Returns
        -------
        References
        -------
        https://github.com/richzhang/PerceptualSimilarity/blob/master/lpips_2imgs.py

        '''
        ## Initializing the model
        self.loss_fn = lpips.LPIPS(net=net)
        self.use_gpu = use_gpu
        if use_gpu:
            self.loss_fn.cuda()

    def calc_lpips(self, img1, img2):
        '''
        Parameters
        ----------
        img1_path : str
            图像1的路径.
        img2_path : str
            图像2的路径.
        Returns
        -------
        dist01 : torch.Tensor
            学习的感知图像块相似度(Learned Perceptual Image Patch Similarity, LPIPS).

        References
        -------
        https://github.com/richzhang/PerceptualSimilarity/blob/master/lpips_2imgs.py

        '''
        # Load images
        img1 = lpips.im2tensor(img1)  # RGB image from [-1,1]
        img2 = lpips.im2tensor(img2)

        if self.use_gpu:
            img1 = img1.cuda()
            img2 = img2.cuda()
        dist01 = self.loss_fn.forward(img1, img2)
        return dist01


#ssim部分

# ssim部分 因只能使用1个channel被抛弃
# 参考 https://www.freesion.com/article/93731198856/

# def correlation(img,kernal):
#     kernal_heigh = kernal.shape[0]
#     kernal_width = kernal.shape[1]
#     cor_heigh = img.shape[0] - kernal_heigh + 1
#     cor_width = img.shape[1] - kernal_width + 1
#     result = np.zeros((cor_heigh, cor_width), dtype=np.float64)
#     for i in range(cor_heigh):
#         for j in range(cor_width):
#             result[i][j] = (img[i:i + kernal_heigh, j:j + kernal_width] * kernal).sum()
#     return result
 
# #产生二维高斯核函数
# #这个函数参考自：https://blog.csdn.net/qq_16013649/article/details/78784791
# def gaussian_2d_kernel(kernel_size=11, sigma=1.5):
#     kernel = np.zeros([kernel_size, kernel_size])
#     center = kernel_size // 2
 
#     if sigma == 0:
#         sigma = ((kernel_size - 1) * 0.5 - 1) * 0.3 + 0.8
 
#     s = 2 * (sigma ** 2)
#     sum_val = 0
#     for i in range(0, kernel_size):
#         for j in range(0, kernel_size):
#             x = i - center
#             y = j - center
#             kernel[i, j] = np.exp(-(x ** 2 + y ** 2) / s)
#             sum_val += kernel[i, j]
#     sum_val = 1 / sum_val
#     return kernel * sum_val
 


# # ssim模型
# def ssim(distorted_image,original_image,window_size=11,gaussian_sigma=1.5,K1=0.01,K2=0.03,alfa=1,beta=1,gama=1):
#     distorted_image=np.array(distorted_image,dtype=np.float64)
#     original_image=np.array(original_image,dtype=np.float64)
#     if not distorted_image.shape == original_image.shape:
#         raise ValueError("Input Imagees must has the same size")
#     if len(distorted_image.shape) > 2:
#         raise ValueError("Please input the images with 1 channel")
#     kernal=gaussian_2d_kernel(window_size,gaussian_sigma)
 
#     #求ux uy ux*uy ux^2 uy^2 sigma_x^2 sigma_y^2 sigma_xy等中间变量
#     ux=correlation(distorted_image,kernal)
#     uy=correlation(original_image,kernal)
#     distorted_image_sqr=distorted_image**2
#     original_image_sqr=original_image**2
#     dis_mult_ori=distorted_image*original_image
#     uxx=correlation(distorted_image_sqr,kernal)
#     uyy=correlation(original_image_sqr,kernal)
#     uxy=correlation(dis_mult_ori,kernal)
#     ux_sqr=ux**2
#     uy_sqr=uy**2
#     uxuy=ux*uy
#     sx_sqr=uxx-ux_sqr
#     sy_sqr=uyy-uy_sqr
#     sxy=uxy-uxuy
#     C1=(K1*255)**2
#     C2=(K2*255)**2
#     #常用情况的SSIM
#     if(alfa==1 and beta==1 and gama==1):
#         ssim=(2*uxuy+C1)*(2*sxy+C2)/(ux_sqr+uy_sqr+C1)/(sx_sqr+sy_sqr+C2)
#         return np.mean(ssim)
#     #计算亮度相似性
#     l=(2*uxuy+C1)/(ux_sqr+uy_sqr+C1)
#     l=l**alfa
#     #计算对比度相似性
#     sxsy=np.sqrt(sx_sqr)*np.sqrt(sy_sqr)
#     c=(2*sxsy+C2)/(sx_sqr+sy_sqr+C2)
#     c=c**beta
#     #计算结构相似性
#     C3=0.5*C2
#     s=(sxy+C3)/(sxsy+C3)
#     s=s**gama
#     ssim=l*c*s
#     return np.mean(ssim)








