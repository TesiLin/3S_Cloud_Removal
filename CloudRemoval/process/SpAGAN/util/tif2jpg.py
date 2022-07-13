


import numpy as np
from osgeo import gdal
import os
from PIL import Image
import argparse



def readTif(fileName, xoff=0, yoff=0, data_width=0, data_height=0):
    '''
    读取tif影像数据
    '''
    dataset = gdal.Open(fileName)
    if dataset == None:
        print(fileName + "文件无法打开")
    #  栅格矩阵的列数
    width = dataset.RasterXSize
    #  栅格矩阵的行数
    height = dataset.RasterYSize
    #  波段数
    bands = dataset.RasterCount
    #  获取数据
    if(data_width == 0 and data_height == 0):
        data_width = width
        data_height = height
    data = dataset.ReadAsArray(xoff, yoff, data_width, data_height)  # np.array
    # print(type(data))
    #  获取仿射矩阵信息
    geotrans = dataset.GetGeoTransform()
    #  获取投影信息
    proj = dataset.GetProjection()
    arrXY = []  # 用于存储每个像素的(x,y)坐标
    for i in range(height):
        row = []
        for j in range(width):
            xx = geotrans[0] + i * geotrans[1] + j * geotrans[2]
            yy = geotrans[3] + i * geotrans[4] + j * geotrans[5]
            col = [xx, yy]
            row.append(col)
        arrXY.append(row)
    return width, height, bands, data, geotrans, proj, arrXY


def tif_to_jpg(src_tifs):
    '''tif转为jpg显示'''
    for src_tif in src_tifs:
        print(readTif(src_tif)[2])
        pre_img = readTif(src_tif)[3]
        

        # print('xxyy.type', type(xxyy))
        pre_img_one = pre_img[4, :, :]# R通道
        pre_img_two = pre_img[3, :, :]# G通道
        pre_img_three = pre_img[2, :, :]#  B通道
        # height, width = pre_img.shape[1:]
        new_array = np.array([pre_img_one, pre_img_two, pre_img_three])#重新将三个通道组成为数组
        new_array = np.transpose(new_array, (1, 2, 0))#将CHW转为HWC
        img = Image.fromarray(np.uint8(new_array))#数组转为图片所用的函数
 
        dir, file_name1 = os.path.split(src_tif)  # split将文件和路径分开
        (prename, suffix) = os.path.splitext(
            file_name1)  # splitext将文件名和后缀分开
        dst_jpg_path = r'./tif_to_rgb/test/'
        dst_jpg = os.path.join(dst_jpg_path, prename+'.jpg')#改为png即可
        print(dst_jpg, img.size)
        img.save(dst_jpg)


parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--filename', type=str, default = 'default')
args = parser.parse_args()
tif_to_jpg([args.filename])




# # coding: utf-8
# import arcpy
# from arcpy import env
# from arcpy import mapping
# import os
# path="D:/test"#读取tif目标路径
# blank_mxd_path = "D:/test/blank.mxd"# 一个空的mxd文件
# target_path="D:/test/out" #转换后的jpg路径
# for file in os.listdir(path):
#     if file[-3:]=="tif":#选择tif格式的图片
#         print (file)
#         # 导入mxd文件，也就是arcmap的保存文件
#         mxd = arcpy.mapping.MapDocument(blank_mxd_path)
#         df = arcpy.mapping.ListDataFrames(mxd)[0]#dataframe没具体意义
#         tif_path = os.path.join(path,file)
#         # 创建raster对象
#         raster = arcpy.Raster(tif_path)
#         arcpy.MakeRasterLayer_management(raster,'rasterLayer')
#         layer = arcpy.mapping.Layer("rasterLayer")# make layer
#         arcpy.mapping.AddLayer(df, layer, "AUTO_ARRANGE")# add layer
#         # mxd.saveACopy("D:/test/test.mxd")
#         # mxd=arcpy.mapping.MapDocument("D:/test/test.mxd")
#         df = arcpy.mapping.ListDataFrames(mxd)[0]
#         new_name=file[:-4]+"jpg"
#         file_target=os.path.join(target_path, new_name)
#         # 导出图片命令
#         arcpy.mapping.ExportToJPEG(mxd, file_target, df, df_export_width=512, df_export_height=512, resolution=300)
#         del mxd, df