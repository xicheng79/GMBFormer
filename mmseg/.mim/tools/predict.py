import os
import sys
import math

import numpy as np
from osgeo.gdalconst import *
from osgeo import gdal
from tqdm import tqdm
import time
import cv2
import fnmatch
from mmseg.apis import init_model, inference_model
# from logging import logMultiprocessing
# from xml.dom import INDEX_SIZE_ERR
# from syslog import LOG_CRIT
# import skimage.io
# import glob
# import torch
#from torch.autograd import Variable as V
#from PIL import Image
# from data import DataTrainInform

# os.environ['PROJ_LIB'] = r'/root/miniconda3/share/proj'
# os.environ['GDAL_DATA'] = r'/root/miniconda3/share'

def GdalData2OpencvData(GdalImg_data):
    if 'int8' in GdalImg_data.dtype.name:
        OpencvImg_data = np.zeros((GdalImg_data.shape[1], GdalImg_data.shape[2], GdalImg_data.shape[0]), np.uint8)
    elif 'int16' in GdalImg_data.dtype.name:
        OpencvImg_data = np.zeros((GdalImg_data.shape[1], GdalImg_data.shape[2], GdalImg_data.shape[0]), np.uint16)
    else:
        OpencvImg_data = np.zeros((GdalImg_data.shape[1], GdalImg_data.shape[2], GdalImg_data.shape[0]), np.float32)
    for i in range(GdalImg_data.shape[0]):
        OpencvImg_data[:, :, i] = GdalImg_data[GdalImg_data.shape[0] - i - 1, :, :]
    return OpencvImg_data

# 定义block类，处理索引、重叠区等信息
class Block:
    def __init__(self, file, idx_row, idx_col, top_overlap, top_overlap_pic, left_overlap, left_overlap_pic, start_x, start_y):
        self.file = file  # 对应pic的文件路径  
        self.idx_row = idx_row      # 列序号（二维）
        self.idx_col = idx_col      # 行序号（二维）
        self.top_overlap = top_overlap              # 与上一行pic重叠的像素，0为没有重叠
        self.top_overlap_pic = top_overlap_pic      # 与上一行有重叠的pic的序号
        self.left_overlap = left_overlap            # 与左一行pic重叠的像素，0为没有重叠
        self.left_overlap_pic = left_overlap_pic    # 与左一行有重叠的pic的序号
        self.start_x  =  start_x                    # 在整幅影像x方向起点的像素值
        self.start_y  =  start_y                    # 在整幅影像y方向起点的像素值

# 模仿SolverFrame，写一个mmseg的预测类
class MMSegSolver():
    def __init__(self, config_file, model_file):
        self.config_file = config_file
        self.checkpoint_file = model_file
        self.model = init_model(self.config_file, self.checkpoint_file, device='cuda')
    
    # 预测概率，需要在cfg中设置 test_cfg.return_logits == True / 为 Fasle 时输出为类别
    # 也可以，在mmseg中默认输出logits，在外部调用时再根据需求用argmax得到类别
    def predict_x_probs(self, img):
        result = inference_model(self.model, img)
        logits = result.pred_sem_seg.data[0].cpu().numpy()
        res = np.uint8(logits * 255)
        return res

    def predict_x(self, img):
        result = inference_model(self.model, img)
        logits = result.pred_sem_seg.data[0].cpu().numpy()
        return logits

class Predict():
    def __init__(self, target_size, overlap_rate, class_number):
        self.target_size = target_size
        self.overlap_rate = overlap_rate
        self.class_number = class_number
    
    # 预测单个block，预测结果按index序号存为图片格式
    # 返回结果png的文件路径
    def predict_block(self, gd_img_block, predict, save_path, idx_row, idx_col):
        # gdal 转为 cv 数组
        img_block = gd_img_block.transpose(1, 2, 0) # (c, h, w) -> (h, w ,c)
        # img_block = img_block.astype(np.float32)

        img_block = GdalData2OpencvData(gd_img_block)

        # weights = np.zeros((x.shape[0], x.shape[1], class_num), dtype=np.float32)

        # 数据集统计标准化，暂时不用，todo
        # for i in range(self.band_num):
        #     img_block[:, :, i] -= self.img_mean[i]
        # img_block = img_block / self.std
        predict_out = predict(img_block)
        
        # predict_out = predict_out.transpose(1, 2, 0)
        # predict_out = np.argmax(predict_out, axis=2)
        predict_out = np.uint8(predict_out * 255)
        # 存为png，灰度
        # img_out = np.zeros(predict_out.shape + (3,))
        # img_out = img_out.astype(np.int16)
        save_file = os.path.join(save_path, str(idx_row) +'+'+ str(idx_col) + '.png')
        # skimage.io.imsave(save_file, img_out)
        cv2.imwrite(save_file,predict_out)
        return save_file
    
    def predict_block_test_write_tf(self, gd_img_block, predict, save_path, idx_row, idx_col):
        # gdal 转为 cv 数组
        img_block = gd_img_block.transpose(1, 2, 0) # (c, h, w) -> (h, w ,c)
        # img_block = img_block.astype(np.float32)

        # 存为png，灰度
        # img_out = np.zeros(predict_out.shape + (3,))
        # img_out = img_out.astype(np.int16)
        save_file = os.path.join(save_path, str(idx_row) +'+'+ str(idx_col) + '.png')
        # skimage.io.imsave(save_file, img_out)
        cv2.imwrite(save_file,img_block)
        return save_file

    # 调用predict_block函数，生成一幅影像所有block对应的预测png
    def predict_as_blocks(self, dataset, overlap_rate, predict, save_path):
        t0 = time.time()
        img_x = dataset.RasterXSize
        img_y = dataset.RasterYSize
        
        target_size = self.target_size
        space = target_size - int(target_size*overlap_rate)
        # x_num = int((img_x-target_size)/space) + 1 + 1  # x方向上blocks数
        # y_num = int((img_y-target_size)/space) + 1 + 1  # y方向上blocks数
        x_num = math.ceil((img_x-target_size)/space)+1   # x方向上blocks数  向上取整
        y_num = math.ceil((img_y-target_size)/space)+1 # y方向上blocks数
        print("x_num:",x_num)
        print("y_num:",y_num)
        # 改用二维数组，方便索引
        dst_pngs = [[Block for i in range(x_num)] for j in range(y_num)]
        '''分块预测并写为分块png'''
        # 按 左上部-下边缘-右边缘-右下 的顺序存储
        # 左上部 x_num*y_num
        overlap = int(target_size*overlap_rate)
        for j in tqdm(range(0, y_num-1)):
            for i in range(0, x_num-1):
        # for j in tqdm(range(0, img_y - target_size, space)):
        #     for i in range(0, img_x - target_size, space):
                x_start = space*i
                y_start = space*j
                img_block = dataset.ReadAsArray(x_start, y_start, target_size, target_size)
                pic = self.predict_block(gd_img_block = img_block, predict = predict, save_path = save_path, idx_row = j, idx_col = i)
                bk = Block(pic, j, i, overlap, "", overlap, "", x_start, y_start)
                dst_pngs[j][i] = bk
        
        # 下侧边缘
        cur_y = y_num-1
        y_start = img_y - target_size
        overlap_y = space*(cur_y-1) + target_size - y_start
        for i in tqdm(range(0, x_num-1)):
            index = x_num*(cur_y+1) + i
            x_start = space*i
            img_block = dataset.ReadAsArray(x_start, y_start, target_size, target_size)
            pic = self.predict_block(gd_img_block = img_block, predict = predict, save_path = save_path, idx_row = cur_y, idx_col = i)
            bk = Block(pic, cur_y, i, overlap_y, "", overlap , "", x_start, y_start)
            dst_pngs[cur_y][i] = bk

        # 右侧边缘
        cur_x = x_num-1
        x_start = img_x - target_size
        overlap_x = space*(cur_x-1) + target_size - x_start
        for j in tqdm(range(0, y_num-1)):
            index = x_num*j + cur_x + 1
            y_start = space*j
            img_block = dataset.ReadAsArray(x_start, y_start, target_size, target_size)
            pic = self.predict_block(gd_img_block = img_block, predict = predict, save_path = save_path, idx_row = j, idx_col = cur_x)
            bk = Block(pic, j, cur_x, overlap, "", overlap_x , "", x_start, y_start)
            dst_pngs[j][cur_x] = bk

        # 右下角
        index = x_num*cur_y + cur_x
        img_block = dataset.ReadAsArray(img_x-target_size, img_y-target_size, target_size, target_size)
        pic = self.predict_block(gd_img_block = img_block, predict = predict, save_path = save_path, idx_row = cur_y, idx_col = cur_x)
        bk = Block(pic, cur_y, cur_x, overlap_y, "", overlap_x , "",img_x-target_size, img_y-target_size)
        dst_pngs[cur_y][cur_x] = bk

        print('分块预测耗费时间: %0.2f(min).' % ((time.time() - t0) / 60))
        return dst_pngs

    # 为block预测pic创建/修改索引
    def build_pic_index(self, dst_pngs):
        # x_num = dst_pngs[-1][-1].idx_x + 1
        # y_num = dst_pngs[-1][-1].idx_y + 1
        # for i, png in enumerate(dst_pngs):
        for j in dst_pngs:
            for png in j:
                # 第0行（最上）没有top_overlap
                if png.idx_row == 0:
                    png.top_overlap = 0
                else:
                    # 上面的png idx
                    top_y = png.idx_row - 1
                    top_x = png.idx_col
                    #idx = x_num*top_y + top_x
                    png.top_overlap_pic = dst_pngs[top_y][top_x].file

                # 第0列（最左）没有left_overlap
                if png.idx_col == 0:
                    png.left_overlap = 0
                else:
                    # 左边的png idx
                    left_x = png.idx_col - 1
                    left_y = png.idx_row
                    # idx = y_num*left_y + left_x
                    png.left_overlap_pic = dst_pngs[left_y][left_x].file
                # print("1*******1")
                # print(png.file)
                # print("top_overlap:", png.top_overlap)
                # print(png.top_overlap_pic)
                # print("left_overlap:", png.left_overlap)
                # print(png.left_overlap_pic)

    # 拼接预测大图,重叠区内各取一半进行拼接
    def stitch_by_blocks(self, dst_pngs, dst_ds):
        target_size = self.target_size
        '''读取分块png并写入dst_ds'''
        #for i, png in enumerate(dst_pngs):
        # for j in dst_pngs:
        #     for png in j:
        for j in tqdm(dst_pngs, desc='拼接分块png:', unit='batch'):
            for png in j:
                # 处理重叠区
                block = cv2.imread(png.file, cv2.IMREAD_GRAYSCALE)

                # block = block.transpose(1, 2, 0) 
                # 上侧重叠区
                if png.top_overlap > 0:
                    # print("2##########2")
                    top_png = png.top_overlap_pic
                    top_block = cv2.imread(top_png, cv2.IMREAD_GRAYSCALE)
                    overlap = png.top_overlap
                    # 重叠区各取1/2
                    half_overlap = top_block[target_size-overlap:target_size-int(overlap*0.5), 0: target_size]
                    # cv2.imshow("test",half_overlap)
                    # cv2.waitKey(0)
                    block[0:overlap-int(overlap*0.5), 0: target_size] = half_overlap
                # 左侧重叠区
                if png.left_overlap > 0:
                    # print("3+++++++++3")
                    left_png = png.left_overlap_pic
                    left_block = cv2.imread(left_png, cv2.IMREAD_GRAYSCALE)
                    overlap = png.left_overlap
                    # 重叠区各取1/2
                    half_overlap = left_block[0: target_size, target_size-overlap:target_size-int(overlap*0.5), ]
                    # cv2.imshow("test2",half_overlap)
                    # cv2.waitKey(0)
                    block[0: target_size, 0:overlap-int(overlap*0.5)] = half_overlap        
                # 写入：根据每个i 对应block相对整幅影像的xoff和yoff
                dst_ds.GetRasterBand(1).WriteArray(block, png.start_x, png.start_y)
                # 更新：block的png
                # block = block.transpose(1, 2, 0) 
                cv2.imwrite(png.file, block)

        dst_ds.FlushCache() # 缓存写入磁盘
        #dst_ds.close()
        
    def main(self, allpath, outpath, solver, overlap_rate=0.5, target_size=512):  
        print('start predict...')
        for one_path in allpath:
            t0 = time.time()
            ds = gdal.Open(one_path)
            if ds == None:
                print("failed to open img")
                sys.exit(1)
            # 用影像名新建文件夹
            d, n = os.path.split(one_path)
            save_pngs_path = os.path.join(outpath, n)
            if not os.path.exists(save_pngs_path):
                os.makedirs(save_pngs_path)
            dst_pngs = self.predict_as_blocks(dataset =ds, overlap_rate = overlap_rate, predict = lambda xx: solver.predict_x(xx), save_path=save_pngs_path)   
            self.build_pic_index(dst_pngs)
            # for j in dst_pngs:
            #     for png in j:
            #         print("#####")
            #         print("this:",png.file)
            #         print("top_overlap:", png.top_overlap)
            #         print(png.top_overlap_pic)
            #         print("left_overlap:", png.left_overlap)
            #         print(png.left_overlap_pic)
    
            '''新建输出tif'''
            projinfo = ds.GetProjection() 
            geotransform = ds.GetGeoTransform()
            format = "GTiff"
            driver = gdal.GetDriverByName(format)  # 数据格式
            name = n[:-4] + '_result' + '.tif'  # 输出文件名
            outtif = os.path.join(outpath, name)
            dst_ds = driver.Create(outtif, ds.RasterXSize, ds.RasterYSize,
                                1, gdal.GDT_Byte)  # 创建一个新的文件
            dst_ds.SetGeoTransform(geotransform)   # 写入投影
            dst_ds.SetProjection(projinfo)  # 写入坐标
            self.stitch_by_blocks(dst_pngs, dst_ds)

            #写为png
            # outpng = outtif[:-3] + "png"
            # im = Image.open(outtif)
            # im.save(outpng, "PNG")

if __name__ == '__main__':
    predictImgPath = r"G:\GoogelEarthImage\H48F017017" # 待预测影像的文件夹路径
    Img_type = '*.tif' # 待预测影像的类型
    output_path = r"G:\GoogelEarthImage\Predict\20251227\swin\H48F017017" # 输出的预测结果路径
    overlap_rate = 0.2
    target_size = 512
    config_file = r"G:\MMSegmentation1x\Model\20251225\swin-base-upernet_8xb2-320k_voc-binary_greenery_final\swin-base-patch4-window12-in22k-384x384-pre_upernet_8xb2-160k_ade20k-512x512.py"
    model_file = r"G:\MMSegmentation1x\Model\20251225\swin-base-upernet_8xb2-320k_voc-binary_greenery_final\iter_288000.pth" # 模型文件完整路径

    numclass = 2 # 样本类别数
    # model = DLinkNet34 #模型
    band_num = 3 #影像的波段数 训练与预测应一致
    label_norm = True # 是否对标签进行归一化 针对0/255二分类标签 训练与预测应一致

    # 载入模型，mmseg形式，用自己实现的MMSegSolver类
    solver = MMSegSolver(config_file = config_file, model_file= model_file)

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    listpic = fnmatch.filter(os.listdir(predictImgPath), Img_type)
    for i in range(len(listpic)):
        listpic[i] = os.path.join(predictImgPath + '/' + listpic[i])
    
    if not listpic:
        print('listpic is none')
        exit(1)
    else:
        print(listpic)

    predict_instantiation = Predict(target_size = target_size, class_number = numclass, overlap_rate=overlap_rate)
    predict_instantiation.main(listpic, output_path, solver, overlap_rate=overlap_rate, target_size=target_size)