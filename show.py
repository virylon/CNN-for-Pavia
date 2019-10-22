# -*- coding: utf-8 -*-
"""
Created on Sat May 11 19:44:29 2019

@author: viryl
"""
import os
import numpy as np
import spectral  # 专门为光谱图像设计的包
import PIL
import scipy.io
import torch
from scipy.io import loadmat
from functools import reduce
from net import Net
from utils import pad, pca, standartize, patch

# ----------获取原始数据---------------
DATA_PATH = os.path.join(os.getcwd(), "PaviaU")                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
data = loadmat(os.path.join(
    DATA_PATH, 'PaviaU.mat'))
data = data[list(data.keys())[-1]]
data = np.transpose(data, (2, 0, 1))  # 通道数提前
print(data.shape)
slice_size = 17  # 移动视窗尺寸 与切片尺寸一致
outputs = np.zeros((data.shape[1],
                    (data.shape[2])))  # 创建等尺寸的 0矩阵接收预测值
data = standartize(data)  # 标准化
data = pad(data, 8)  # 填充边框 与切片填充大小一致
# -----------生成网络模型---------------
logdir = "10-12_18-51-02"  # 日志文件
NET_PARAMS_PATH = os.path.join(os.getcwd(), "log", logdir, "net_params.pkl")
net_params = torch.load(NET_PARAMS_PATH)  # 加载训练好的模型参数
cnn = Net()  # 生成网络
cnn=cnn.double()
cnn.load_state_dict(net_params)  # 参数放进网络
if torch.cuda.is_available():  # 使用GPU
    cnn = cnn.cuda()

# -----------将数据输入模型-------------
for h in range(data.shape[1]-slice_size+1):
    for w in range(data.shape[2]-slice_size+1):
        img = patch(data, slice_size, h, w)
        img = np.expand_dims(img, axis=0)  # 再增加一个维度（模型接受三维数据）
        img = torch.from_numpy(img).double()
        if torch.cuda.is_available():
            img = img.cuda()
        output = cnn(img)
        _, predicted = torch.max(output.data, 1)
        outputs[h, w] = np.array(predicted.cpu())[0]

# -----------将target保存为图片------------
pic_name = logdir+".tif"
SAVE_PATH = os.path.join(os.getcwd(), "predicted", pic_name)
PIL.Image.fromarray(outputs).save(SAVE_PATH)


# PIL.Image.fromarray(target_mat).save('D:/Code/test.tif')

# files = os.listdir(DATA_PATH)
# for file in files:
#     image = PIL.Image.open(os.path.join(DATA_PATH, file))
#     width = image.size[0]
#     height = image.size[1]
#     for h in range(0, height, 16):
#         for w in range(0, width, 16):
#             img = image.crop((h, w, h+16, w+16))
#             img.save(os.path.join(DATA_PATH, ('(%d,%d)'+file) % (h, w)))


# target_mat=data[64]#最后一维的数据是标记的样本
# print(data.shape)
# print(target_mat.shape)
# print("有 %d 个波段"%input_mat.shape[0])
# 将target保存为图片
# PIL.Image.fromarray(target_mat).save('D:/Code/data/gl.tif')
#
# 统计每类样本所含个数
# dick_k={}
# for i in range(target_mat.shape[0]):
#    for j in range(target_mat.shape[1]):
#        if target_mat[i][j] in [m for m in range(1,22)]:#输出22个类
#            if target_mat[i][j] not in dick_k:
#                dick_k[target_mat[i][j]]=0
#            dick_k[target_mat[i][j]]+=1
# print(dick_k)
# print("共 %d 类"%reduce(lambda x,y:x+y,dick_k.values()))
##
# 展示地物
# gt=spectral.imshow(classes=target_mat.astype(int),figsize=(16,16))

# 不同的类使用不同的颜色
# ksc_color =np.array([[255,255,255],[184,40,99],[74,77,145],[35,102,193],[238,110,105],[117,249,76],
#     [114,251,253],[126,196,59],[234,65,247],[141,79,77],[183,40,99],[0,39,245],[90,196,111],])
#
# gt = spectral.imshow(classes = gt_mat.astype(int),figsize =(16，16),colors=ksc_color)
