# -*- coding: utf-8 -*-
"""
Created on Mon May 20 13:07:09 2019

@author: viryl
"""
from __future__ import print_function
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datetime import datetime
from tensorboardX import SummaryWriter
from utils import DataSet, validate, show_confMat  # 自定义类
from net import Net  # 导入模型

# 定义超参
EPOCH = 5
BATCH_SIZE = 24
classes_name = [str(c) for c in range(16)]

# --------------------加载数据---------------------
# Indian Pines .mat文件路径(每个文件都是一个单独的类)
path = os.path.join(os.getcwd(), "patch")
training_dataset = DataSet(path=path, train=True)
testing_dataset = DataSet(path=path, train=False)
# Data Loaders
train_loader = torch.utils.data.DataLoader(
    dataset=training_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    dataset=testing_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

# 检查cuda是否可用
use_cuda = torch.cuda.is_available()
# 生成log
now_time = datetime.now()
time_str = datetime.strftime(now_time, '%m-%d_%H-%M-%S')
log_path = os.path.join(os.getcwd(), "log")
log_dir = os.path.join(log_path, time_str)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
writer = SummaryWriter(log_dir)


# ---------------------搭建网络--------------------------
cnn = Net()  # 创建CNN
cnn.init_weights()  # 初始化权值
cnn = cnn.double()

# --------------------设置损失函数和优化器----------------------
optimizer = optim.Adam(cnn.parameters())  # lr:(default: 1e-3)优化器
criterion = nn.CrossEntropyLoss()  # 损失函数
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=EPOCH/2, gamma=0.5)  # 设置学习率下降策略

# --------------------训练------------------------------
if(use_cuda):  # 使用GPU
    cnn = cnn.cuda()
for epoch in range(EPOCH):
    loss_sigma = 0.0    # 记录一个epoch的loss之和
    correct = 0.0
    total = 0.0
    scheduler.step()  # 更新学习率

    for batch_idx, data in enumerate(train_loader):
        # 获取图片和标签
        inputs, labels = data
        if(use_cuda):
            inputs, labels = inputs.cuda(), labels.cuda()
        optimizer.zero_grad()  # 清空梯度
        cnn = cnn.train()
        outputs = cnn(inputs)
        loss = criterion(outputs, labels)
        loss.backward()  # 反向传播
        optimizer.step()  # 更新权值

        # 统计预测信息
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += ((predicted == labels).squeeze().sum()).item()
        loss_sigma += loss.item()

        # 每 BATCH_SIZE 个 iteration 打印一次训练信息，loss为 BATCH_SIZE 个 iteration 的平均
        if batch_idx % BATCH_SIZE == BATCH_SIZE-1:
            loss_avg = loss_sigma / BATCH_SIZE
            loss_sigma = 0.0
            print("Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                epoch + 1, EPOCH, batch_idx + 1, len(train_loader), loss_avg, correct / total))
            # 记录训练loss
            writer.add_scalars(
                'Loss_group', {'train_loss': loss_avg}, epoch)
            # 记录learning rate
            writer.add_scalar(
                'learning rate', scheduler.get_lr()[0], epoch)
            # 记录Accuracy
            writer.add_scalars('Accuracy_group', {
                               'train_acc': correct / total}, epoch)
    # 每个epoch，记录梯度，权值
    for name, layer in cnn.named_parameters():
        writer.add_histogram(
            name + '_grad', layer.grad.cpu().data.numpy(), epoch)
        writer.add_histogram(name + '_data', layer.cpu().data.numpy(), epoch)

    # ------------------------------------ 观察模型在验证集上的表现 ------------------------------------
    if epoch % 1 == 0:
        loss_sigma = 0.0
        cls_num = len(classes_name)
        conf_mat = np.zeros([cls_num, cls_num])  # 混淆矩阵
        cnn.eval()
        for batch_idx, data in enumerate(test_loader):
            images, labels = data
            if(use_cuda):
                images, labels = images.cuda(), labels.cuda()
            cnn = cnn.train()
            outputs = cnn(images)  # forward
            outputs.detach_()  # 不求梯度
            loss = criterion(outputs, labels)  # 计算loss
            loss_sigma += loss.item()

            _, predicted = torch.max(outputs.data, 1)  # 统计
            # labels = labels.data    # Variable --> tensor
            # 统计混淆矩阵
            for j in range(len(labels)):
                cate_i = labels[j]
                pre_i = predicted[j]
                conf_mat[cate_i, pre_i] += 1.0
        print('{} set Accuracy:{:.2%}'.format(
            'Valid', conf_mat.trace() / conf_mat.sum()))
        # 记录Loss, accuracy
        writer.add_scalars(
            'Loss_group', {'valid_loss': loss_sigma / len(test_loader)}, epoch)
        writer.add_scalars('Accuracy_group', {
                           'valid_acc': conf_mat.trace() / conf_mat.sum()}, epoch)
print('Finished Training')

# ----------------------- 保存模型 并且绘制混淆矩阵图 -------------------------
cnn_save_path = os.path.join(log_dir, 'net_params.pkl')
torch.save(cnn.state_dict(), cnn_save_path)

conf_mat_train, train_acc = validate(cnn, train_loader, 'train', classes_name)
conf_mat_valid, valid_acc = validate(cnn, test_loader, 'test', classes_name)

show_confMat(conf_mat_train, classes_name, 'train', log_dir)
show_confMat(conf_mat_valid, classes_name, 'valid', log_dir)
