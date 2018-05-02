# -*- coding: utf-8 -*-
#!/usr/bin/env python

"""
处理原始数据集：对数据进行归一化处理并转换为输入张量的形式
"""

import pandas as pd
import numpy as np
import glob
from sklearn import preprocessing

FILE_NUMS = 33

# 读取所有.txt 文件的列
def read_files():
    """
    :param allframe: 所有文件夹里的数据帧
    :param frame: 某个文件夹里的数据帧
    :return: 所有的数据帧拼接而成的数据帧
    """
    allframes = pd.DataFrame()
    list_ = []
    for i in range(FILE_NUMS):
        path = r'./PeMS/station' + str(i+1)
        allfiles = glob.glob(path + "/*.txt")
        frame = pd.DataFrame()
        frame_ = []
        for file_ in allfiles:
            table = pd.read_table(file_, usecols=[0,1])
            frame_.append(table)
        frame = pd.concat(frame_)
        list_.append(frame)
    allframes = pd.concat(list_)
    return allframes

# 按时间序列对数据分组并标准化
def group_by_time():
    """
    :param key: 按时间分组
    :param zscore: z分数标准化
    :param grouped: 将数据按时间进行分组并标准化后的结果
    :param vehicles: 把数据转换为矩阵形式
    """
    frame = read_files()
    frame['5 Minutes'] = pd.to_datetime(frame['5 Minutes'], format='%m/%d/%Y %H:%M')
    values = frame.groupby('5 Minutes')['Flow (Veh/5 Minutes)'].apply(list)
    vehicles = []
    for i in range(len(values)):
        vehicles.append(values[i])
    #vehicles = np.asarray(vehicles)
    #vechicles = vehicles.reshape((196, 288*FILE_NUMS))
    #vechicles = np.array([np.reshape(x, (288, FILE_NUMS)) for x in vechicles])
    return vehicles

vehicles = group_by_time()
scaler = preprocessing.MinMaxScaler()
samples = scaler.fit_transform(vehicles)

"""
if __name__ == "__main__":
    save = group_by_time()
    print(save) 
"""
