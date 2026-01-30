import numpy as np
import os

"""
Created by: Rainer Trauth
Created on: 01.04.2020
"""


def load_data(path2inputs_trainingdata: str,
              filename_trainingdata: str) -> np.array:
    """Loads training data for neural network training.

    :param path2inputs_trainingdata:        path to inputs folder which contains training data
    :type path2inputs_trainingdata: str
    :param filename_trainingdata:           filename of .csv-file which contains training data to load
    :type filename_trainingdata: str
    :return:                                loaded training data
    :rtype: np.array
    """

    file_counting = 0       # 文件计数

    if os.path.exists(path2inputs_trainingdata):

        for file in os.listdir(path2inputs_trainingdata):

            if file.startswith('data_to_train'):
                file_counting += 1

    # 4. 初始化一个固定长度的列表，长度等于文件数量
    # 注意：data 不是一个巨大的 numpy 数组，而是一个列表，每个元素是一个 numpy 数组。
    # 这样设计是为了处理不同 Bag 文件时长（行数）不一样的情况。
    data = [None] * file_counting

    for i in range(0, file_counting):

        with open(os.path.join(path2inputs_trainingdata, filename_trainingdata) + '_' + str(i) + '.csv', 'r') as fh:
            data[i] = np.loadtxt(fh, delimiter=',')

    print('LOADING DATA DONE')

    return data
