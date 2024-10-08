import os

import numpy as np
import scipy.io as scio


def validateInput(file_path, user_num=8, antennas=16):
    K = user_num
    N_T = antennas

    # 加载原始数据
    raw_h = scio.loadmat(os.path.join(file_path, "1_100000.mat"))["h_gather_all"]
    raw_w = scio.loadmat(os.path.join(file_path, "1_100000.mat"))["w_gather_all"]
    number = scio.loadmat(os.path.join(file_path, "1_100000.mat"))["EE_gather_all"].shape[1]
    print(number)
    print(raw_w.shape)
    print(raw_h.shape)

    # # 进行一个拼接操作然后查看是否有重复
    temp = np.zeros([K * N_T, number], dtype=np.complex64)
    for i in range(number):
        temp_h = raw_h[:, i * K:(i + 1) * K]
        for j in range(temp_h.shape[1]):
            temp[j * N_T:(j + 1) * N_T, i] = temp_h[:, j]
    temp_unique = np.unique(temp, axis=1)
    if (temp.shape[1] != temp_unique.shape[1]):
        print("原始数据集真的有重复！！")
    print(temp.shape, temp_unique.shape)


if __name__ == '__main__':
    file_path = 'C:/Users/lyh/Desktop/DATASET/MuserCode/8u16n/clean'
    validateInput(file_path, user_num=8, antennas=16)
