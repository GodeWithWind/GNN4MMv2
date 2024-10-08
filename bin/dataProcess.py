import scipy.io as scio
import os
import numpy as np
import torch
import random


def process_raw_data(base_path, save_path='./', ):
    """

    :param base_path: 原始路径
    :param save_path: 保存的路径
    :return:
    """
    # 加载原始数据
    data_all = np.load(os.path.join(base_path, "datasetV3.npy"), allow_pickle=True)

    train_number = 40000
    val_number = 5000

    # 进行数据集的划分
    set_train = data_all[:train_number]
    set_val = data_all[train_number:train_number + val_number]
    set_tes = data_all[train_number + val_number:]

    print("训练集{}，验证集{},测试集{}".format(len(set_train), len(set_val), len(set_tes)))

    # 存储新数据
    np.save(os.path.join(save_path, "train/raw", "data.npy"), set_train)
    np.save(os.path.join(save_path, "val/raw", "data.npy"), set_val)
    np.save(os.path.join(save_path, "test_8u_16n/raw", "data.npy"), set_tes)


def process_mat_data(base_path, save_path='./', ):
    """

    :param base_path: 原始路径
    :param save_path: 保存的路径
    :return:
    """
    # 加载原始数据
    dataset = scio.loadmat(os.path.join(base_path, "BS5.mat"))

    train_number = 80000
    val_number = 10000

    # 进行数据集的划分
    train_H = dataset["H_gather"][:train_number]
    val_H = dataset["H_gather"][train_number:train_number + val_number]
    test_H = dataset["H_gather"][train_number + val_number:]

    # train_RF = dataset["RF_gather"][:train_number]
    # val_RF = dataset["RF_gather"][train_number:train_number + val_number]
    # test_RF = dataset["RF_gather"][train_number + val_number:]

    # train_BB = dataset["BB_gather"][:train_number]
    # val_BB = dataset["BB_gather"][train_number:train_number + val_number]
    # test_BB = dataset["BB_gather"][train_number + val_number:]

    # train_SR = dataset["SR_gather"][:train_number]
    # val_SR = dataset["SR_gather"][train_number:train_number + val_number]
    # test_SR = dataset["SR_gather"][train_number + val_number:]

    print("训练集{}，验证集{},测试集{}".format(len(train_H), len(val_H), len(test_H)))

    # 存储新数据
    # scio.savemat(os.path.join(os.getcwd(), save_path, "train/raw", "data.mat"),
    #              {'H': train_H, 'RF': train_RF, 'BB': train_BB, 'SR': train_SR})
    # scio.savemat(os.path.join(os.getcwd(), save_path, "val/raw", "data.mat"),
    #              {'H': val_H, 'RF': val_RF, 'BB': val_BB, 'SR': val_SR})
    # scio.savemat(os.path.join(os.getcwd(), save_path, "test_8u_16n/raw", "data.mat"),
    #              {'H': test_H, 'RF': test_RF, 'BB': test_BB, 'SR': test_SR})
    
    scio.savemat(os.path.join(os.getcwd(), save_path, "train_bs5/raw", "data.mat"),
                {'H': train_H})
    scio.savemat(os.path.join(os.getcwd(), save_path, "val_bs5/raw", "data.mat"),
                {'H': val_H})
    scio.savemat(os.path.join(os.getcwd(), save_path, "test_bs5/raw", "data.mat"),
                {'H': test_H})


def process_mat_data_test(base_path, save_path='./', ):
    """

    :param base_path: 原始路径
    :param save_path: 保存的路径
    :return:
    """
    # 加载原始数据
    dataset = scio.loadmat(os.path.join(base_path, "8u_16n_3p.mat"))

    # 进行数据集的划分
    test_H = dataset["H_gather"]
    test_RF = dataset["RF_gather"]
    test_BB = dataset["BB_gather"]
    test_SR = dataset["SR_gather"]
    print(test_H.shape)
    print("测试集{}".format( len(test_H)))
    # 存储新数据
    scio.savemat(os.path.join(os.getcwd(), save_path, "test_8u_16n_3p/raw", "data.mat"),
                 {'H': test_H, 'RF': test_RF, 'BB': test_BB, 'SR': test_SR})


def getChannel(base_path, save_path='./', ):
    """
    :param base_path: 原始路径
    :param save_path: 保存的路径
    :return:
    """

    # 加载原始数据
    dataset = np.load(os.path.join(base_path, "8u_16n.npy"), allow_pickle=True)

    H_gather = []

    for i in range(len(dataset)):
        # 获得单个样本
        data = dataset[i].item()
        # 获得信道
        channel = np.einsum('knp -> kn', data["user"]["channel"].squeeze())
        H_gather.append(channel)
    # 这里要进行打算一下
    random.shuffle(H_gather)
    H_gather = np.stack(H_gather)
    print(H_gather.shape)
    # 这里存储为
    scio.savemat(os.path.join(save_path, "H.mat"), {"H_gather": H_gather})


def PZF(base_path):
    """
    :param base_path: 原始路径
    :param save_path: 保存的路径
    :return:
    """
    dataset = scio.loadmat(os.path.join(base_path, "8u_16n.mat"))
    print(type(dataset))
    print(dataset.keys())
    for index in range(dataset["H_gather"].shape[0]):
        H = torch.from_numpy(dataset["H_gather"][index]).to(torch.complex64)
        RF = torch.from_numpy(dataset["RF_gather"][index]).to(torch.complex64)
        BB = torch.from_numpy(dataset["BB_gather"][index]).to(torch.complex64)
        W = torch.mm(RF, BB)
        # 按照列进行归一化
        W = W / torch.norm(W, p=2, dim=0, keepdim=True)
        W = W.T

        I = torch.real(torch.einsum('im,jm,jn,in -> ij', W.conj(), H, H.conj(), W))
        P = torch.ones([8, 1]) / 8
        I = P * I

        dr_temp1 = torch.einsum('mi -> i', I) - torch.einsum('ii -> i', I) + 1.99e-12
        R = torch.log2(1 + torch.einsum('ii -> i', I) / dr_temp1)
        if abs(torch.sum(R) - dataset['SR_gather'][index]) > 0.1:
            print("误差过大")
        print(index)


if __name__ == '__main__':
    save_path = "../dataset/BS_8u_16n_real/"
    base_path = '../dataset/raw_data/'
    # getChannel(base_path, save_path)
    # PZF(base_path)
    process_mat_data(base_path, save_path)
