import scipy.io as scio
import os
import torch


def process_raw_data(base_path):
    train_number = 40000
    val_number = 10000
    test_number = 10000

    files = ["7u_16n.mat", "8u_16n.mat", "9u_16n.mat"]
    train_dict = {}
    val_dict = {}

    for file in files:
        # 加载原始数据
        raw_h = scio.loadmat(os.path.join(base_path, "raw/", file))["h_gather"]
        raw_h = torch.from_numpy(raw_h)
        user_num = raw_h.shape[2]
        # 进行数据集的划分
        h_set_train = raw_h[:train_number]
        h_set_val = raw_h[train_number:train_number + val_number]
        h_set_tes = raw_h[train_number + val_number:]
        train_dict[str(user_num) + "u_h"] = h_set_train
        val_dict[str(user_num) + "u_h"] = h_set_val
        # 训练集和测试集存储到字典中，测试集则直接保存
        test_dict = {str(user_num) + "u_h": h_set_tes}
        # 保存张量字典到文件
        torch.save(test_dict, os.path.join(base_path, "test_" + str(user_num) + "u/", "raw/", "data.pth"))
    # 最后保存一下训练集和验证集
    torch.save(train_dict, os.path.join(base_path, "train/", "raw/", "data.pth"))
    torch.save(val_dict, os.path.join(base_path, "val/", "raw/", "data.pth"))


if __name__ == '__main__':
    base_path = '../dataset/16n/'
    save_path = "../dataset/16n/"
    process_raw_data(base_path)
