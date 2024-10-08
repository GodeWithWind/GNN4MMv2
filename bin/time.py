import json
import os
import sys

import torch
from munch import Munch

sys.path.append("/home/code/MaxMin/SimMaxMinGAT")
from data.dataset import MyDatasetV1
from models.build import build_model

if __name__ == '__main__':
    # 设置实验名称
    expr = '8u_16n'
    iterations = 1000

    # 获得模型路径
    cfg_path = os.path.join("/home/code/MaxMin/SimMaxMinGAT/expr", expr, "train.json")

    # 加载实验配置文件
    with open(cfg_path, 'r') as f:
        cfg = json.load(f)
        cfg = Munch(cfg)
    model = build_model(cfg)
    model = model.to(cfg.device)
    # 加载模型 不用加载模型，模型的推理时间是不变的
    model.load_state_dict(
        torch.load(os.path.join("/home/code/MaxMin/SimMaxMinGAT",cfg.model_dir, "best_model.pth"), map_location=cfg.device))

    # 创建网络模型
    model.eval()
    # 构建数据集
    train_dataset = MyDatasetV1("/home/code/MaxMin/SimMaxMinGAT/dataset/8u_16n_40w/train", userNum=cfg.user_num)
    input = train_dataset[0]
    input = input.to(cfg.device)
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    # GPU预热
    for _ in range(1000):
        _ = model(input)

    # 测速
    times = torch.zeros(iterations)  # 存储每轮iteration的时间

    with torch.no_grad():
        for iter in range(iterations):
            starter.record()
            _ = model(input)
            ender.record()
            # 同步GPU时间
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)  # 计算时间
            times[iter] = curr_time

    mean_time = times.mean().item()

    print("Inference time: {:.6f}, FPS: {} ".format(mean_time, 1000 / mean_time))
