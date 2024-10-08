import os
import json
import torch
import sys
from munch import Munch
from torch_geometric.data import DataLoader
import numpy as np
sys.path.append("/home/dby/liyuhang/ScmuMm/MM")
from data.dataset import MyDataset
from models.build import build_model
from solver.loss import EE_loss_save




@torch.no_grad()
def predict(gen_val, model, args):
    loss_sum_gather = []
    loss_sum = 0.0
    loss_rate_gather = []
    loss_rate_sum = 0.0
    ee_gather_all = []
    ee_hat_gather_all = []
    for step, batch in enumerate(gen_val):
        or_h = batch.x.to(args.device)
        for i in range(batch.x.shape[0]):
            noise = 10e-2 * (torch.randn(batch.x.shape[1]) + 1j * torch.randn(batch.x.shape[1]))
            batch.x[i, :] += noise
        out = model(batch.to(args.device))

        batch_loss, batch_loss_rate, ee_gather, ee_hat_gather = EE_loss_save(out, batch.y.to(torch.complex64),
                                                                             or_h.to(torch.complex64), args.user_num)
        loss_sum_gather.append(batch_loss.item())
        loss_rate_gather.append(batch_loss_rate.item())
        loss_sum += batch_loss.item()
        loss_rate_sum += batch_loss_rate.item()
        ee_gather_all = ee_gather_all + ee_gather
        ee_hat_gather_all = ee_hat_gather_all + ee_hat_gather

    print("loss_sum:{},loss_rate:{}".format(np.mean(loss_sum_gather), np.mean(loss_rate_gather)))
    print("loss_sum:{},loss_rate:{}".format(loss_sum / (step + 1), loss_rate_sum / (step + 1)))
    print(1 - np.mean(loss_rate_gather))


if __name__ == '__main__':
    # 设置实验名称
    expr = 're_3u_8n_su_best'
    data_path = './dataset/data_3u_8n/test_3u_8n'

    # 获得模型路径
    cfg_path = os.path.join("expr", expr, "train.json")

    # 加载实验配置文件
    with open(cfg_path, 'r') as f:
        cfg = json.load(f)
        cfg = Munch(cfg)
    net = build_model(cfg)
    net = net.to(cfg.device)
    # 加载模型
    net.load_state_dict(
        torch.load(os.path.join(cfg.model_dir, "best_model.pth"), map_location=cfg.device))

    # 创建网络模型
    net.eval()
    # 构建数据集
    datasets = MyDataset(data_path, userNum=cfg.user_num)
    gen_val = DataLoader(datasets, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    predict(gen_val, net, cfg)
