import argparse
import json
import os
import random
import shutil
import sys
import platform

import numpy as np
from torch.backends import cudnn

import torch
from munch import Munch

from utils.file import save_json, prepare_dirs
from utils.misc import get_datetime, str2bool


def load_cfg():
    """
    加载训练参数（一般用shell文件）
    :return: 配置文件
    """
    # There are two ways to load config, use a json file or command line arguments.
    if len(sys.argv) >= 2 and sys.argv[1].endswith('.json'):
        with open(sys.argv[1], 'r') as f:
            cfg = json.load(f)
            cfg = Munch(cfg)
            if len(sys.argv) >= 3:
                cfg.exp_id = sys.argv[2]
            else:
                print("Warning: using existing experiment dir.")
            if not cfg.about:
                cfg.about = f"Copied from: {sys.argv[1]}"
    else:
        cfg = parse_args()
        cfg = Munch(cfg.__dict__)
        # 判断线程数
        if (platform.system() == 'Windows'):
            nw = 0
        else:
            nw = min([os.cpu_count(), cfg.batch_size if cfg.batch_size > 1 else 0, 40])  # number of workers
        cfg.num_workers = nw
    return cfg


def parse_args():
    """
    转化参数
    :return:
    """
    parser = argparse.ArgumentParser()

    # About this experiment.
    parser.add_argument('--about', type=str, default="")
    parser.add_argument('--exp_id', type=str, help='Folder name and id for this experiment.')
    parser.add_argument('--exp_dir', type=str, default='expr')

    # Meta arguments.
    parser.add_argument('--mode', type=str, default='nni', choices=['train', 'test', 'nni'])

    parser.add_argument('--model', type=str, default='GAT',
                        choices=['GATReal','CNN', 'HetGATV2','MLP','GCN','GAT','GATV3'])
    
    parser.add_argument('--loss_type', type=str, default='sum',
                    choices=['sum', 'ee', 'min','hybrid'])
    # 边特征维度
    parser.add_argument('--edge_dim', type=int, default=0)

    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    # Model related arguments.
    parser.add_argument('--user_num', type=int, default=8)
    parser.add_argument('--antenna_num', type=int, default=16)

    # 整个小区的最大能量
    parser.add_argument('--p_max', type=int, default=1)

    # 白噪声
    parser.add_argument('--noise', type=float, default=1e-1)

    # Dataset related arguments.
    parser.add_argument('--train_path', type=str, required=False, default="./dataset/data_8u_16n_v2/train")
    parser.add_argument('--val_path', type=str, required=False, default="./dataset/data_8u_16n_v3/val")
    parser.add_argument('--test_path', type=str, required=False)

    # DataLoader related arguments
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=60)

    # train related arguments
    parser.add_argument('--start_epoch', type=int, default=0, help='begin epoch')
    parser.add_argument('--end_epoch', default=200, type=int, help='train_epochs')
    parser.add_argument('--save_model', type=str2bool, default=False, help='save the model during training')
    parser.add_argument('--progress_bar', type=str2bool, default=False, help='show progress during training')

    # test related arguments
    parser.add_argument('--pre_model_path', type=str, required=False)
    parser.add_argument('--model_path', type=str, required=False)
    parser.add_argument('--save_result', type=str2bool, default=False, required=False)

    # Optimizing related arguments.
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate for generator.")
    parser.add_argument('--weight_decay', type=float, default=0, help="Learning rate for generator.")
    parser.add_argument('--factor', type=float, default=5e-1, help="Learning rate decay factor")
    parser.add_argument('--patience', type=int, default=1)

    # Log related arguments.
    parser.add_argument('--use_tensorboard', type=str2bool, default=True)
    parser.add_argument('--use_log', type=str2bool, default=True)

    # Others
    parser.add_argument('--seed', type=int, default=0, help='Seed for random number generator.')
    parser.add_argument('--cudnn_benchmark', type=str2bool, default=False)
    parser.add_argument('--cudnn_deterministic', type=str2bool, default=True)

    return parser.parse_args()


def setup_cfg(args):
    cudnn.benchmark = args.cudnn_benchmark
    cudnn.deterministic = args.cudnn_deterministic
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    # 复数暂时不支持多GPU
    # if args.mode == 'train' and torch.cuda.device_count() > 1:
    #     print(f"We will train on {torch.cuda.device_count()} GPUs.")
    #     args.multi_gpu = True
    # else:
    #     args.multi_gpu = False
    if args.mode == 'train':
        if args.exp_id is None:
            args.exp_id = get_datetime()
            # Tip: you can construct the exp_id automatically here by use the args.
    # else:
    #     if args.exp_id is None:
    #         args.exp_id = input("Please input exp_id: ")
    #     if not os.path.exists(os.path.join(args.exp_dir, args.exp_id)):
    #         all_existed_ids = os.listdir(args.exp_dir)
    #         for existed_id in all_existed_ids:
    #             if existed_id.startswith(args.exp_id + "-"):
    #                 args.exp_id = existed_id
    #                 print(f"Warning: exp_id is reset to {existed_id}.")
    #                 break
    # 设置线程数
    if os.name == 'nt' and args.num_workers != 0:
        print("Warning: reset num_workers = 0, because running on a Windows system.")
        args.num_workers = 0

    args.log_dir = os.path.join(args.exp_dir, args.exp_id, "logs")
    args.model_dir = os.path.join(args.exp_dir, args.exp_id, "models")
    args.eval_dir = os.path.join(args.exp_dir, args.exp_id, "eval")
    prepare_dirs([args.log_dir, args.model_dir, args.eval_dir])
    args.record_file = os.path.join(args.exp_dir, args.exp_id, "records.txt")

    if args.mode == 'train':
        # 复制配置文件
        if os.path.exists(f'./scripts/{args.exp_id}.sh'):
            shutil.copyfile(f'./scripts/{args.exp_id}.sh',
                            os.path.join(args.exp_dir, args.exp_id, f'{args.exp_id}.sh'))
    # 远程启动 tensorboard
    # if args.mode == 'train' and args.start_tensorboard:
    #     start_tensorboard(os.path.join(args.exp_dir, args.exp_id), 'logs')


def save_cfg(cfg):
    exp_path = os.path.join(cfg.exp_dir, cfg.exp_id)
    os.makedirs(exp_path, exist_ok=True)
    filename = cfg.mode
    # if cfg.mode == 'train' and cfg.start_iter != 0:
    #     filename = f"resume_{cfg.start_iter}"
    save_json(exp_path, cfg, filename)
    # 存储网络架构
    shutil.copyfile('models/gnn.py', os.path.join(cfg.exp_dir, cfg.exp_id, 'model.py'))


def print_cfg(cfg):
    print(json.dumps(cfg, indent=4))
