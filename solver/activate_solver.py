import os
import functools

import nni
import torch
from solver.loss import SumRateMM, MM_Loss
from models.build import build_model
from solver.utils import train_one_epoch, evaluate, predict
from utils.checkpoint import EarlyStopping
from utils.file import write_record


class Solver:
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = torch.device(args.device)
        self.net = build_model(args)
        self.net = self.net.to(self.device)

        if args.mode == 'train' or args.mode == 'nni':
            # Setup optimizers for net to learn.
            self.optimizer = torch.optim.AdamW(self.net.parameters(),
                                              lr=args.lr,
                                              betas=(0.97, 0.999),
                                              eps=1e-08,
                                              weight_decay=args.weight_decay)
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor=args.factor,
                                                                        patience=args.patience,
                                                                        verbose=True)
            # 设置早停
        if args.mode == 'train' or args.mode == 'nni':
            self.early_stopping = EarlyStopping(patience=10, verbose=True, delta=1e-4,
                                                path=os.path.join(args.model_dir, "best_model.pth"),
                                                save=args.save_model, mode='min')
            self.use_tensorboard = args.use_tensorboard
            # 设置tensorboard
            if self.use_tensorboard:
                from utils.logger import Logger
                self.logger = Logger(args.log_dir)
            # 设置record
            self.record = functools.partial(write_record, file_path=args.record_file)

    def load_model(self):
        self.net.load_state_dict(
            torch.load(os.path.join(self.args.model_dir, "best_model.pth"), map_location=self.device))

    def load_model_from_path(self, path):
        self.net.load_state_dict(torch.load(path, map_location=self.device))

    def train(self, loaders):
        args = self.args
        net = self.net
        optimizer = self.optimizer
        scheduler = self.scheduler

        train_loader = loaders.train
        val_loader = loaders.val

        # Load or initialize the model parameters.
        if args.start_epoch > 0:
            self.load_model_from_path(self.args.pre_model_path)
            print("Loading model successfully")

        print('Start training...')
        # 根据模型获得损失函数
        if args.loss_type == "hybrid":
            loss = MM_Loss(args=args)
        else:
            loss = SumRateMM(args=args)

        for epoch in range(args.start_epoch + 1, args.end_epoch + 1):
            train_loader.dataset.epoch_now = epoch
            val_loader.dataset.epoch_now = epoch
            self.net.train()

            train_loss,train_sum,train_min,train_ee = train_one_epoch(net,
                                         optimizer,
                                         train_loader,
                                         epoch,
                                         args,
                                         loss)
            # 模型评估
            self.net.eval()
            val_loss,val_sum,val_min,val_ee = evaluate(net,
                                val_loader,
                                epoch, args,
                                loss)

            # 学习率下降
            scheduler.step(metrics=val_loss)
            # scheduler.step()
            # 判断早停
            self.early_stopping(val_loss, self.net)

            if self.use_tensorboard:
                self.logger.scalar_summary(tag="val_loss", value=val_loss, step=epoch)

            # if args.use_log:
            #     self.record(
            #         "TRAIN_INFO:epoch {},loss: {:.4f}".format(
            #             epoch, train_loss))
            #     self.record(
            #         "VAL_INFO:epoch {},loss: {:.4f}".format(
            #             epoch, val_loss))
            # 输出的太多了，所以分两次输出
            print("TRAIN_INFO:epoch {},loss: {:.4f},sum: {:.4f},min: {:.4f},ee: {:.4f}"
                  .format(epoch, train_loss,train_sum,train_min,train_ee))
            print("VAL_INFO:epoch {},loss: {:.4f},sum: {:.4f},min: {:.4f},ee: {:.4f}"
                  .format(epoch, val_loss,val_sum,val_min,val_ee))
            if self.early_stopping.early_stop:
                break

            if optimizer.param_groups[0]['lr'] < 5e-6:
                print("早停")
                break

    def test(self, loader):
        if self.args.model_path is None:
            self.load_model()
            self.net.eval()
            predict(loader, self.net, self.args)
        else:
            self.load_model_from_path(self.args.model_path)
            # 模型评估
            self.net.eval()
            predict(loader, self.net, self.args)

    def nni(self, loaders):
        pass
        # args = self.args
        # net = self.net
        # optimizer = self.optimizer
        # scheduler = self.scheduler
        #
        # train_loader = loaders.train
        # val_loader = loaders.val
        #
        # print('Start training...')
        # # 根据模型获得损失函数
        # loss = SumRateV2(args=args)
        # val_rates = []
        # for epoch in range(args.start_epoch + 1, args.end_epoch + 1):
        #     train_loader.dataset.epoch_now = epoch
        #     val_loader.dataset.epoch_now = epoch
        #     self.net.train()
        #
        #     train_loss, train_power_satisfy, train_min_rate = train_one_epoch(net,
        #                                                                       optimizer,
        #                                                                       train_loader,
        #                                                                       epoch,
        #                                                                       args,
        #                                                                       loss)
        #     # 模型评估
        #     self.net.eval()
        #     val_loss, val_rate, val_power_satisfy, val_min_rate = evaluate(net,
        #                                                                    val_loader,
        #                                                                    epoch, args,
        #                                                                    loss)
        #
        #     # 学习率下降
        #     scheduler.step(metrics=val_rate)
        #     # 保存中间的指标结果
        #     nni.report_intermediate_result(val_rate)
        #     val_rates.append(val_rate)
        #
        #     # 输出的太多了，所以分两次输出
        #     print(
        #         "TRAIN_INFO:epoch {},loss: {:.4f},power_sat: {:.4f},min_rate: {:.4f}".format(
        #             epoch, train_loss, train_power_satisfy, train_min_rate))
        #     print(
        #         "VAL_INFO:epoch {},loss: {:.4f},power_sat: {:.4f},percent: {:.4f},min_rate: {:.4f}".format(
        #             epoch, val_loss, val_power_satisfy, val_rate, val_min_rate))
        #
        #     # 判断早停
        #     self.early_stopping(val_rate, self.net)
        #
        #     # 早停策略，当学习率小于1e-6时候停止
        #     if optimizer.param_groups[0]['lr'] < 1e-6:
        #         print("早停")
        #         break
        #
        #     # if val_power_satisfy < 0.9:
        #     #     print("能量不满足早停")
        #     #     break
        #     # # 效果太差
        #     # if epoch == 1 and val_rate < 0.42:
        #     #     print("效果太差没超过42")
        #     #     break
        #
        #     if self.early_stopping.early_stop:
        #         break
        #
        # # report final result
        # nni.report_final_result(max(val_rates))
