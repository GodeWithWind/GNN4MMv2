import torch
from tqdm import tqdm
from solver.misc import get_beamforming
import numpy as np
import pandas as pd
import scipy
import os

def train_one_epoch(model, optimizer, data_loader, epoch, args, loss):
    train_loss = []
    sum_rate = []
    min_rate = [] 
    ee = []
    power_sta = []
    if args.progress_bar:
        data_loader = tqdm(data_loader)
    for step, batch in enumerate(data_loader):
        batch = batch.to(args.device)
        if args.model == "GATReal" or args.model == "GCN" or args.model == "GAT" or args.model == "GATV3":
            RF, BB, P = model(batch)
            H = torch.complex(batch.x[:,:args.antenna_num], batch.x[:,args.antenna_num:]).to(torch.complex64)
            batch_loss, batch_sum, batch_min, batch_ee, bath_sta = loss(RF, BB, P, H)
        elif args.model == "HetGATV2":
            RF, BB, P = model(batch)
            # H = batch.x_dict['user'].to(torch.complex64)
            H = torch.complex(batch.x_dict['user'][:,:args.antenna_num], batch.x_dict['user'][:,args.antenna_num:2*args.antenna_num]).to(torch.complex64)
            batch_loss, batch_sum, batch_min, batch_ee, bath_sta = loss(RF, BB, P, H)
        elif args.model == "CNN":
            RF, BB, P = model(batch)
            # 这里获得CNN的信道
            H = torch.complex(batch[:,1,:,:], batch[:,2,:,:]).to(torch.complex64)
            batch_loss, batch_sum, batch_min, batch_ee, bath_sta = loss(RF, BB, P, H)
        elif args.model == "MLP":
            RF, BB, P = model(batch)
            # 这里获得CNN的信道
            H = torch.complex(batch[:,:,:args.antenna_num], batch[:,:,args.antenna_num:]).to(torch.complex64)
            batch_loss, batch_sum, batch_min, batch_ee, bath_sta = loss(RF, BB, P, H)
        # print(batch_loss)
        train_loss.append(batch_loss.item())
        sum_rate.append(batch_sum.item())
        min_rate.append(batch_min.item())
        ee.append(batch_ee.item())
        power_sta.append(bath_sta.item())
        if args.progress_bar:
            data_loader.desc = "[train epoch {}]".format(epoch)
        # 反向求梯度
        optimizer.zero_grad()
        batch_loss.backward()
        
        # 在更新权重之前，对梯度进行裁剪，使其不超过0.5
        torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)
        optimizer.step()

    return torch.mean(torch.tensor(train_loss)), torch.mean(torch.tensor(sum_rate)),torch.mean(torch.tensor(min_rate)),torch.mean(torch.tensor(ee)),torch.mean(torch.tensor(power_sta))


# 模型评估函数
@torch.no_grad()
def evaluate(model, data_loader, epoch, args, loss):
    val_loss = []
    sum_rate = []
    target_sum_rate = []
    power_sta = []
    min_rate = [] 
    ee = []
    if args.progress_bar:
        data_loader = tqdm(data_loader)
    for step, batch in enumerate(data_loader):
        batch = batch.to(args.device)
        # 先让模型进行推理
        if args.model == "GATReal" or args.model == "GCN" or args.model == "GAT" or args.model == "GATV3":
            H = torch.complex(batch.x[:,:args.antenna_num], batch.x[:,args.antenna_num:]).to(torch.complex64)
            # 这里添加噪声
            RF, BB, P = model(batch)
            # H = torch.complex(batch.x[:,:args.antenna_num], batch.x[:,args.antenna_num:]).to(torch.complex64)
            batch_loss, batch_sum, batch_min, batch_ee,batch_sta = loss(RF, BB, P, H)
        elif args.model == "HetGATV2":
            RF, BB, P = model(batch)
            # H = batch.x_dict['user'].to(torch.complex64)
            H = torch.complex(batch.x_dict['user'][:,:args.antenna_num], batch.x_dict['user'][:,args.antenna_num:2*args.antenna_num]).to(torch.complex64)
            batch_loss, batch_sum, batch_min, batch_ee,batch_sta = loss(RF, BB, P, H)
        elif args.model == "CNN":
            RF, BB, P = model(batch)
            # 这里获得CNN的信道
            H = torch.complex(batch[:,1,:,:], batch[:,2,:,:]).to(torch.complex64)
            batch_loss, batch_sum, batch_min, batch_ee, batch_sta = loss(RF, BB, P, H)
        elif args.model == "MLP":
            RF, BB, P = model(batch)
            # 这里获得CNN的信道
            H = torch.complex(batch[:,:,:args.antenna_num], batch[:,:,args.antenna_num:]).to(torch.complex64)
            batch_loss, batch_sum, batch_min, batch_ee, batch_sta = loss(RF, BB, P, H)

        val_loss.append(batch_loss.item())
        sum_rate.append(batch_sum.item())
        min_rate.append(batch_min.item())
        # target_sum_rate.append(torch.mean(batch.y).item())
        power_sta.append(batch_sta.item())
        # true_sum_rate.append(torch.mean(rate_user_sum).item())
        ee.append(batch_ee.item())
        if args.progress_bar:
            data_loader.desc = "[val epoch {}]".format(epoch)
        # 返回验证集上的准确率和平均loss
    return torch.mean(torch.tensor(val_loss)) , torch.mean(torch.tensor(sum_rate)),torch.mean(torch.tensor(min_rate)),torch.mean(torch.tensor(ee)),torch.mean(torch.tensor(target_sum_rate)),torch.mean(torch.tensor(power_sta))


def get_bath_rateV2(RF, BB, P, H,args):
    """

    :param RF: [b,N_T,k]
    :param BB: [b,k,k]
    :param P:  [b,k,1]
    :param H: [b*k,N_T]
    :return:
    """
    # 先计算出w
    # 这里是有问题的
    # W = torch.bmm(RF, BB)
    # # 对每个用户的beam 进行归一化即 列归一化
    # W = W / torch.norm(W, p=2, dim=1, keepdim=True)
    # # W = torch.transpose(W, 1, 2)

    # BB_tem = torch.sqrt(P) * BB
    # W_tem = torch.bmm(RF, BB_tem)
    # W_tem_norm = torch.norm(W_tem, p='fro', dim=[1, 2], keepdim=True) ** 2

    # scaling_factor = torch.sqrt(args.p_max / W_tem_norm)
    # BB = BB * scaling_factor

    # BB_tem = torch.sqrt(P) * BB
    # W_tem = torch.bmm(RF, BB_tem)
    # W_tem_norm = torch.norm(W_tem, p='fro', dim=[1, 2], keepdim=True) ** 2
    # scaling_factor = torch.sqrt(args.p_max / W_tem_norm)
    # BB = BB * scaling_factor

    # W = torch.bmm(RF, BB)


    # W_norm = torch.norm(W, p='fro', dim=[1, 2], keepdim=True)

    # W = torch.sqrt(torch.tensor(args.p_max).to('cuda')) * torch.div(W, torch.max(W_norm, torch.ones(
    # W_norm.size()).to('cuda')))
    # nt,k ,所以这里进行列归一化
    W = torch.bmm(RF, BB)
    # print(RF.shape)
    # print(BB.shape)
    # print(P.shape)
    # print(W.shape)
    # print(torch.norm(W, p=2, dim=1, keepdim=True).shape)
    norm = torch.norm(W, p=2, dim=1, keepdim=True)
    W = W / norm
    
    # 计算所有样本的power
    NBB = BB / norm
    NBB = torch.sqrt(torch.transpose(P, 1, 2)) * NBB
    NW = torch.bmm(RF, NBB)
    power = torch.norm(NW, p='fro', dim=[1, 2]) ** 2

    # 计算rate 
    W = torch.transpose(W, 1, 2)
    I = torch.real(torch.einsum('bim,bjm,bjn,bin -> bij', W.conj(), H, H.conj(), W))
    I = P * I
    # 按行求和
    dr_temp1 = torch.einsum('bmi -> bi', I) - torch.einsum('bii -> bi', I)  + 1
    R = torch.log2(1 + torch.einsum('bii -> bi', I) / dr_temp1)
    return R , power


@torch.no_grad()
def predict(gen_val, model, args):
    sum_rate_hat = []
    min_rate_hat = [] 
    ee_hat = []
    power_sta = []
    target_sum_rate = []
    rate_user_hat_list = []
    # P_hat_list = []
    for step, batch in enumerate(gen_val):
        batch = batch.to(args.device)
        if args.model == "GATReal" or args.model == "GCN" or args.model == "GAT"or args.model == "GATV3":
            H = torch.complex(batch.x[:,:args.antenna_num], batch.x[:,args.antenna_num:])
            # snr_linear = 10 ** (-15 / 10)
            # signal_power = torch.norm(H, p=2, dim=1, keepdim=True) ** 2
            # noise_power = signal_power * snr_linear
            # noise = torch.randn(size=H.shape, dtype=torch.complex64).to('cuda')
            # noise = noise / torch.norm(noise, p=2, dim=1, keepdim=True)
            # noise = noise * torch.sqrt(noise_power)
            # Hn = H + noise.to('cuda')
            # batch.x = torch.cat((Hn.real,Hn.imag),dim =1)
            RF, BB, P = model(batch)
            H = H.reshape(-1,args.user_num,args.antenna_num)

            #H = torch.complex(batch.x[:,:args.antenna_num], batch.x[:,args.antenna_num:]).to(torch.complex64).reshape(-1,args.user_num,args.antenna_num)
        elif args.model == "HetGATV2":
            H = batch.x_dict['user'].to(torch.complex64)
            # snr_linear = 10 ** (-20 / 10)
            # signal_power = torch.norm(H, p=2, dim=1, keepdim=True) ** 2
            # noise_power = signal_power * snr_linear
            # noise = torch.randn(size=H.shape, dtype=torch.complex64).to('cuda')
            # noise = noise / torch.norm(noise, p=2, dim=1, keepdim=True)
            # noise = noise * torch.sqrt(noise_power)
            # Hn = H + noise.to('cuda')
            # batch.x_dict['user'] = Hn

            # ant_x = batch.x_dict['ant'].to(torch.complex64)
            # signal_power2 = torch.norm(ant_x, p=2, dim=1, keepdim=True) ** 2
            # noise_power2 = signal_power2 * snr_linear
            # noise2 = torch.randn(size=ant_x.shape, dtype=torch.complex64).to('cuda')
            # noise2 = noise2 / torch.norm(noise2, p=2, dim=1, keepdim=True)
            # noise2 = noise2 * torch.sqrt(noise_power2)
            # ant_x = ant_x + noise2.to('cuda')
            # batch.x_dict['ant'] = ant_x
            # H = torch.complex(batch.x_dict['user'][:,:args.antenna_num], batch.x_dict['user'][:,args.antenna_num:2*args.antenna_num]).to(torch.complex64)
            RF, BB, P = model(batch)
            H = H.reshape(-1,args.user_num,args.antenna_num)
        elif args.model == "CNN":
            RF, BB, P = model(batch)
            # 这里获得CNN的信道
            H = torch.complex(batch[:,1,:,:], batch[:,2,:,:]).to(torch.complex64).reshape(-1,args.user_num,args.antenna_num)
        elif args.model == "MLP":
            RF, BB, P = model(batch)
            # 这里获得CNN的信道
            H = torch.complex(batch[:,:,:args.antenna_num], batch[:,:,args.antenna_num:]).to(torch.complex64)
            # batch_loss, batch_sum, batch_min, batch_ee, batch_sta = loss(RF, BB, P, H)
             
        if args.loss_type == "hybrid":
            rate_user_sum = get_bath_rate(RF_sum, BB_sum, P_sum, H)
            sum_rate = torch.mean(torch.sum(input=rate_user_sum, dim=1))
            # 计算minloss
            rate_user_min = get_bath_rate(RF_min, BB_min, P_min, H)
            min_rate =  torch.mean( torch.min(input=rate_user_min, dim=1).values)

            # 计算eeloss
            rate_user_ee = get_bath_rate(RF_ee, BB_ee, P_ee, H)
            rate_sum = torch.sum(input=rate_user_ee, dim=1)
            power =  torch.sum(input=P_ee, dim=1) + torch.tensor(0.1).to("cuda")
            ee = torch.mean(torch.div(rate_sum,power))

            sum_rate_hat.append(sum_rate)
            min_rate_hat.append(min_rate)
            ee_hat.append(ee)
        else:
            # 这里计算速率和每个样本的能量
            rate_user , power = get_bath_rateV2(RF, BB, P, H,args)
            # print(power)
            rate_user_sum = torch.sum(input=rate_user, dim=1)
            rate_user_min = torch.min(input=rate_user, dim=1).values
            # power =  torch.sum(input=P, dim=1) + torch.tensor(0.1).to("cuda")
            ee = torch.div(rate_user_sum,power)

            sum_rate_hat.append(torch.mean(rate_user_sum))
            min_rate_hat.append(torch.mean(rate_user_min))
            ee_hat.append(torch.mean(ee))
            # target_sum_rate.append(torch.mean(batch.y))

            # 计算满足率
            pow_cons = torch.le(power, args.p_max + 1e-6)
            power_satisfy = torch.sum(pow_cons) / pow_cons.shape[0]
            power_sta.append(power_satisfy)
            # 添加到list中
            rate_user_hat_list.append(rate_user_sum)
            # P_hat_list.append(P)
    # print(sum_rate_hat)
    print("sum:{:.4f},min: {:.4f},ee: {:.4f},power_sta: {:.4f},target_sum: {:.4f}"\
    .format(torch.mean(torch.tensor(sum_rate_hat)),torch.mean(torch.tensor(min_rate_hat)) \
    ,torch.mean(torch.tensor(ee_hat)),torch.mean(torch.tensor(power_sta)),torch.mean(torch.tensor(target_sum_rate))))
    # # 再存储一下结果
    # # 使用torch.cat将列表中的张量按照第一维度进行拼接
    result_hat = torch.cat(rate_user_hat_list, dim=0)
    scipy.io.savemat(os.path.join('./re/',args.model,"quantizeB4.mat"), {'SR_gather': result_hat.cpu().numpy()})
    # print(result_hat.shape)
    # P_hat = torch.cat(P_hat_list, dim=0)
    # # result_real = torch.cat(rate_user_real_list, dim=0)
    # # 创建两个 DataFrame 对象
    # df1 = pd.DataFrame(result_hat.cpu().numpy())
    # df2 = pd.DataFrame(P_hat.cpu().numpy())

    # # 创建一个 Excel writer 对象
    # writer = pd.ExcelWriter('output.xlsx')

    # # 将 DataFrame 对象写入 Excel 文件中的不同表格
    # df1.to_excel(writer, sheet_name='rate_hat')
    # df2.to_excel(writer, sheet_name='p_hat')

    # # 保存 Excel 文件
    # writer.save()


def get_bath_rate(RF, BB, P, H):
    """

    :param RF: [b,N_T,k]
    :param BB: [b,k,k]
    :param P:  [b,k,1]
    :param H: [b*k,N_T]
    :return:
    """
    # 先计算出w
    W = torch.bmm(RF, BB)
    W = torch.transpose(W, 1, 2)

    I = torch.real(torch.einsum('bim,bjm,bjn,bin -> bij', W.conj(), H, H.conj(), W))
    I = P.unsqueeze(-1) * I
    # 按行求和
    dr_temp1 = torch.einsum('bmi -> bi', I) - torch.einsum('bii -> bi', I) + 0.1

    R = torch.log2(1 + torch.einsum('bii -> bi', I) / dr_temp1)
    return R
