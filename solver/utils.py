import torch
from tqdm import tqdm
from torch_geometric.utils import unbatch


def train_one_epoch(model, optimizer, data_loader, epoch, args, loss):
    train_loss = []
    if args.progress_bar:
        data_loader = tqdm(data_loader)
    for step, batch in enumerate(data_loader):
        batch = batch.to(args.device)
        # 先让模型进行推理
        dicts, users = model(batch)
        batch_loss = loss(dicts, users)
        train_loss.append(batch_loss.item())

        if args.progress_bar:
            data_loader.desc = "[train epoch {}]".format(epoch)
        # 反向求梯度
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

    return torch.mean(torch.tensor(train_loss))


# 模型评估函数
@torch.no_grad()
def evaluate(model, data_loader, epoch, args, loss):
    val_loss = []
    if args.progress_bar:
        data_loader = tqdm(data_loader)
    for step, batch in enumerate(data_loader):
        batch = batch.to(args.device)
        # 先让模型进行推理
        dicts, users = model(batch)
        batch_loss = loss(dicts, users)
        val_loss.append(batch_loss.item())
        val_loss.append(batch_loss.item())
        if args.progress_bar:
            data_loader.desc = "[val epoch {}]".format(epoch)
        # 返回验证集上的准确率和平均loss
    return torch.mean(torch.tensor(val_loss))


@torch.no_grad()
def predict(gen_val, model, args):
    sum_rate_hat = []
    for step, batch in enumerate(gen_val):
        batch = batch.to(args.device)
        RF, BB, P = model(batch)
        if args.model == "GAT":
            H = batch.x.to(torch.complex64).reshape(-1, args.user_num, args.antenna_num)
        elif args.model == "GATReal":
            H = torch.complex(batch.x[:, :args.antenna_num], batch.x[:, args.antenna_num:]).to(torch.complex64).reshape(
                -1, args.user_num, args.antenna_num)
        elif args.model == "HetGAT":
            H = batch.x_dict['ant'].to(torch.complex64).reshape(-1, args.user_num, args.antenna_num)
        else:
            H = batch.x_s.to(torch.complex64).reshape(-1, args.user_num, args.antenna_num)
            # 计算所有样本的速率
        rate_user = get_bath_rate(RF, BB, P, H)
        # 获得总速率
        rate_user_sum = torch.mean(torch.sum(input=rate_user, dim=1))
        sum_rate_hat.append(rate_user_sum)
    # print(sum_rate_hat)
    print("test_loss:{:.4f}".format(torch.mean(torch.tensor(sum_rate_hat))))
    # # 再存储一下结果
    # # 使用torch.cat将列表中的张量按照第一维度进行拼接
    # result_hat = torch.cat(rate_user_hat_list, dim=0)
    # result_real = torch.cat(rate_user_real_list, dim=0)
    # # 创建两个 DataFrame 对象
    # df1 = pd.DataFrame(result_hat.cpu().numpy())
    # df2 = pd.DataFrame(result_real.cpu().numpy())

    # # 创建一个 Excel writer 对象
    # writer = pd.ExcelWriter('output.xlsx')

    # # 将 DataFrame 对象写入 Excel 文件中的不同表格
    # df1.to_excel(writer, sheet_name='hat')
    # df2.to_excel(writer, sheet_name='real')

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
    # 按行求和
    dr_temp1 = torch.einsum('bmi -> bi', I) - torch.einsum('bii -> bi', I) + 0.1

    R = torch.log2(1 + P * torch.einsum('bii -> bi', I) / dr_temp1)
    return R


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
    # 按行求和
    dr_temp1 = torch.einsum('bmi -> bi', I) - torch.einsum('bii -> bi', I) + 0.1

    R = torch.log2(1 + P * torch.einsum('bii -> bi', I) / dr_temp1)
    return R


def unbatch_mm(RF, BB, P, H, batch):
    """
    :param RF: [b*k,N_T]
    :param BB: [b*k,2]
    :param P:  [b*k,1]
    :param H: [b*k,N_T]
    :return:
    """
    dicts = {}
    users = []
    # 因为都是节点级别所以都是对应的
    RFs = unbatch(src=RF, batch=batch)
    BBs = unbatch(src=BB, batch=batch)
    Ps = unbatch(src=P, batch=batch)
    Hs = unbatch(src=H, batch=batch)
    for i in range(len(RFs)):
        rf, bb, p, h = RFs[i], BBs[i], Ps[i], Hs[i]
        user_num = rf.shape[0]
        users.append(user_num)
        if str(user_num) + "_rf" in dicts:
            dicts[str(user_num) + "_rf"].append(rf)
            dicts[str(user_num) + "_bb"].append(bb)
            dicts[str(user_num) + "_p"].append(p)
            dicts[str(user_num) + "_h"].append(h)
        else:
            dicts[str(user_num) + "_rf"] = [rf]
            dicts[str(user_num) + "_bb"] = [bb]
            dicts[str(user_num) + "_p"] = [p]
            dicts[str(user_num) + "_h"] = [h]
    # 在这里进行更新
    # 遍历张量字典
    for key, value in dicts.items():
        dicts[key] = torch.stack(tensors=value, dim=0)
    # 这里再处理对应bb
    for user in users:
        BB = dicts[str(user) + "_bb"]
        BB1 = torch.complex(BB[:, 0], BB[:, 2]).reshape(-1, user, 1)
        BB2 = torch.complex(BB[:, 1], BB[:, 3]).reshape(-1, 1, user)
        dicts[str(user) + "_bb"] = torch.bmm(BB1, BB2)
        # 这里复原原始信道
        H = dicts[str(user) + "_h"]
        ant_num = H.shape[1] // 2
        dicts[str(user) + "_h"] = torch.complex(H[:, :ant_num], H[:, ant_num:]).to(torch.complex64)
    return dicts, users
