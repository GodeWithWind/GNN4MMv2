import torch
import torch.nn as nn


class SumRateMM(nn.Module):
    def __init__(self, args):
        super(SumRateMM, self).__init__()
        self.K = args.user_num
        self.noise = args.noise
        self.p_max = args.p_max
        self.Nt = args.antenna_num
        self.loss_type = args.loss_type
        self.P_c = 0.1

    def forward(self, RF, BB, P, H):
        """
        :param RF: [b,N_T,k] 网络输出后转置过
        :param BB: [b,k,k] 同样转置过 每个用户为一列
        :param P:  [b,k]
        :param H: [b*k,N_T]
        :return:
        """

        H = H.reshape(-1, self.K, self.Nt)
        # print(RF.shape)
        # print(BB.shape)
        # print(P.shape)
        # print(H.shape)
        # print(BB)

        # 计算所有样本的速率
        rate_user = self.get_bath_rate(RF, BB, P, H)
        # 获得总速率
        rate_user_sum = torch.sum(input=rate_user, dim=1)
        # 获得总速率
        rate_user_min = torch.min(input=rate_user, dim=1).values
        # 获得总速率
        rate_user_max = torch.max(input=rate_user, dim=1).values
        # print(P)
        power = torch.sum(input=P, dim=1) + torch.tensor(self.P_c).to("cuda")
        ee = torch.div(rate_user_sum, power)

        if self.loss_type == "sum":
            loss = - torch.mean(rate_user_sum)
        elif self.loss_type == "min":
            loss = - torch.mean(rate_user_sum) + torch.mean(rate_user_max - rate_user_min)
        elif self.loss_type == "ee":
            loss = - torch.mean(ee)
        return loss, torch.mean(rate_user_sum), torch.mean(rate_user_min), torch.mean(ee)

    def get_bath_rate(self, RF, BB, P, H):
        """

        :param RF: [b,N_T,k]
        :param BB: [b,k,k]
        :param P:  [b,k,1]
        :param H: [b*k,N_T]
        :return:
        """
        # 先计算出w
        # 这里是有问题的
        W = torch.bmm(RF, BB)
        W = torch.transpose(W, 1, 2)
        I = torch.real(torch.einsum('bim,bjm,bjn,bin -> bij', W.conj(), H, H.conj(), W))
        I = P.unsqueeze(-1) * I
        # 按行求和
        dr_temp1 = torch.einsum('bmi -> bi', I) - torch.einsum('bii -> bi', I) + self.noise
        R = torch.log2(1 + torch.einsum('bii -> bi', I) / dr_temp1)
        return R


class MM_Loss(nn.Module):
    def __init__(self, args):
        super(MM_Loss, self).__init__()
        self.K = args.user_num
        self.noise = args.noise
        self.p_max = args.p_max
        self.Nt = args.antenna_num
        self.loss_type = args.loss_type
        self.P_c = 0.1
        self.MT = MultiLossLayer(num_loss=3)

    def forward(self, RF_sum, BB_sum, P_sum, RF_ee, BB_ee, P_ee, RF_min, BB_min, P_min, H):
        """

        :param RF: [b,N_T,k] 网络输出后转置过
        :param BB: [b,k,k] 同样转置过 每个用户为一列
        :param P:  [b,k]
        :param H: [b*k,N_T]
        :return:
        """
        H = H.reshape(-1, self.K, self.Nt)

        # 计算sumloss
        rate_user_sum = self.get_bath_rate(RF_sum, BB_sum, P_sum, H)
        sum_loss = - torch.mean(torch.sum(input=rate_user_sum, dim=1))

        # 计算minloss
        rate_user_min = self.get_bath_rate(RF_min, BB_min, P_min, H)
        min_loss = -torch.mean(torch.min(input=rate_user_min, dim=1).values)
        # print(min_loss)

        min_lossv2 = -torch.mean(torch.sum(input=rate_user_min, dim=1)) + torch.mean(
            torch.max(input=rate_user_min, dim=1).values - torch.min(input=rate_user_min, dim=1).values)

        # 计算ee loss
        rate_user_ee = self.get_bath_rate(RF_ee, BB_ee, P_ee, H)
        rate_sum = torch.sum(input=rate_user_ee, dim=1)
        power = torch.sum(input=P_ee, dim=1) + torch.tensor(self.P_c).to("cuda")
        ee_loss = - torch.mean(torch.div(rate_sum, power))
        # print(ee_loss)

        loss = self.MT.get_loss(torch.cat((sum_loss.unsqueeze(-1), min_lossv2.unsqueeze(-1), ee_loss.unsqueeze(-1))))
        # loss  = sum_loss + 100* min_loss + ee_loss

        return loss, sum_loss, min_loss, ee_loss

    def get_bath_rate(self, RF, BB, P, H):
        """

        :param RF: [b,N_T,k]
        :param BB: [b,k,k]
        :param P:  [b,k,1]
        :param H: [b*k,N_T]
        :return:
        """
        # 先计算出w
        # 这里是有问题的
        W = torch.bmm(RF, BB)
        W = torch.transpose(W, 1, 2)
        I = torch.real(torch.einsum('bim,bjm,bjn,bin -> bij', W.conj(), H, H.conj(), W))
        I = P.unsqueeze(-1) * I
        # 按行求和
        dr_temp1 = torch.einsum('bmi -> bi', I) - torch.einsum('bii -> bi', I) + self.noise
        R = torch.log2(1 + torch.einsum('bii -> bi', I) / dr_temp1)
        return R


class MultiLossLayer(nn.Module):
    """
        计算自适应损失权重
        implementation of "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics"
    """

    def __init__(self, num_loss):
        """
        Args:
            num_loss (int): number of multi-task loss
        """
        super(MultiLossLayer, self).__init__()
        # sigmas^2 (num_loss,)
        # uniform init
        # 从均匀分布U(a, b)中生成值，填充输入的张量或变量，其中a为均匀分布中的下界，b为均匀分布中的上界
        # self.sigmas_sq = nn.Parameter(torch.ones(num_loss), requires_grad=True).to('cuda')
        self.loss_scale = nn.Parameter(torch.tensor([-0.5] * num_loss, requires_grad=True, device="cuda"))

    def get_loss(self, loss_set):
        """
        Args:
            loss_set (Tensor): multi-task loss (num_loss,)
        """
        # 1/2σ^2
        # (num_loss,)
        # self.sigmas_sq -> tensor([0.9004, 0.4505]) -> tensor([0.6517, 0.8004]) -> tensor([0.7673, 0.6247])
        # # 出现左右两个数随着迭代次数的增加，相对大小交替变换
        # factor = torch.div(1.0, torch.mul(2.0, self.sigmas_sq))
        # # loss part (num_loss,)
        # loss_part = torch.sum(torch.mul(factor, loss_set))
        # # regular part 正则项，防止某个σ过大而引起训练严重失衡。
        # regular_part = torch.sum(torch.log(self.sigmas_sq))
        # loss = loss_part + regular_part
        loss = (loss_set / (2 * self.loss_scale.exp()) + self.loss_scale / 2).sum()
        return loss