import torch
import torch.nn as nn


class SumRateMM(nn.Module):
    def __init__(self, args):
        super(SumRateMM, self).__init__()
        self.K = args.user_num
        self.noise = args.noise
        self.p_max = args.p_max
        self.Nt = args.antenna_num

    def forward(self, RF, BB, P, H):
        """
        :param RF: [b,N_T,k] 网络输出后转置过
        :param BB: [b,k,k] 同样转置过 每个用户为一列
        :param P:  [b,k]
        :param H: [b*k,N_T]
        :return:
        """
        H = H.reshape(-1, self.K, self.Nt)

        # 计算所有样本的速率
        rate_user = self.get_bath_rate(RF, BB, P, H)

        # 获得总速率
        rate_user_sum = torch.sum(input=rate_user, dim=1)

        loss = - torch.mean(rate_user_sum)

        return loss

    def get_bath_rate(self, RF, BB, P, H):
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
        dr_temp1 = torch.einsum('bmi -> bi', I) - torch.einsum('bii -> bi', I) + self.noise

        R = torch.log2(1 + P * torch.einsum('bii -> bi', I) / dr_temp1)
        return R


class SumRateMMv2(nn.Module):
    def __init__(self, args):
        super(SumRateMM, self).__init__()
        self.noise = args.noise
        self.p_max = args.p_max
        self.Nt = args.antenna_num

    def forward(self, dicts, users):
        user_loss = []
        for user in users:
            RF, BB, P = dicts[str(user) + "_rf"], dicts[str(user) + "_bb"], dicts[str(user) + "_p"]
            H = dicts[str(user) + "_h"]
            H = H.reshape(-1, user, self.Nt)
            # 计算所有样本的速率
            rate_user = self.get_bath_rate(RF, BB, P, H)
            # 获得总速率
            rate_user_sum = torch.sum(input=rate_user, dim=1)
            loss = - torch.mean(rate_user_sum)
            user_loss.append(loss)

        return torch.mean(torch.tensor(user_loss))

    def get_bath_rate(self, RF, BB, P, H):
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
        dr_temp1 = torch.einsum('bmi -> bi', I) - torch.einsum('bii -> bi', I) + self.noise

        R = torch.log2(1 + P * torch.einsum('bii -> bi', I) / dr_temp1)
        return R
