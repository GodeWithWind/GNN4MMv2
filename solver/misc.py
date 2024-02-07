import torch


def get_EEv2(h, w):
    """
    计算ee
    :param h:信道值 shape[antennas,user_num]
    :param w: beam矩阵
    :return:ee
    """
    P_c = 0.1
    noise = 0.01
    I = torch.real(torch.multiply(torch.matmul(w.T.conj(), h), torch.matmul(w.T.conj(), h).conj()))
    P = torch.real(torch.diag(torch.matmul(w.T.conj(), w)))
    dr_temp = torch.sum(I, dim=0) - torch.diag(I) + noise
    R = torch.log2(1 + torch.div(torch.diag(I), dr_temp))
    ee = (torch.sum(R)) / (torch.sum(P) + P_c)
    return ee


def compute_V(H):
    """
    :param H: 信道 [b,k,nt]
    :return:
    """
    # 先获得G
    G = H.conj()
    # 计算 G 的共轭转置
    G_conj_transpose = torch.conj(G.transpose(1, 2))

    # 计算 G G^H 的乘积
    G_G_conj_transpose = torch.einsum('bij,bjk->bik', G, G_conj_transpose)

    # 计算 (G G^H)^-1 的逆矩阵
    G_G_conj_transpose_inv = torch.inverse(G_G_conj_transpose)

    # 计算 V = G^H (G G^H)^-1
    V = torch.einsum('bij,bjk->bik', G_conj_transpose, G_G_conj_transpose_inv)

    return V


def hybrid_beamforming(H, V, alpha, P):
    """

    :param H:  信道值 [b,k,nt]
    :param V:  [b,nt,k]
    :param alpha: [b,k]
    :param P: [b,k]
    :return:
    """
    # 计算方向分量
    norm_H = torch.norm(H, dim=2, keepdim=True, p=2)  # 计算 H 的 L2范数
    direction_component = H / norm_H  # 计算 H 的单位向量

    # 计算功率分量
    V = V.transpose(1, 2)
    norm_V = torch.norm(V, dim=2, keepdim=True, p=2)  # 计算 V 的 L2范数
    power_component = V / norm_V  # 计算 V 的单位向量

    # 组合方向分量和功率分量
    w = alpha.unsqueeze(2) * direction_component + (1 - alpha.unsqueeze(2)) * power_component
    # 根据 P 缩放向量
    w = torch.sqrt(P.unsqueeze(2)) * w
    return w


def get_hybrid_beamforming(H, alpha, P):
    """

    :param H:  信道值 [b,k,nt]
    :param alpha: [b,k]
    :param P: [b,k]
    :return:
    """
    # 先获得G
    H = H.reshape(alpha.shape[0], alpha.shape[1], -1)
    G = H.conj()
    # 计算 G 的共轭转置
    G_conj_transpose = torch.conj(G.transpose(1, 2))

    # 计算 G G^H 的乘积
    G_G_conj_transpose = torch.einsum('bij,bjk->bik', G, G_conj_transpose)

    # 计算 (G G^H)^-1 的逆矩阵
    G_G_conj_transpose_inv = torch.inverse(G_G_conj_transpose)

    # 计算 V = G^H (G G^H)^-1
    V = torch.einsum('bij,bjk->bik', G_conj_transpose, G_G_conj_transpose_inv)

    # 计算方向分量
    norm_H = torch.norm(H, dim=2, keepdim=True, p=2)  # 计算 H 的 L2范数
    direction_component = H / norm_H  # 计算 H 的单位向量

    # 计算功率分量
    V = V.transpose(1, 2)
    norm_V = torch.norm(V, dim=2, keepdim=True, p=2)  # 计算 V 的 L2范数
    power_component = V / norm_V  # 计算 V 的单位向量

    # 组合方向分量和功率分量
    w = alpha.unsqueeze(2) * direction_component + (1 - alpha.unsqueeze(2)) * power_component

    # 根据 P 缩放向量
    w = torch.sqrt((P + 1e-6).unsqueeze(2)) * w

    return w


def get_beamforming(H, args, out):
    """
    :param H:  信道值 [b,k,nt]
    :param args: 配置参数
    :param out: 网络输出的值 p,alpha 或者单独的 p
    :return: 返回beam
    """
    # 先获得G
    H = H.reshape(-1, args.user_num, args.antenna_num)

    G = H.conj()
    # 计算 G 的共轭转置
    G_conj_transpose = torch.conj(G.transpose(1, 2))

    # 计算 G G^H 的乘积
    G_G_conj_transpose = torch.einsum('bij,bjk->bik', G, G_conj_transpose)

    # 计算 (G G^H)^-1 的逆矩阵
    G_G_conj_transpose_inv = torch.inverse(G_G_conj_transpose)

    # 计算 V = G^H (G G^H)^-1
    V = torch.einsum('bij,bjk->bik', G_conj_transpose, G_G_conj_transpose_inv)

    # 计算功率分量
    V = V.transpose(1, 2)
    norm_V = torch.norm(V, dim=2, keepdim=True, p=2)  # 计算 V 的 L2范数
    power_component = V / norm_V  # 计算 V 的单位向量

    # 计算方向分量
    norm_H = torch.norm(H, dim=2, keepdim=True, p=2)  # 计算 H 的 L2范数
    direction_component = H / norm_H  # 计算 H 的单位向量

    if args.type == 'MRT':
        # 根据 P 缩放向量
        # print(direction_component.shape)
        # print(out.shape)
        w = torch.sqrt((out + 1e-8).unsqueeze(2)) * direction_component
        return w
    elif args.type == 'ZF':
        # 根据 P 缩放向量
        w = torch.sqrt((out +  1e-8).unsqueeze(2)) * power_component
        return w
    elif args.type == 'MMSE':
        # 计算 (G G^H)^-1 的逆矩阵
        G_G_conj_transpose_inv = torch.inverse(G_G_conj_transpose + args.noise * torch.eye(args.user_num).to('cuda'))

        # 计算 V = G^H (G G^H)^-1
        M = torch.einsum('bij,bjk->bik', G_conj_transpose, G_G_conj_transpose_inv)

        # 计算功率分量
        M = M.transpose(1, 2)
        norm_M = torch.norm(M, dim=2, keepdim=True, p=2)  # 计算 V 的 L2范数
        power_component = M / norm_M  # 计算 V 的单位向量
        # 根据 P 缩放向量
        w = torch.sqrt((out + 1e-7).unsqueeze(2)) * power_component
        return w
    else:
        # 组合方向分量和功率分量
        w = out[1].unsqueeze(2) * direction_component + (1 - out[1].unsqueeze(2)) * power_component

        # 根据 P 缩放向量
        w = torch.sqrt((out[0] + 1e-6).unsqueeze(2)) * w

        return w

