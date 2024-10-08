import torch
import math

# 生成一个随机的6x6复数张量
real_part = torch.randn(6, 6)  # 实部
imag_part = torch.randn(6, 6)  # 虚部
W = real_part + 1j * imag_part  # 复数张量


# print(torch.angle(W))
# print(math.pi)

# 定义量化函数
def quantize_phase(B, W):
    delta = 2 * math.pi / 2 ** B  # 量化间隔
    # r = torch.zeros_like(W, dtype=torch.complex64)  # 初始化复数张量

    # 计算相位并量化
    phase = torch.angle(W)  # 获取相位
    phase_quantized = torch.round(phase / delta) * delta  # 优化后的量化相位
    magnitude = torch.abs(W)  # 获取幅度
    # 量化后的复数张量
    r = magnitude * torch.exp(1j * phase_quantized)
    return r

# torch.pi


# 量化复数张量W的相位（以3位精度为例）
B = 3
W_quantized = quantize_phase(B, W)
#
# # 输出原始和量化后的复数张量
delta = 2 * math.pi / 2 ** B
print(delta)
print("Original Tensor:\n", torch.angle(W))
print("\nQuantized Tensor:\n", torch.angle(W_quantized))
print(torch.angle(W) / delta)
print(torch.angle(W_quantized) / delta)
