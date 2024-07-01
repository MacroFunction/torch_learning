import torch
x = torch.arange(12)
print(x)
# tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])
# 通过张量的shape属性可以访问张量的形状（沿着每个轴的长度）
print(x.shape)
# torch.Size([12])
# reshape函数来改变张量的的形状
X = x.reshape(3, 4)
print(X)
# tensor([[ 0,  1,  2,  3],
#         [ 4,  5,  6,  7],
#         [ 8,  9, 10, 11]])

# 可以通过再希望张量自动推断的维度放置-1使得张量再给出其他部分侯自动计算出一个维度
X = x.reshape(-1, 4)

# 可以使用zeros/ones/randn获得已过期全0/全1/特定分布中堆积采样的数字初始化矩阵
print(torch.zeros((2, 3, 4)))
print(torch.ones((2, 3, 4)))
print(torch.randn((2, 3, 4)))

x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
print(x + y)
print(x - y)
print(x * y)
print(x / y)
print(x ** y)
tmp = x ** y
# exp为e的幂次方
print(torch.exp(tmp))

# 通过cat函数将多个张量连接到一起
x = torch.arange(12, dtype=torch.float32).reshape((3, 4))
y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
print(torch.cat((x, y), dim=0))
print(torch.cat((x, y), dim=1))
print(x == y)
# 对张量中的所有的元素进行求和产生一个只有一个元素的张量
print(x.sum())

a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))
print(a, b)
print(a + b)

print(x[-1])
print(x[1:3])
print(x[1, 2])
print(x[0:2, :])

import os
os.makedirs(os.path.join('..', 'data'), exist_ok=True)
data_file = os.path.join('..', 'data', 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n')
    f.write('NA,NA,127500\n')
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')

import pandas as pd

data = pd.read_csv(data_file)
print(data)
inputs,outputs = data.iloc[:,0:2],data.iloc[:, 2]
inputs = inputs.fillna(inputs.mean())
print(inputs)

inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)

import torch
x, y = torch.tensor(inputs.values), torch.tensor(outputs.values)
print(x, y)

import torch
x = torch.tensor([3.0])
y = torch.tensor([2.0])

print(x + y, x * y, x / y, x**y)

x = torch.arange(4)
print(x)
print(x[3])

print(len(x))
print(x.shape)
A = torch.arange(20).reshape(5, 4)
print(A)
print(A.T)

B = torch.tensor([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
print(B)
print(B == B.T)

X = torch.arange(24).reshape(2, 3, 4)
print(X)


A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
B = A.clone()
print(A, A + B)
print(A * B)

a = 2
X = torch.arange(24).reshape(2, 3, 4)
print(a + X)
print(a * X)

x = torch.arange(4, dtype=torch.float32)
print(x, x.sum())
print(A.shape, A.sum())

A_sum_axis0 = A.sum(axis=0)
print(A_sum_axis0, A_sum_axis0.shape)

A_sum_axis1 = A.sum(axis=1)
print(A_sum_axis1, A_sum_axis1.shape)

print(A.sum(axis=[0, 1]))
print(A.mean(), A.sum() / A.numel())
print(A.mean(axis=0))
print(A)
print(A.sum(axis=0))
sum_A = A.sum(axis=1, keepdims=True)
print(sum_A)
print(A / sum_A)
print(A.cumsum(axis=0))

y = torch.ones(4, dtype=torch.float32)
print(x, y, torch.dot(x, y))
print(torch.sum(x * y))
print(A.shape, x.shape, torch.mv(A, x))
print(A)
B = torch.ones(4, 3)
print(torch.mm(A, B))
u = torch.tensor([3.0, -4.0])
print(torch.norm(u))

A = torch.abs(u).sum()
print(A)

torch.norm(torch.ones((4, 9)))

import numpy as np

def f(x):
    return 3 * x ** 2 - 4 * x
def numerical_lim(f, x, h):
    return (f(x + h) - f(x)) / h
h = 0.1
for i in range(5):
    print(f'h = {h:.5f}, numerical limit={numerical_lim(f, 1, h):.5f}')
    h *= 0.1
# import numpy as np
# from IPython import display
# from d2l import torch as d2l
# def use_svg_display():
#     # display.set_matplotlib_formats('svg')
#     pass
#
# def set_figsize(figsize=(3.5, 2.5)):
#     use_svg_display()
#     d2l.plt.rcParams['figure.figsize'] = figsize
#
# def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
#     axes.set_xlabel(xlabel)
#     axes.set_ylabel(ylabel)
#     axes.set_xscale(xscale)
#     axes.set_yscale(yscale)
#     axes.set_xlim(xlim)
#     axes.set_ylim(ylim)
#     if legend:
#         axes.legend(legend)
#     axes.grid()
#
# def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None, ylim=None, xscale='linear', yscale='linear',fmts=('-','m--','g-.', 'r:'), figsize=(3.5, 2.5), axes=None):
#     '''绘制数据点'''
#     if legend is None:
#         legend = []
#
#     set_figsize((figsize))
#     axes = axes if axes else d2l.plt.gca()
#
#     def has_one_axis(X):
#         return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list) and not hasattr(X[0], "__len__"))
#     if has_one_axis(X):
#         X = [X]
#     if Y is None:
#         X, Y = [[]] * len(X), X
#     elif has_one_axis(Y):
#         Y = [Y]
#     if len(X) != len(Y):
#         X = X * len(Y)
#     axes.cla()
#     for x, y, fmt in zip(X, Y, fmts):
#         if len(x):
#             axes.plot(x, y, fmt)
#         else:
#             axes.plot(y, fmt)
#     set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
#
# x = np.arange(0, 3, 0.1)
# plot(x, [f(x), 2*x-3], 'x', 'f(x)', legend=['f(x)', 'Tangent line (x=1)'])

import torch
x = torch.arange(4.0)
print(x)

x.requires_grad_(True)
print(x.grad)
y = 2 * torch.dot(x, x)
print(y)
print(y.backward())
print(x.grad == 4 * x)

x.grad.zero_()
y = x.sum()
y.backward()
print(x.grad)

x.grad.zero_()
y = x * x
y.sum().backward()
print(x.grad)

x.grad.zero_()
y = x * x
u = y.detach()
z = u * x

z.sum().backward()
print(x.grad == u)

def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c

a = torch.randn(size=(), requires_grad=True)
d = f(a)
d.backward()
print(a.grad)
print(d / a)
