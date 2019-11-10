# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 23:10:14 2019

@author: us
"""


'''
1. What does a neuron compute?
Ans:一个神经元计算包括一部分线性计算和一个激活函数

2. Why we use non-linear activation funcitons in neural networks?
Ans:如果不用激活函数函数这类非线性函数，在这种情况下每一层输出都是上层输入的线性函数，
    无论神经网络有多少层，输出都是输入的线性组合，与没有隐藏层效果相当

3. What is the 'Logistic Loss' ?
Ans:Logistic Loss和Crossing Entropy Loss本质上是一回事
    回归问题loss function可以用MSE和绝对误差
    分类问题会用Logistic Loss： cost=-Σp(i)log(q(i))
    
4. Assume that you are building a binary classifier for detecting 
if an image containing cats, which activation functions 
would you recommen using for the output layer ?
Ans:
    ReLU或者Leaky ReLU
    
A. ReLU：Relu=max(0,x)
    优点：
    1） 解决了gradient vanishing问题 (在正区间)
    2）计算速度非常快，只需要判断输入是否大于0
    3）收敛速度远快于sigmoid和tanh
    问题：
    1）ReLU的输出不是zero-centered
    2）Dead ReLU Problem，指的是某些神经元可能永远不会被激活，导致相应的参数永远不能被更新
B. Leaky ReLU：f(x)=max(αx,x)
    将ReLU的前半段设为αx而非0，通常α=0.01
    理论上来讲，Leaky ReLU有ReLU的所有优点，外加不会有Dead ReLU问题

C. sigmoid:能够把输入的连续实值变换为0和1之间的输出.
    缺点：在深度神经网络中梯度反向传递时导致梯度爆炸和梯度消失，
    其中梯度爆炸发生的概率非常小，而梯度消失发生的概率比较大。
    Sigmoid 的 output 不是0均值（即zero-centered）
D. tanh：解决了Sigmoid函数的不是zero-centered输出问题，
    然而，梯度消失（gradient vanishing）的问题和幂运算的问题仍然存在。
    

5. Why we don't use zero initialization for all parameters ?
Ans:w初始化全为0，由于参数相同以及输出值都一样，不同的结点根本无法学到不同的特征
    很可能直接导致模型失效，无法收敛
    
6. Can you implement the softmax function using python ?
Ans:SoftMaxLoss就是一般二分类LogisitcLoss的推广,可用于多分类问题
基于numy实现softmax功能
    
'''

import numpy as np
 
def softmax_np(logits):
    assert (isinstance(logits, np.ndarray)), 'only numpy is available'
    exp_value = np.exp(logits)  # 计算指数值
    dim_ext = np.sum(exp_value, 1).reshape(-1, 1)
    return exp_value / dim_ext
 
x_val = [[1, 2, 3], [3, 2, 2]]
logits = np.array(x_val)
s_v = softmax_np(logits)
print (s_v, np.sum(s_v, 1))