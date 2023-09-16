import torch
import torch.nn.functional as F
import numpy as np


def onehot_from_logits(logits, avaliable_action, is_used_in_gumbel_softmax=False, eps=0.05):
    """ 生成最优动作的独热（one-hot）形式 """

    ava_action = []  # 可以动作集
    unava_action = []  # 不可行动作集

    for i in range(avaliable_action.size(1)):
        if avaliable_action[0][i] == 1.0:
            ava_action.append(i)
        else:
            if avaliable_action[0][i] == -float('inf'):
                unava_action.append(i)

    if is_used_in_gumbel_softmax:
        # 如果用在gumbel_softmax里面
        argmax_acs = (logits == logits.max(1, keepdim=True)[0]).float()
    else:
        # 如果不是
        for i in range(logits.size(0)):
            for j in unava_action:
                logits[i][j] = -float('inf')
        argmax_acs = (logits == logits.max(1, keepdim=True)[0]).float()
    # 生成随机动作,转换成独热形式

    #####
    #  进行mask
    #####

    rand_acs = torch.autograd.Variable(torch.eye(logits.shape[1])[[
        np.random.choice(ava_action, size=logits.shape[0])  # 存在BUG
    ]], requires_grad=False).to(logits.device)
    # 通过epsilon-贪婪算法来选择用哪个动作
    return torch.stack([
        argmax_acs[i] if r > eps else rand_acs[i]
        for i, r in enumerate(torch.rand(logits.shape[0]))
    ])


def sample_gumbel(shape, eps=1e-20, tens_type=torch.FloatTensor):
    """从Gumbel(0,1)分布中采样"""
    U = torch.autograd.Variable(tens_type(*shape).uniform_(),
                                requires_grad=False)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    """ 从Gumbel-Softmax分布中采样"""
    y = logits + sample_gumbel(logits.shape, tens_type=type(logits.data)).to(
        logits.device)
    return F.softmax(y / temperature, dim=1)


def gumbel_softmax(logits, avaliable_action, temperature=1.0):
    """从Gumbel-Softmax分布中采样,并进行离散化"""
    logits_ = logits
    for index in range(avaliable_action.size(1)):
        if avaliable_action[0][index] == -float('inf'):
            for logit_index in range(logits_.size(0)):
                logits_[logit_index][index] = -float('inf')

    y = gumbel_softmax_sample(logits_, temperature)
    y_hard = onehot_from_logits(y, avaliable_action, True)
    y = (y_hard.to(logits_.device) - y).detach() + y
    # 返回一个y_hard的独热量,但是它的梯度是y,我们既能够得到一个与环境交互的离散动作,又可以
    # 正确地反传梯度
    return y


if __name__ == "__main__":
    logits = torch.randn(1, 5)
    ava = torch.ones(1, 5)
    ava[0][3] = -float('inf')
    ava[0][4] = -float('inf')
    print(logits)

    print(gumbel_softmax(logits, ava))

    print(onehot_from_logits(logits, ava, False))
