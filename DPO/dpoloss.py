import torch
import torch.nn as nn

class DPOLoss(nn.Module):
    def __init__(self, beta=0.1):
        super().__init__()
        
        self._beta = beta
        
    def forward(self, chosen_logit, 
                ref_chosen_logit,
                reject_logit,
                ref_reject_logit,
                chosen_lable,
                reject_lable):
        
        ### 训练的过程是 x[:, :-1] 输入到模型中 假设10个token，因此输入的是 9个token
        ### 去预测 lable[:, 1: ]对应的下一个token，因此并行运算时是因果矩阵，9个token一起计算
        ### 预测时的损失计算是 x = x.reshape(-1, 30000), y = y.reshape(-1)
        ### 因此损失是 假设批次是 1, 那么就是 9个token的 概率分布进行损失计算， 每个概率分布的大小是30000
        ### _chosen_log_prob = log_softmax((9, 30000), dim=-1) 得到一个 (9, 30000) 个负数此时还没计算损失
        _chosen_log_prob = torch.log_softmax(chosen_logit, dim=-1)
        _ref_chosen_log_prob = torch.log_softmax(ref_chosen_logit, dim=-1)
        _reject_log_prob = torch.log_softmax(reject_logit, dim=-1)
        _ref_reject_log_prob = torch.log_softmax(ref_reject_logit, dim=-1)
        
        ### 选出每个对应模型，每个lable上(30000 中的1个)对应的预测值，因为是log，因此是负数
        ### 不用管批次与序列，因为现在仅需要计算损失,要添加维度，因为维度要保持一致 ex. chosenlabel [1, 10, 1]
        ### 最后出来就是 NS1的形状，因此我们要去掉最后一个维度
        print(chosen_lable[:,None])
        _chosen_lable = torch.gather(_chosen_log_prob, dim=-1, index=chosen_lable[:,None])[:,0]
        _ref_chosen_lable = torch.gather(_ref_chosen_log_prob, dim=-1, index=chosen_lable[:,None])[:,0]
        _reject_label = torch.gather(_reject_log_prob, dim=-1, index=reject_lable[:,None])[:,0]
        _ref_reject_label = torch.gather(_ref_reject_log_prob, dim=-1, index=reject_lable[:,None])[:,0]
        
        ### 求和, 代表这个批次，1句话的损失
        _chosen_loss = _chosen_lable.sum(-1) ## 因为是NS结构，[1,10] 损失 因为已经求过log，直接求和
        _ref_chosen_loss = _ref_chosen_lable.sum(-1)
        _reject_loss = _reject_label.sum(-1)
        _ref_reject_loss = _ref_reject_label.sum(-1)
        
        ### DPO的策略，也就是正则，因为全是log值，那么可以直接进行加减
        ### chosen(9, 30000) - _ref_chosen(9, 30000), 最开始chosen和ref是相同的，因此是0，随着训练，chosen一定
        ### 大于ref_chosen, 得到正值
        ### rej随着训练做完log后会变得更小，ref_rej仍保持原值，也就是rej-ref_rej会变得更小，但仍是负值
        ### 减去参考模型的输出 后者reject就是让模型的更新幅度避免过大 前者是要求模型比参考训练表现得更好
        _diff = (_chosen_loss - _ref_chosen_loss) - (_reject_loss - _ref_reject_loss)
        
        _loss = torch.sigmoid(self._beta * _diff).mean()
        return _loss

if __name__ == "__main__":
    dpo = DPOLoss(0.1)
    a,b,c,d = torch.randn(3, 10),torch.randn(3, 10),torch.randn(3, 10),torch.randn(3, 10)
    label1, lable2 = torch.randint(0, 30000, (2,3)).squeeze(1), torch.randint(0, 30000, (2,3)).squeeze(dim=0)
    print(dpo(a,b,c,d,label1,lable2))
    
