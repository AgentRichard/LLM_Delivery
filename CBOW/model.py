import torch
import torch.nn as nn

class CBOW(nn.Module):
    """最主要的部分就是将一个句子中的不同token的所有特征进行融合，来预测被覆盖的token"""
    def __init__(self, vocb_size, embedding_dim):
        super().__init__()
        
        self._emb = nn.Linear(vocb_size, embedding_dim)
        self._output = nn.Linear(embedding_dim, vocb_size)
        
    def forward(self, x):
        embedding = self._emb(x)                  ## 这里的dim处理是对序列进行求均值，一个样本中，不同token对应特征位置求均值。
        embedding = torch.mean(embedding, dim=0)  
        ## 对批次求平均，回想batchnormal，每个token的对应特征位置求平均纵向 [2, 20]，融合的是不同样本中即不同话的对应位置token的
        ## 对序列求平均，即每个token的对应位置纵向求平均并融合，得到[1, 20]， 融合的是两个token的对应位置特征
        ## 对向量求平均，即每个token的所有特征加总除以20，得到一个标量, 融合的是最后一个维度的所有特征
        output = self._output(embedding.unsqueeze(0))
        return output
    
if __name__ == "__main__":
    cbow = CBOW(9, 20)
    test = torch.randn(2, 9) ## 模型的输入是单个样本的处理
    print(cbow(test).shape)

    
