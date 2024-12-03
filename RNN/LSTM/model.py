from torch import nn
import torch

class LSTM_Model(nn.Module):
    def __init__(self, input_size, output_size, num_layers):
        super().__init__()
        self._emb = nn.Embedding(30000, 512)
        self._net = nn.LSTM(input_size, output_size, num_layers)
        
    def forward(self, x):
        x = self._emb(x)
        _x = self._net(x)
        return _x[0]@self._emb.weight.T, _x[1] 
    
if __name__ == "__main__":
    lstm = LSTM_Model(512, 512, 6)
    a = torch.randint(0, 30000, (2, 7))
    print(lstm(a)[0].shape, lstm(a)[1][0].shape, lstm(a)[1][1].shape)
    ## 上面的输出分别是 所有数据的最后一层lstm的输出为NSV结构，
    ## 所有层每个样本的最后一个Token的输出H
    ## 所有层每个样本的最后一个Token的记忆C
    ## Outputs: output, (h_n, c_n)