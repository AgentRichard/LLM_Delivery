from torch import nn
import torch

class RNNLayer(nn.Module):
    """
    每层的处理方式:
    :params = x, h_pre
    :timestep(一个token计算完才能处理下一个token)
    :以下仅显示对每个token的处理过程，且是一层RNN
    :RNN 要求每个批次的Seq长度保持一致，不一致的添加padd
    """
    def __init__(self, input_dim, output_dim, bias=True):
        super().__init__()
        self._xlayer = nn.Linear(input_dim, output_dim)
        self._hlayer = nn.Linear(output_dim, output_dim)
        self._gate_act = nn.Sigmoid()
        
    def forward(self, x, h_prev_layer_not_t):
        """
        神经网络设计非常灵活，前一次输出与当前的输入的维度一致，因此组合起来即可以Concate，也可以加起来
        标准的是concate, 这里用的是累加
        """
        return self._gate_act(self._xlayer(x) + self._hlayer(h_prev_layer_not_t))
    
class RNNLayers(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers):
        super().__init__()
        
        self._output_dim = output_dim
        self._num_layers = num_layers
        
        self._layers = nn.ModuleList(
            [RNNLayer(input_dim, self._output_dim) for _ in range(self._num_layers)] ### 这样创建的方法权重也不一样，不是直接复制,放心用
        )
        
    
    def forward(self, x):
        """
        对于文本，NSV结构就是一个样本中仅包含一句话，每句话有多个token也就是多个词，每个词有相同的维度
        因为RNN结构特性，非常擅长处理时间序列任务，因此不仅具有记忆能力，也可以理解每个词在当前句子中的关系
        因此如果处理的是 文本 整个计算逻辑为：
        1. 将x转置为 SNV结构，发挥一定并行计算能力，依次处理不同样本中，每个token在所属样本中的相关性。
        2. 根据上文，我们需要创建一个隐藏状态表用来保存 不同样本的每个token再经过每一层的隐藏状态 h，
        3. 表尺寸为 [num_layers, batch, hiddensize], 首先每次保存的状态的最后一个维度一定是hiddensize，其次在相同的时间点上 不同的样本有自己独立的状态，因此是batch
        4. 最后 相同时间点上模型处理的是不同批次的相同位置token，并且随着时间步的循环，每个token都是同样大小的隐藏状态表尺寸[num, b, hid]，用来更新即可
        5. 因此计算完每个句子的最后一个token之后(包括padding)，隐藏状态表一定是每个样本的最后一个token的6层隐藏状态 
        
        因为我们的数据是NSV结构，因此直接转置成SNV
        X是NSV
        H是NV，因为我们每次是对序列中的1个token进行处理
        """
        _x = x.transpose(0, 1)
        _init_hidden_frame = torch.zeros(self._num_layers, _x.shape[1], self._output_dim, dtype=torch.float32, device=_x.device) ## [layers_num, batch, outdim]
        
        h_prev_layer_not_t = _init_hidden_frame
        h_layer_final_output = _init_hidden_frame
        final_ans = []
        
        for t in range(_x.shape[0]):
            """以序列长度为时间点"""
            for _index, layer in enumerate(self._layers):
                """取出每一层RNN"""
                
                layer_output = layer(_x[t], h_prev_layer_not_t[_index].clone())                   ## [NV:所有样本的第t个token, 每个token当前的第layer层]
                h_layer_final_output[_index] = layer_output.clone()                             ## 记录一下一个token的每层的输出状态
            
            final_ans.append(h_layer_final_output[-1].clone())                                    ## 我们只取每个token的最后一个状态进行记录
            h_prev_layer_not_t = h_layer_final_output.clone()                                     ## 后续token的所有层的初始状态均为前一个token的所有层的初始状态
        # print(len(final_ans))
        # print(h_layer_final_output[-1].shape)
        final_ans = torch.stack(final_ans)
        # print(final_ans.shape)
        final_ans = final_ans.transpose(0, 1)                                             ## final_ans 是不同样本中所有token的最终输出列表，[(NV_token1), (NV_token2), ...]
        # print(final_ans.shape)
        return final_ans, h_layer_final_output                                            ## 因此stack将他们并起来得到 [S, N, V] 转置 [N, S, V]
        # 输出 [N,S,V]张量和 [num_layers, N,V] 且是每个seq的最后一个token的所有层的输出状态

class RNNModel(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers):
        super().__init__()
        
        self._net = RNNLayers(input_dim, output_dim, num_layers)
        self._emb = nn.Embedding(30000, 256)
        self._linear = nn.Linear(64, 256)
        self._output = nn.Linear(output_dim, 30000)
    
    def forward(self, x):
        _x = self._emb(x)
        _h = self._net(_x)
        _h1 = self._linear(_h[0])
        return _h1@self._emb.weight.T, _h[1]

if __name__ == "__main__":
    model = RNNModel(256, 64, 6).cuda().to(dtype=torch.float32)
    x = torch.randint(0, 30000, (32, 63)).cuda()
    h = model(x)
    print("模型经过6层RNN输出NSV结构的数据,本质上NSV结构中每个S的最后一个token最能体现它在整句话中的特征，并且包含了整句话的所有信息。",h[0].shape)
    print("模型经过6层RNN输出每句话也就是每个批次最后一个token的所有隐藏状态为 L，N,V结构", h[1].shape)
    for k, v in model.named_modules():
        print(k)
    """
    可以看到RNN的优化是在批次维度上的计算速度优化，优势是模型有记忆，但是当句子长起来之后记忆肯定不会好，并且每个token其实会与最接近它的token更相关
    除此之外，token与token间无法做到并行运算，无法充分发挥并行运算的优势，并且不同批次句子间没有任何联系，如果文本过长容易梯度弥散或者梯度爆炸，因为前一次的输出状态
    与当前的输出状态紧密关联，当反向传播的时候，特别是在最开始的几层容易梯度弥散。
    RNN模型本质上也不适用于深层网络。
    """