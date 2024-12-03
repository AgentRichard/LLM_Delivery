import torch
import torch.nn as nn

class Convolution_ResNet(nn.Module):
    """
    1. 残差网络最重要的是保证单样本的形状完全相同就可以加
    2. 多加残差保证梯度不会弥散，残差的相加不改变梯度，不改变权重，改变的是输入与输出。
    """
    def __init__(self):
        super().__init__()
                                        # [3, 1, 28, 28]
        self._layer0 = nn.Sequential(
            nn.Conv2d(1, 6, 3, 1, 1),   # [3, 6, 28, 28]
            nn.BatchNorm2d(6),
            nn.SiLU(),                  
        )
        
        self._layer1 = nn.Sequential(
            nn.Conv2d(6, 12, 3, 1, 1),  # [3, 12, 28, 28]
            nn.Dropout(0.5),
            nn.BatchNorm2d(12),
            nn.SiLU(),
            nn.Conv2d(12, 6, 3, 1, 1),  # [3, 6, 28, 28]
            nn.Dropout(0.5),
            nn.BatchNorm2d(6),
            nn.SiLU()
        )
        
        self._layer2 = nn.Sequential(
            nn.Conv2d(6, 6, 3, 1, 1),    # [3, 6, 28, 28]
            nn.Dropout(0.5),
            nn.SiLU(),
        )
        
        self._output_layer = nn.Sequential(
            nn.MaxPool2d(2, 2),         # [3, 6, 14, 14]
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(1176, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Linear(256, 10)
        )
        
        self.apply(self._init_weight)
        
    def _init_weight(self, model):
        for m in model.modules():
            if isinstance(m, torch.nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                # if m.bias:
                #     nn.init.zeros_(m.bias)
            elif isinstance(m, torch.nn.Linear):
                nn.init.xavier_normal_(m.weight)
                # if m.bias:
                #     nn.init.zeros_(m.bias)
        
    def forward(self, x):
        _x = self._layer0(x)
        _x1 = self._layer1(_x)
        _h = _x1 + _x
        _h1 = self._layer2(_h)
        _h1 = _h1 + _h + _x          ### 把能加残差地方都加了
        _y = self._output_layer(_h1)
        return _y

    
if __name__ == "__main__":
    net = Convolution_ResNet()
    x = torch.randn(3, 1, 28, 28)
    y = net(x)
    print(y.shape)