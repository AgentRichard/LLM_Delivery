import torch
import torch.nn as nn

class Convolution(nn.Module):
    '''只要卷积核大小是3，步长是1，空洞是1，padding是1，那么输出一定跟原图形尺寸一致'''
    def __init__(self):
        super().__init__()
        self._layer1 = nn.Sequential(
            nn.Conv2d(                  #下一次卷积的通道数通常是前一次的两倍，共识
                in_channels=1,out_channels=3,
                kernel_size=3,stride=1,
                padding=1, dilation=1   #默认空洞卷积是1         
                ),                      #[N, c=1, h=28, w=28]
            nn.BatchNorm2d(3),
            nn.Dropout2d(0.5),
            nn.SiLU(),
            nn.MaxPool2d(2, 2),         #[N,3,14,14]
        )
        
        self._layer2 = nn.Sequential(
            nn.Conv2d(3, 6, 3, 1, 1),   #[N,3,14,14]
            nn.BatchNorm2d(6),
            nn.Dropout2d(0.5),
            nn.SiLU(),
            nn.MaxPool2d(2, 2),         #[N,6,7,7]
        )
        
        self._layer3 = nn.Sequential(
            nn.Conv2d(6, 12, 3, 1, 1),  #[N,6,7,7]
            nn.BatchNorm2d(12),
            nn.Dropout2d(0.5),
            nn.SiLU(),
            nn.MaxPool2d(2, 2),         #[N,12,3,3]  笔记中的池化尺寸计算，向上取整 
        )
        
        self._layer4 = nn.Sequential(
            nn.Conv2d(12, 24, 3, 1, 1),  #[N,24,3,3]
            nn.BatchNorm2d(24),
            nn.Dropout2d(0.5),
            nn.SiLU()
        )
        
        self._output_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(216, 216),
            nn.SiLU(),
            nn.BatchNorm1d(216),  ### 这里之所以会报错，是因为之前batchsize设置为1 lolllllll
            nn.Linear(216, 10),
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
        _x = self._layer1(x)
        _x = self._layer2(_x)
        _x = self._layer3(_x)
        _x = self._layer4(_x)
        _x = self._output_layer(_x)
        return _x
    
if __name__ == "__main__":
    conv = Convolution()
    inputs = torch.randn(2, 1, 28, 28)
    print(conv(inputs).shape)
    print(conv(inputs))
    for k,v in conv.named_modules():
        if k.startswith("_layer1") and isinstance(v, nn.Conv2d):
            print(v)
            print(v.weight)
            print(v.weight.data)
            print(v.weight.grad)
    print("--------------")
    for m in conv.modules():
        print(m)