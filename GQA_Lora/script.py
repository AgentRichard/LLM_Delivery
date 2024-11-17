import torch
import torch.nn as nn

class test(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(2, 3, bias=False)
        self.layer.reset_parameters()
        print("layer权重初始化后的值为:", self.layer.weight)

        self.layer2 = nn.Linear(3, 2, bias=False)
        nn.init.zeros_(self.layer2.weight)
        print("layer2初始化后的权重为:",self.layer2.weight)
        
    def forward(self, x):
        _x = self.layer(x)
        print("x经过layer层后的值为:", _x)
        _x = self.layer2(_x)
        print("x经过layer2层后的值为:", _x)
        print("经过x运算后的层2的权重",self.layer2.weight)
        return _x

torch.manual_seed(0)
t = test()
y = t(torch.randn(3, 2))
print(y)




