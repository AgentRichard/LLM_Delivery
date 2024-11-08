import torch
import torch.nn as nn

class FCNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        self._layer1 = nn.Linear(784, 1024)
        self._act1 = nn.SiLU()
        self._dropout1 = nn.Dropout(0.5)
        self._layer_norm = nn.LayerNorm(1024)
        self._layer2 = nn.Linear(1024, 1024)
        self._act2 = nn.SiLU()
        self._dropout2 = nn.Dropout(0.5)
        self._layer_norm = nn.LayerNorm(1024)
        self._layer3 = nn.Linear(1024, 784)
        self._act3 = nn.SiLU()
        self._dropout3 = nn.Dropout(0.5)
        self._layer_norm = nn.LayerNorm(784)
        self._layer4 = nn.Linear(784, 10)
        self._fact = nn.Softmax(-1)
        
    def forward(self, x):
        _x = x
        _x = self._dropout1(self._act1(self._layer1(_x)))
        _x = self._dropout2(self._act2(self._layer2(_x)))
        _x = self._dropout3(self._act3(self._layer3(_x)))
        _x = self._fact(self._layer4(_x))
        return _x
    
if __name__ == "__main__":
    x = torch.randn(2, 784)
    model = FCNN()
    print(model(x), model(x).shape)
    

    

        
        
        