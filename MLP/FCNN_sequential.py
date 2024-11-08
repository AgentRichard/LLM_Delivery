import torch
import torch.nn as nn

class FCNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        self._layer1 = nn.Sequential(
            nn.Linear(784, 1024),
            nn.SiLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 2048),
            nn.SiLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 10),
            nn.Softmax(-1)
        )

        
    def forward(self, x):
        return self._layer1(x)
    
if __name__ == "__main__":
    x = torch.randn(2, 784)
    model = FCNN()
    print(model(x), model(x).shape)
    