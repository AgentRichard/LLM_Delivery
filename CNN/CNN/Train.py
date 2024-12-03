import torch.nn as nn
from Dataloader import Data
from torch.optim.adam import Adam
from torch.utils.data import DataLoader
from Model import Convolution
import torch
from torch.utils.tensorboard import SummaryWriter

class Train:
    def __init__(self):
        
        self._net = Convolution().cuda()
        
        self._train_data = Data("/root/project/My_projects/FCNN/data/train")
        self._test_data = Data("/root/project/My_projects/FCNN/data/test")
        self._train_dataset = DataLoader(self._train_data, batch_size=5000, shuffle=True, num_workers=20)
        self._test_dataset = DataLoader(self._test_data, batch_size=5000, shuffle=True, num_workers=20)
        
        self._opt = Adam(self._net.parameters(), lr=0.0001, betas=[0.85, 0.95], weight_decay=0.001)
        self._log = SummaryWriter(log_dir="/root/project/My_projects/CNN/runs")
        self._lossfn = nn.CrossEntropyLoss() 
        ### 交叉熵损失自带One HOT 因此在数据处理时不需要 转成onehot也不需要转成浮点数！！！！！！！！！！！！！！！！！！
    
    def __call__(self):
        for _epoch in range(100000000000000000):
            
            self._net.train()
            _loss_sum = 0
            self._net.load_state_dict(torch.load("/root/project/My_projects/CNN/weights/conv_weight.pt"))
            for _index, (_data, _label) in enumerate(self._train_dataset):
                _x = _data.cuda()
                _t = _label.cuda()
                _y = self._net(_x)
                _loss = self._lossfn(_y, _t)
                
                self._opt.zero_grad()
                _loss.backward()
                self._opt.step()
                
                _loss_sum += _loss.detach().cpu().item()
            self._log.add_scalar(f"loss", _loss_sum/len(self._train_dataset), _epoch)
            try:
                for _name, _layer in self._net.named_modules():
                    if _name.startswith("_layer1") and isinstance(_layer, nn.Conv2d):
                        self._log.add_histogram(f"weight_{_name}", _layer.weight.data, _index)
                        self._log.add_histogram(f"grad_{_name}", _layer.weight.grad, _index)
                    # if _name.startwith("_tf_layer._out_norm._w") and isinstance(_layer, nn.Linear):
                    #     self.log.add_histogram(f"output_{_name}", _layer.)
            except Exception as e:
                print(e,"Error with weight extraction.")
            
            torch.save(self._net.state_dict(), "/root/project/My_projects/CNN/weights/conv_weight.pt")
            
            self._net.eval()
            _accsum = 0
            for _index, (_data, _label) in enumerate(self._test_dataset):
                _x = _data.cuda()
                _t = _label.cuda()
                _y = self._net(_x)
                
                _accsum += (_y.argmax(-1) == _t).sum()
                
            self._log.add_scalar("accuracy", _accsum/len(self._test_dataset), _epoch)
    
if __name__ == "__main__":
    train = Train()
    train()
