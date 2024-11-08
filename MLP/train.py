import torch
from data_load import Data
from FCNN_sequential import FCNN
from torch.utils.data import DataLoader
from torch.optim.adam import Adam
from torch.utils.tensorboard import SummaryWriter

class Train:
    
    def __init__(self):
        self._net = FCNN().cuda()
        
        self._train_dataset = Data("/root/project/My_projects/FCNN/data/train")
        self._test_dataset = Data("/root/project/My_projects/FCNN/data/test")
        self._train_data = DataLoader(self._train_dataset, batch_size=5000, shuffle=True)
        self._test_data = DataLoader(self._test_dataset, batch_size=5000, shuffle=True)
        
        self._opt = Adam(self._net.parameters())
        self._loss_fn = torch.nn.SmoothL1Loss()
        
        self._log = SummaryWriter("/root/project/My_projects/FCNN/runs")
    
    def __call__(self):

        for _epoch in range(1000000000):
            self._net.train()
            _lossSum = 0
            self._net.load_state_dict(torch.load("/root/project/My_projects/FCNN/pretrain_checkpoints/weight.pt"))
            for _index, (_x, _label) in enumerate(self._train_data):
                _x = _x.cuda()
                _label = _label.cuda()
                _output = self._net(_x)
                _loss = self._loss_fn(_output, _label)
                
                self._opt.zero_grad()
                _loss.backward()
                self._opt.step()
                _lossSum += _loss.detach().cpu().item()
            
            print("loss", _lossSum / len(self._train_data)*1000)
            self._log.add_scalar("Loss", _lossSum/len(self._train_data)*1000, _epoch)
            torch.save(self._net.state_dict(), "/root/project/My_projects/FCNN/pretrain_checkpoints/weight.pt")
            
            self._net.eval()
            _accsum = 0
            for _index, (_data, _label) in enumerate(self._test_data):
                _x = _data.cuda()
                _t = _label.cuda()
                _y = self._net(_x)
                
                _accsum += (_y.argmax(-1) == _t).sum()
                
            self._log.add_scalar("accuracy", _accsum/len(self._test_data) / 100, _epoch)
    
if __name__ == "__main__":
    train = Train()
    train()