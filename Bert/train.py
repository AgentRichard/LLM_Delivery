import torch.nn as nn
import torch
from model import Bert
from data import Data
from torch.utils.data import DataLoader
from torch.optim.adam import Adam
from torch.utils.tensorboard import SummaryWriter

config = {
    "vinput": 256,
    "hidden": 720,
    "qheads_num": 16,
    "kvheads_num": 2,
    "voc_size": 30000,
    "rope_length": 5000,
    "cache_max_batch_size": None,
    "cache_max_seq_len": None,
    "layers_num" : 3
}

class Train:
    def __init__(self, data_dir):
        self._net = Bert(**config).cuda()
        
        self._dataset = Data(data_dir, 200)
        self._data = DataLoader(self._dataset, batch_size=20, shuffle=True, num_workers=16)
        self._opt = Adam(self._net.parameters(), lr=0.001, weight_decay=0.1, betas=(0.85, 0.95))
    
        self._loss = nn.CrossEntropyLoss(ignore_index=0)
        self._log = SummaryWriter(log_dir="runs")
        
    def __call__(self):
        for _epoch in range(1):
            
            self._net.train()
            loss_sum = 0
            for _i, (_prompt, _label) in enumerate(self._data):
                _prompt = _prompt.to(device="cuda", dtype=torch.long)
                _label = _label.to(device="cuda", dtype=torch.long)
                
                _output = self._net(_prompt)
                _output = _output.reshape(-1, 30000)
                _label = _label.reshape(-1)
                
                _loss = self._loss(_output, _label)
                self._opt.zero_grad()
                _loss.backward()
                self._opt.step()
                print(_output[0].argmax(-1), _label[0])
                loss_sum += _loss.detach()
                if _i % 5 == 0:
                    self._log.add_scalar("Loss", _loss.detach(), _i)  
            print("sumLoss", loss_sum/len(self._data))
            torch.save(self._net.state_dict(), "/root/project/My_projects/Transformer/Bert/weights/weight.pt")

if __name__ == "__main__":
    train = Train("/root/project/My_projects/Transformer/Bert/datas/data/train")
    train()                    