import torch
from torch.utils.data import DataLoader
from data_load import Data
from model import VIT
import torch.nn as nn
from torch.optim.adam import Adam
import warnings
from torch.utils.tensorboard import SummaryWriter
warnings.filterwarnings("ignore")

class Pretrain:
    def __init__(self, train_dir, eval_dir, log_dir):
        
        self._net = VIT().cuda()
        
        self._train_dataset = Data(train_dir)
        self._train_data = DataLoader(self._train_dataset, batch_size=500, shuffle=True, num_workers=20)
        self._test_dataset = Data(eval_dir)
        self._test_data = DataLoader(self._test_dataset, batch_size=3000, shuffle=True, num_workers=20)
        
        self._opt = Adam(self._net.parameters(), lr=0.0001, betas=[0.85, 0.95], weight_decay=0.001)
        self._log = SummaryWriter(log_dir=log_dir)
        self._loss_fn = nn.CrossEntropyLoss()
        
    def __call__(self):
        
        self._net.load_state_dict(torch.load("/root/project/My_projects/Transformer/VisionTransformer/weights/weight.pt"))
        
        for _epoch in range(1000000000):
            self._net.train()
            
            print(f"开始第{_epoch}轮训练")
            _loss_sum = 0
            for _index, (_x, _labels) in enumerate(self._train_data):
                _x = _x.to(torch.float32).cuda()
                _labels = _labels.to(torch.long).cuda()
                _logits = self._net(_x)
                _loss = self._loss_fn(_logits, _labels)
                self._opt.zero_grad()
                _loss.backward()
                self._opt.step()
                _loss_sum += _loss.detach().item()
                print("Loss", _loss_sum, _epoch)
            self._log.add_scalar("train_loss", _loss_sum, _epoch)
            try:
                for _name, _layer in self._net.named_modules():
                    if _name.startswith("_tf_layer._layers.0") and isinstance(_layer, nn.Linear):
                        self._log.add_histogram(f"weight_{_name}", _layer.weight.data, _epoch)
                        self._log.add_histogram(f"grad_{_name}", _layer.weight.grad, _epoch)
                    # if _name.startwith("_tf_layer._out_norm._w") and isinstance(_layer, nn.Linear):
                    #     self.log.add_histogram(f"output_{_name}", _layer.)
            except Exception as e:
                print(e,"Error with weight extraction.")
            
            torch.save(self._net.state_dict(), "/root/project/My_projects/Transformer/VisionTransformer/weights/weight.pt")
                
            self._net.eval()
            _acc_sum = 0
            print(f"开始第{_epoch}轮测试")
            for _index, (_x, _label) in enumerate(self._test_data):
                _x = _x.to(torch.float32).cuda()
                _label = _label.to(torch.float32).cuda()
                _output = self._net(_x)
                _acc_sum += torch.sum((_output.argmax(-1) == _label))
                print(_output.argmax(-1)[0], _label[0])
            self._log.add_scalar("acc_sum", _acc_sum/len(self._test_data), _epoch) 
    
if __name__ == "__main__":
    train = Pretrain("/root/project/My_projects/Transformer/VisionTransformer/data/train",
                     "/root/project/My_projects/Transformer/VisionTransformer/data/test",
                     "/root/project/My_projects/Transformer/VisionTransformer/runs")
    train()
    