from data import Data
import torch.nn as nn
import deepspeed
import argparse
import torch
from model import RNNModel
from torch.utils.tensorboard import SummaryWriter

class PreTrain:
    def __init__(self):
        deepspeed.init_distributed()
        self._args = ArgumentParse()
        self._data_file = self._args.data_file
        
        self._net = RNNModel(256, 64, 6)
        
        self._engine, self._opt, self._dataloader, self._lrscheduler = deepspeed.initialize(
            args = self._args,
            model = self._net,
            model_parameters = self._net.parameters(),
            training_data = Data(self._data_file, 64),
            config = "./deepspeed_config.json"
        )
        
        self._lossfn = nn.CrossEntropyLoss(ignore_index=0)
        self._rank = deepspeed.comm.get_rank()
        if self._rank == 0:
            self._log = SummaryWriter("Tensorboard_res")
    
    def __call__(self):
        self._engine.train()
        
        _, _client_state = self._engine.load_checkpoint("Weights")
        if _client_state is None:
            _client_state = {"step" : 0}
            
        for _i, _ids in enumerate(self._dataloader):
            _ids = _ids.cuda()
            _xs = _ids[:, :-1]
            _ys = _ids[:, 1: ]
            _output = self._net(_xs)[0]
            
            _output = _output.reshape(-1, 30000)
            _output = _output - _output.mean(-1, keepdim=True)
            _ys = _ys.reshape(-1)
            _loss = self._lossfn(_output, _ys)
            torch.autograd.set_detect_anomaly(True)
            self._engine.backward(_loss)
            self._engine.step()
            
            _step = _client_state["step"]
            _client_state["step"] += 1
            if self._rank == 0:
                self._log.add_scalar(f"loss", _loss, _step)
                try:
                    for _name, _layer in self._net.named_modules():
                        if _name.startswith("_net._layers.0._xlayer") and isinstance(_layer, nn.Linear):
                            self._log.add_histogram(f"weight_{_name}", _layer.weight.data, _step)
                            self._log.add_histogram(f"grad_{_name}", _layer.weight.grad, _step)
                        # if _name.startwith("_tf_layer._out_norm._w") and isinstance(_layer, nn.Linear):
                        #     self.log.add_histogram(f"output_{_name}", _layer.)
                except Exception as e:
                    print(e,"Error with weight extraction.")
                    
        _ss = self._args.ss
        self._engine.save_checkpoint(
            save_dir = "Weights", tag = f"mytsfmr_{_ss}",
            client_state = {"step" : _client_state["step"]}
        )
    
def ArgumentParse():
    parser = argparse.ArgumentParser(description="My_transformer")
    # 指那张显卡,deepspeed将模型复制多份（几卡几份）
    # 数据拆成卡份，然后跑完了数据进行汇总
    # rank相当于看在哪张卡上
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--data_file", type=str)
    parser.add_argument("--ss", type=str)
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    pretrain = PreTrain()
    pretrain()        