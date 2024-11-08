import deepspeed.comm
from data import ModelDataSet
import torch.nn as nn
import deepspeed
import argparse
from model.SkyFall import SkyFall
from sft_data import SFTDataSet

from torch.utils.tensorboard import SummaryWriter

class PreTrain:
    def __init__(self):
        deepspeed.init_distributed()
        self._args = ArgumentParse()
        self._data_file = self._args.data_file
        
        self._net = SkyFall(
            num_layers = 10,
            vinput = 512,
            hidden = 320,
            qheads_num = 8,
            kvheads_num = 4,
            RoPE_Len = 20,
            voc_Len = 30000
        )
        
        self._engine, self._opt, self._dataloader, self._lrscheduler = deepspeed.initialize(
            args = self._args,
            model = self._net,
            model_parameters = self._net.parameters(),
            training_data = SFTDataSet(self._data_file, 20),
            config = "./deepspeed_config.json"
        )
        
        self._lossfn = nn.CrossEntropyLoss(ignore_index = 0)
        self._rank = deepspeed.comm.get_rank()
        if self._rank == 0:
            self._log = SummaryWriter("Tensorboard_res")
    
    def __call__(self):
        self._engine.train()
        
        _, _client_state = self._engine.load_checkpoint("Weights")
        if _client_state is None:
            _client_state = {"step" : 0}
            
        for _i, (_prompt, _tag) in enumerate(self._dataloader):
            _xs = _prompt[:, :-1].to(device = self._engine.device)
            _ys = _tag[:, 1: ].to(device = self._engine.device)
            _output = self._net(_xs)
            
            _output = _output.reshape(-1, 30000)
            _output = _output - _output.mean(-1, keepdim=True)
            
            _ys = _ys.reshape(-1)
            _loss = self._lossfn(_output, _ys)
            self._engine.backward(_loss)
            self._engine.step()
            
            _step = _client_state["step"]
            if self._rank == 0 and _i % 10 == 0:
                self._log.add_scalar("Loss", _loss, _step)
                _client_state["step"] += 1
        
        _ss = self._args.ss
        self._engine.save_checkpoint(
            save_dir = "Weights", tag = f"mytsfmr_{_ss}",
            client_state = {"step" : _client_state["step"]}
        )
    
def ArgumentParse():
    parser = argparse.ArgumentParser(description="My_transformer")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--data_file", type=str)
    parser.add_argument("--ss", type=str)
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    pretrain = PreTrain()
    pretrain()        