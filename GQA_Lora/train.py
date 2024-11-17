import deepspeed.comm
from model import GQA
from data import Data
import torch
import deepspeed
import torch.nn as nn
import argparse
from torch.utils.tensorboard import SummaryWriter

def ArgumentParse():
    parser = argparse.ArgumentParser(description="Lora")
    parser.add_argument("--data_file", type=str)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--ss", type=str)
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args

config = {
    "vinput" : 480,
    "hidden" : 1024,
    "qheads" : 12,
    "kvheads": 2,
    "rope_len" : 5000,
    "voc_size" : 30000,
    "cache_max_batch" : None,
    "cache_max_seq" : None,
    "num_layers" : 5
}

class Train:
    def __init__(self):
        deepspeed.init_distributed()
        self._args = ArgumentParse()
        
        self._net = GQA(**config)
        
        self._engine, self._opt, self._dataloader, self._lrscheduler = deepspeed.initialize(
            args = self._args,
            model = self._net,
            model_parameters = self._net.parameters(),
            training_data = Data(self._args.data_file, 300),
            config = "./deepspeed_config.json"
        )
        
        self._lossfn = nn.CrossEntropyLoss(ignore_index=0)
        self._rank = deepspeed.comm.get_rank()
        if self._rank == 0:
            self._log = SummaryWriter(log_dir="./runs")
    
    def __call__(self):
        
        self._engine.train()
        
        _, _client_states = self._engine.load_checkpoint("/root/project/My_projects/Transformer/GQA_PEFT/weights/weight.pt")
        if _client_states is None:
            _client_states = {"step":0}
            
        for _index, (_prompt, _label) in enumerate(self._dataloader):
            _prompt = _prompt.to(device = self._engine.device, dtype = torch.long)
            _label = _label.to(device = self._engine.device, dtype = torch.long)
            
            _prompt = _prompt[:, :-1]
            _label = _label[:, 1: ]
            _output = self._net(_prompt)
            
            _output = _output.reshape(-1, 30000)
            _label = _label.reshape(-1)
            
            _loss = self._lossfn(_output, _label)
            self._engine.backward(_loss)
            self._engine.step()
            
            _step = _client_states["step"]
            if self._rank == 0 and _index % 10 == 0:
                self._log.add_scalar("Loss", _loss.detach().item(), _step)
                try:
                    for k, v in self._net.named_modules():
                        if k == "_encoder._layers.0._attn._ol":
                            self._log.add_histogram("ol_weight", v.weight.data, global_step=_step)
                            self._log.add_histogram("ol_grad", v.weight.grad, global_step=_step)
                except Exception as e:
                    print(e) 
            _client_states["step"] += 1

        
        _ss = self._args.ss
        self._engine.save_checkpoint(
            save_dir=f"/root/project/My_projects/Transformer/GQA_SFT/weights/weight.pt", tag=f"lora_{_ss}",
            client_state = {"step":_client_states["step"]}
        )
            
if __name__ == "__main__":
    train = Train()
    train()
            