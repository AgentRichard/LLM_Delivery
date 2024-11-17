import deepspeed.comm
import torch
from data import Data
import deepspeed
import argparse
from model import MoE
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

def ArgumentParse():
    parser = argparse.ArgumentParser(description="train_MoE")
    parser.add_argument("--data_file", type=str)
    parser.add_argument("--ss", type=str)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args

config = {
    "vinput": 256,
    "hidden": 1024,
    "q_heads": 16,
    "kv_heads": 2,
    "voc_size": 30000,
    "cache_max_batch_size": None,
    "cahce_max_seq_size": None,
    "Rope_Length": 5000,
    "layers_num" : 5
}
    
class Train:
    def __init__(self):
        deepspeed.init_distributed()
        self._args = ArgumentParse()
        self._net = MoE(**config)
        self._data_file = self._args.data_file
        
        self._engine, self._opt, self._dataloader, self._lr_scheduler = deepspeed.initialize(
            args = self._args,
            model = self._net,
            model_parameters = self._net.parameters(),
            training_data = Data(self._data_file, 300),
            config = "./deepspeed_config.json"
        )

        self._lossfn = nn.CrossEntropyLoss(ignore_index=0)
        
        self._rank = deepspeed.comm.get_rank()
        if self._rank == 0:
            self._log = SummaryWriter("/root/project/My_projects/Transformer/GQA_with_MOE/moe_runs")
        
    def __call__(self):
        self._engine.train()
        
        _, _client_state = self._engine.load_checkpoint("/root/project/My_projects/Transformer/GQA_with_MOE/Weights")
        if _client_state is None:
            _client_state = {"step":0}
            
        for _index, _ids in enumerate(self._dataloader):
            _ids = _ids.to(device = self._engine.device)
            xs = _ids[:, :-1]
            ys = _ids[:, 1: ]
            _output = self._net(xs)
            _output = _output.reshape(-1, 30000)
            # _output = _output - _output.mean(dim=-1, keepdim=True)
            ys = ys.reshape(-1)
            _loss = self._lossfn(_output, ys)
            self._engine.backward(_loss)
            self._engine.step()
            
            _step = _client_state["step"]
            if self._rank == 0 and _index % 10 == 0:
                self._log.add_scalar("Loss", _loss.item(), _step)
                
                try:
                    for _name, _layer in self._net.named_modules():
                        if _name.startswith("_tfmr._tf_layers.0._attn") and isinstance(_layer, nn.Linear):
                            self._log.add_histogram(f"weight_{_name}", _layer.weight.data, _step)
                            self._log.add_histogram(f"grad_{_name}", _layer.weight.grad, _step)
                        if _name.startswith("_tfmr._tf_layers.0._mlp._mlp3") and isinstance(_layer, nn.Linear):
                            self._log.add_histogram(f"weight_{_name}", _layer.weight.data, _step)
                            self._log.add_histogram(f"grad_{_name}", _layer.weight.grad, _step)
                            
                        # if _name.startwith("_tf_layer._out_norm._w") and isinstance(_layer, nn.Linear):
                        #     self.log.add_histogram(f"output_{_name}", _layer.)
                except Exception as e:
                    print(e)
                _client_state["step"] += 1
        
        _ss = self._args.ss
        self._engine.save_checkpoint(
            save_dir = "Weights", tag = f"moe_{_ss}",
            client_state = {"step" : _client_state["step"]}
        )
    
if __name__ == "__main__":
    train = Train()
    train()
    