import torch
import torch.nn as nn
import sentencepiece as sp
from model import MoE

class Inference:
    def __init__(self, voc_dir, weight_dir, topk=40, temp=0.8, **config):
        self._spm = sp.SentencePieceProcessor()
        self._spm.Load(voc_dir)
        
        self._topk = topk
        self._temp = temp
        
        self._net = MoE(**config).cuda()
        
        self._net.load_state_dict(torch.load(weight_dir)["module"])
        
    @staticmethod
    def _tsoftmax(xs, temp):
        return torch.exp(torch.log(xs/temp)) / (torch.exp(torch.log(xs/temp)).sum(dim=-1) + 1e-4)
    
    def _forward(self, id, startpos):
        _output = self._net(id, startpos)
        _output = _output[:, -1]
        _weight, _indices = torch.topk(_output, k=self._topk, dim=-1)
        _probs = self._tsoftmax(_weight, self._temp)
        _index = torch.multinomial(_probs, 1)
        _id = torch.gather(_indices, dim=-1, index=_index)
        return _id, self._spm.Decode(_id.item())
    
    def __call__(self, prompt, startpos=0):
        _ids = self._spm.Encode(prompt)
        print(_ids)
        _ids = torch.tensor(_ids, dtype=torch.long)[None].cuda()
        _vocs = prompt
        _id, _voc = self._forward(_ids, 0)
        _vocs += _voc
        startpos = _id.shape[1]
        
        for _ in range(20):
            _id, _voc = self._forward(_id, startpos)
            startpos += 1
            _vocs += _voc
        return _vocs
    
if __name__ == "__main__":
    config = {
    "vinput": 256,
    "hidden": 1024,
    "q_heads": 16,
    "kv_heads": 2,
    "voc_size": 30000,
    "cache_max_batch_size": 1,
    "cahce_max_seq_size": 200,
    "Rope_Length": 5000,
    "layers_num" : 5
    }   
    eval = Inference("/root/project/My_projects/Transformer/GQA_with_MOE/tokenizer.model",
                     "/root/project/My_projects/Transformer/GQA_with_MOE/Weights/moe_0/mp_rank_00_model_states.pt", **config)
    print(eval("中国"))
        
    
        
        
        