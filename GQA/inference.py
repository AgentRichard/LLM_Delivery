import torch
import torch.nn as nn
import sentencepiece as sp
from model.SkyFall import SkyFall

class Eval:
    def __init__(self, vocmodel_dir, weight_dir, topk = 4, temp = 1):
        self._spm = sp.SentencePieceProcessor()
        self._spm.Load(vocmodel_dir)
        
        self._topk = topk
        self._temp = temp
        
        self._net = SkyFall(
            num_layers = 10,
            vinput = 512, 
            hidden = 320,
            qheads_num = 8,
            kvheads_num = 4,
            RoPE_Len = 160,
            voc_Len = 30000,
            cache_maxBS = 200,
            cache_maxSeqL = 160
        ).cuda()
        self._net.load_state_dict(torch.load(weight_dir)["module"])
        
    @staticmethod
    def _tsoftmax(xs, temp):
        _xs = xs
        return torch.exp(torch.log(_xs/temp)) / (torch.exp(torch.log(_xs/temp)).sum(-1) + 1e-3)
    
    def _forward(self, id, start_pos):
        _output = self._net(id, start_pos)
        _output = _output[:, -1]
        _weight, _indices = torch.topk(_output, k = self._topk, dim = -1)
        _probs = self._tsoftmax(_weight, self._temp)
        _index = torch.multinomial(_probs, 1)
        _id = torch.gather(_indices, dim = -1, index = _index)
        return _id, self._spm.Decode(_id.item())
    
    def __call__(self, prompt, start_pos = 0):
        _ids = self._spm.Encode(prompt)
        _ids = torch.tensor(_ids, dtype=torch.long)[None].cuda()
        _vocs = prompt
        _id, _voc = self._forward(_ids, 0)
        _vocs += _voc
        start_pos = _id.shape[1]
        
        for _ in range(20):
            _id, _voc = self._forward(_id, start_pos)
            start_pos += 1
            _vocs += _voc
        return _vocs
    
if __name__ == "__main__":
    eval = Eval("SingleCard_GQA_without_MOE/tokenizer.model", "SingleCard_GQA_without_MOE/Weights/mytsfmr_0/mp_rank_00_model_states.pt")
    print(eval("我爱北京"))