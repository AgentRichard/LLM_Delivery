import torch
import torch.nn as nn
from flash_attn.flash_attn_interface import flash_attn_func

def precompute_freqs_cis(dim, end, theta=50000.0):      
    freqs = 1.0/(theta ** (torch.arange(0, dim, 2)[:(dim//2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis        

def apply_rotary_emb(xq, freqs_cis):   
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq)

def reshape_for_broadcast(freqs_cis, x):
    freqs_cises = freqs_cis[:x.shape[1]]
    return freqs_cises[None, :, None]

class Bert(nn.Module):
    def __init__(self, vinput, hidden, qheads_num,
                 kvheads_num, voc_size, rope_length,
                 cache_max_batch_size, cache_max_seq_len, layers_num):
        super().__init__()
        
        self._emb = nn.Embedding(voc_size, vinput)
        self._encoder = Encoder(vinput, hidden, qheads_num,
                                kvheads_num, rope_length,
                                cache_max_batch_size, cache_max_seq_len, layers_num)
        self._cache_batch_size = cache_max_batch_size
        # self._class_token = nn.Parameter(torch.randn(vinput)[None,None])
        
        self.apply(self._init_weight)
        
    def _init_weight(self, model):
        if isinstance(model, nn.Linear):
            model.reset_parameters()
            if model.bias is not None:
                nn.init.zeros_(model.bias)
        if isinstance(model, nn.Embedding):
            model.reset_parameters()
        
    def _forward(self, x, startpos):
        token = self._emb(x)
        # token = torch.concat((token, self._class_token.repeat(x.shape[0], 1, 1)), dim=1)  ### concate左右一个是添加到尾部，一个是首部
        features = self._encoder(token, startpos)
        output = features[:,1]  ## 因为preprocess已经处理过cls，因此这里直接输出第一个clstoken即可
        return output @ self._emb.weight.T
    
    def forward(self, x, startpos=0):
        if self._cache_batch_size is None:
            return self._forward(x, startpos)
        else:
            with torch.no_grad():
                return self._forward(x, startpos)
    
class Encoder(nn.Module):
    def __init__(self, vinput, hidden, qheads_num,
                 kvheads_num, rope_length,
                 cache_max_batch_size, cache_max_seq_len, layers_num):
        super().__init__()
        
        self._freqs_cis = precompute_freqs_cis(vinput//qheads_num, rope_length)
        self._layers = nn.ModuleList(
            [Encoder_layer(vinput, hidden, qheads_num, kvheads_num,
                           cache_max_batch_size, cache_max_seq_len) for _ in range(layers_num)]
        )
        
        self._rms = RMS(vinput)
        self.register_buffer("freqs_cis", self._freqs_cis, persistent=False)
        
    def forward(self, x, startpos):
        for _layer in self._layers:
            x = _layer(x, self.freqs_cis, startpos)
        return self._rms(x)
    
class Encoder_layer(nn.Module):
    def __init__(self, vinput, hidden, qheads_num,
                 kvheads_num, cache_max_batch_size, cache_max_seq_len):
        super().__init__()
        
        self._attn_rms = RMS(vinput)
        self._attn = Attention(vinput, qheads_num, kvheads_num,
                               cache_max_batch_size, cache_max_seq_len)
        self._mlp_rms = RMS(vinput)
        self._mlp = MLP(vinput, hidden)
        
    def forward(self, x, freqs_cis, startpos):
        _x = x
        _x = self._attn_rms(_x)
        _x = self._attn(_x, freqs_cis, startpos)
        _x = _x + x
        
        _h = _x
        _h = self._mlp_rms(_h)
        _h = self._mlp(_h)
        _h = _h + _x
        return _h
    
class Attention(nn.Module):
    def __init__(self, vinput, qheads_num, kvheads_num, 
                 cache_max_batch_size, cache_max_seq_len):
        super().__init__()
        
        self._qheads = qheads_num
        self._kvheads = kvheads_num
        self._scale = qheads_num // kvheads_num
        self._headsize = vinput // qheads_num
        self._cachek = cache_max_batch_size
        
        if cache_max_batch_size is not None:
            self.cache_k = torch.zeros(
                            cache_max_batch_size,
                            cache_max_seq_len,
                            kvheads_num,
                            self._headsize
                        )
            self.cache_v = torch.zeros(
                            cache_max_batch_size,
                            cache_max_seq_len,
                            kvheads_num,
                            self._headsize
                        )
            self.register_buffer("cachek", self.cache_k, persistent=False)
            self.register_buffer("cachev", self.cache_v, persistent=False)
            
        self._ql = nn.Linear(vinput, qheads_num * self._headsize)
        self._kl = nn.Linear(vinput, kvheads_num * self._headsize)
        self._vl = nn.Linear(vinput, kvheads_num * self._headsize)
        
        self._ol = nn.Linear(vinput, vinput)
        
    def forward(self, x, freqs_cis, startpos):
        _batch, _seq, _ = x.shape
        
        _q, _k, _v = self._ql(x), self._kl(x), self._vl(x)
        _q = _q.reshape(_batch, _seq, self._qheads, self._headsize)
        _k = _k.reshape(_batch, _seq, self._kvheads, self._headsize)
        _v = _v.reshape(_batch, _seq, self._kvheads, self._headsize)
        
        _q = apply_rotary_emb(_q, freqs_cis[startpos:startpos+_seq])
        _k = apply_rotary_emb(_k, freqs_cis[startpos:startpos+_seq])
        
        _q = _q.permute(0, 2, 1, 3)
        _k = _k.permute(0, 2, 1, 3)
        _v = _v.permute(0, 2, 1, 3)
        
        if self._cachek is not None:
            self.cachek[:_batch, startpos:startpos+_seq] = _k
            self.cachev[:_batch, startpos:startpos+_seq] = _v
            
            _k = self.cache_k[:_batch, :startpos+_seq]
            _v = self.cache_v[:_batch, :startpos+_seq]

        _k = _k[:,:,None].repeat(1, 1, self._scale, 1, 1).reshape(_batch, -1, _seq, self._headsize)
        _v = _v[:,:,None].repeat(1, 1, self._scale, 1, 1).reshape(_batch, -1, _seq, self._headsize)
        # print(_q.shape, _k.shape, _v.shape)
        _score = nn.functional.scaled_dot_product_attention(_q, _k, _v, 
                            attn_mask=None, dropout_p=0.3, is_causal=False)
        
        _output = _score.permute(0, 2, 1, 3).reshape(_batch, _seq, -1)
        return self._ol(_output)
        
class MLP(nn.Module):
    def __init__(self, vinput, hidden):
        super().__init__()
        
        self._mlp1 = nn.Linear(vinput, hidden)
        self._mlp2 = nn.Linear(vinput, hidden)
        self._mlp3 = nn.Linear(hidden, vinput)
        
        self._gate = nn.SiLU()
        
    def forward(self, x):
        return self._mlp3(self._mlp1(x) * self._gate(self._mlp2(x)))
    
class RMS(nn.Module):
    def __init__(self, vinput):
        super().__init__()
        
        self._learnable_scale = nn.Parameter(torch.randn(vinput))
        
    def forward(self, x):
        return self._learnable_scale * x / torch.sqrt(torch.pow(x, 2).mean(dim=-1, keepdim=True) + 1e-5)
    
if __name__ == "__main__":
    config = {
        "vinput": 256,
        "hidden": 1024,
        "qheads_num": 16,
        "kvheads_num": 2,
        "voc_size": 30000,
        "rope_length": 5000,
        "cache_max_batch_size": None,
        "cache_max_seq_len": None,
        "layers_num" : 3
    }
    bert = Bert(**config).cuda()
    data = torch.randint(0, 30000, (3, 6)).cuda()
    print(bert(data).shape)
    