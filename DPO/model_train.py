import torch
import torch.nn as nn
import torch.nn.functional as F
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

class Lora_layer(nn.Module):
    def __init__(self, input_features:int, output_features:int, alpha:int = 32, rank:int = 20,
                 device=None, bias=None, dtype=None):
        super().__init__()
        
        self._linear = nn.Linear(input_features, output_features, bias)
        self._lora_A = nn.Linear(input_features, rank, bias=True)
        self._lora_B = nn.Linear(rank, output_features, bias=True)
        
        self._rank = rank
        self._alpha = alpha
        self._drop = nn.Dropout(0.5)
        
        nn.init.xavier_normal_(self._linear.weight)
        if bias is not None:
            nn.init.zeros_(self._linear.weight)
        nn.init.xavier_normal_(self._lora_A.weight)
        if self._lora_A.bias is not None:
            nn.init.zeros_(self._lora_A.bias)
        nn.init.zeros_(self._lora_B.weight)
        if self._lora_B.bias is not None:
            nn.init.zeros_(self._lora_B.bias)

    def forward(self, xs):
        with torch.no_grad():
            _h = self._linear(xs)
        _xs = self._lora_A(xs)
        _xs = self._lora_B(self._drop(_xs))
        return _h.detach() + (1/self._alpha) * _xs

class GQA(nn.Module):
    def __init__(self, vinput, hidden, qheads,
                 kvheads, rope_len, voc_size,
                 cache_max_batch, cache_max_seq, num_layers):
        super().__init__()
        
        self._emb = nn.Embedding(voc_size, vinput)
        
        self._encoder = EncoderLayers(vinput, hidden, qheads, kvheads,
                                      rope_len, cache_max_batch, cache_max_seq, num_layers)
        self._cach_max_batch = cache_max_batch
        self.apply(self._init_weight)
        
    def _init_weight(self, model):
        if isinstance(model, nn.Linear):
            model.reset_parameters()
        if isinstance(model, nn.Embedding):
            model.reset_parameters()
    
    def _forward(self, xs, startpos):
        _token = self._emb(xs)
        _features = self._encoder(_token, startpos)
        return _features @ self._emb.weight.T
    
    def forward(self, xs, startpos=0):
        if self._cach_max_batch is None:
            return self._forward(xs, startpos)
        else:
            with torch.no_grad():
                return self._forward(xs, startpos)
    
class EncoderLayers(nn.Module):
    def __init__(self, vinput, hidden, qheads,
                 kvheads, rope_len, 
                 cache_max_batch, cache_max_seq, num_layers):
        super().__init__()
        
        self._layers = nn.ModuleList(
            [Encoder(vinput, hidden, qheads, kvheads,
                     cache_max_batch, cache_max_seq) for _ in range(num_layers)]
        )
        
        self._rms = RMS(vinput)
        self._freqs_cis = precompute_freqs_cis(vinput//qheads, rope_len)
        self.register_buffer("freqs_cis", self._freqs_cis, persistent=False)
        
    def forward(self, xs, startpos):
        for layer in self._layers:
            xs = layer(xs, startpos, self.freqs_cis)
        return self._rms(xs)
    
class Encoder(nn.Module):
    def __init__(self, vinput, hidden, qheads,
                 kvheads, cache_max_batch, cache_max_seq):
        super().__init__()
        
        self._att_rms = RMS(vinput)
        self._attn = Attention(vinput, qheads, kvheads, cache_max_batch, cache_max_seq)
        self._mlp_rms = RMS(vinput)
        self._mlp = MLP(vinput, hidden)
         
    def forward(self, xs, startpos, freqs_cis):
        _xs = xs
        _xs = self._att_rms(_xs)
        _xs = self._attn(_xs, startpos, freqs_cis)
        _xs = _xs + xs
        
        _h = _xs
        _h = self._mlp_rms(_h)
        _h = self._mlp(_h)
        return _h + _xs

class Attention(nn.Module):
    def __init__(self, vinput,  qheads,
                 kvheads, cache_max_batch, cache_max_seq):
        super().__init__()
        
        self._qheads = qheads
        self._kvheads= kvheads
        self._scale = self._qheads // self._kvheads
        self._cache_batch = cache_max_batch
        self._headsize = vinput // self._qheads
        
        self._ql = nn.Linear(vinput, vinput)
        self._kl = nn.Linear(vinput, self._kvheads * self._headsize)
        self._vl = nn.Linear(vinput, self._kvheads * self._headsize)
        self._ol = nn.Linear(vinput, vinput)
        
        if self._cache_batch is not None:
            _cachek = torch.zeros(
                cache_max_batch,
                cache_max_seq,
                kvheads,
                self._headsize
            )
            
            _cachev = torch.zeros(
                cache_max_batch,
                cache_max_seq,
                kvheads,
                self._headsize
            )
            
            self.register_buffer("cachek", _cachek, persistent=False)
            self.register_buffer("cachev", _cachev, persistent=False)
        
    def forward(self, xs, startpos, freqs_cis):
        _batch, _seq, _ = xs.shape
        
        _q, _k, _v = self._ql(xs), self._kl(xs), self._vl(xs)

        _q = _q.reshape(_batch, _seq, self._qheads, self._headsize)
        _k = _k.reshape(_batch, _seq, self._kvheads, self._headsize)
        _v = _v.reshape(_batch, _seq, self._kvheads, self._headsize)
        
        _q = apply_rotary_emb(_q, freqs_cis[startpos:startpos+_seq])
        _k = apply_rotary_emb(_k, freqs_cis[startpos:startpos+_seq])
        
        if self._cache_batch is not None:
            self.cachek[:_batch, startpos:startpos+_seq] = _k
            self.cachev[:_batch, startpos:startpos+_seq] = _v
            _k = self.cachek[:_batch, :startpos+_seq]
            _v = self.cachev[:_batch, :startpos+_seq]
        
        _q = _q.permute(0, 2, 1, 3)
        _k = _k.permute(0, 2, 1, 3)
        _v = _v.permute(0, 2, 1, 3)
        
        _k = _k[:,:,None].repeat(1, 1, self._scale, 1, 1).reshape(_batch, -1, startpos+_seq, self._headsize)
        _v = _v[:,:,None].repeat(1, 1, self._scale, 1, 1).reshape(_batch, -1, startpos+_seq, self._headsize)
        
        if startpos == 0:
            _scores = F.scaled_dot_product_attention(_q, _k, _v,
                    attn_mask=None, is_causal=True)   ### is_causual = True 强覆盖attnmask, attnmask是自己写的注意力掩码
        else:
            _scores = F.scaled_dot_product_attention(_q, _k, _v,
                    attn_mask=None, is_causal=False)
            
        # _q, _k, _v = _q.bfloat16(), _k.bfloat16(), _v.bfloat16()
        # if startpos == 0:
        #     _scores = flash_attn_func(_q, _k, _v, dropout_p=0.3, causal=True)
        # else:
        #     _scores = flash_attn_func(_q, _k, _v, dropout_p=0.3, causal=False)
        
        # _scores = _scores.float()
        _output = _scores.permute(0, 2, 1, 3).reshape(_batch, _seq, -1)
        return self._ol(_output)

class MLP(nn.Module):
    def __init__(self, vinput, hidden):
        super().__init__()
        
        self._mlp1 = Lora_layer(vinput, hidden)
        self._mlp2 = Lora_layer(vinput, hidden)
        self._mlp3 = Lora_layer(hidden, vinput)
        self._gate = nn.SiLU()
        
    def forward(self, xs):
        return self._mlp3(self._mlp1(xs) * self._gate(self._mlp2(xs)))
    
class RMS(nn.Module):
    def __init__(self, vinput):
        super().__init__()
        
        self._learnable_scale = nn.Parameter(torch.randn(vinput))
        
    def forward(self, xs):
        return self._learnable_scale * xs / torch.sqrt(torch.pow(xs, 2).mean(dim=-1, keepdim=True) + 1e-5)
    
if __name__ == "__main__":
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
    gqa= GQA(**config).cuda()
    test = torch.randint(0,10,(2, 6)).cuda()
    print(gqa(test).shape)
    for k,v in gqa.named_modules():
        print(k,v)