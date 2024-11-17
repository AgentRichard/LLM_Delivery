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

class MoE(nn.Module):
    def __init__(self, vinput, hidden, 
                 q_heads, kv_heads, voc_size, 
                 cache_max_batch_size, cahce_max_seq_size,
                 Rope_Length, layers_num):
        super().__init__()
        
        self._emb = nn.Embedding(voc_size, vinput)
        self._tfmr = Decoder(vinput, hidden, q_heads, kv_heads,
                             cache_max_batch_size, cahce_max_seq_size,
                             Rope_Length, layers_num)

        self._cache_max_batch_size = cache_max_batch_size
        self.apply(self._init_weight)
    
    def _init_weight(self, model):
        if isinstance(model, nn.Linear):
            nn.init.xavier_uniform_(model.weight)
            if model.bias is not None:
                nn.init.zeros_(model.bias)
        if isinstance(model, nn.Embedding):
            nn.init.xavier_uniform_(model.weight)
    
    def _forward(self, x, startpos):
        token = self._emb(x)
        token = token.to(dtype=torch.bfloat16)
        features = self._tfmr(token, startpos)
        return features @ self._emb.weight.T
    
    def forward(self, x, startpos = 0):
        if self._cache_max_batch_size is None:
            return self._forward(x, startpos)
        else:
            with torch.no_grad():
                return self._forward(x, startpos)
    
class Decoder(nn.Module):
    def __init__(self, vinput, hidden,
                q_heads, kv_heads, 
                cache_max_batch_size, cahce_max_seq_size,
                Rope_Length, layers_num):
        super().__init__()

        self._tf_layers = nn.ModuleList(
            [Decoder_layer(vinput, hidden, q_heads, kv_heads,
                cache_max_batch_size, cahce_max_seq_size) for _ in range(layers_num)]
        )
        
        self._rms = RMS(vinput)
        self._freqs_cis = precompute_freqs_cis(vinput//q_heads, Rope_Length)
        
        self.register_buffer("freqs_cis", self._freqs_cis, persistent=False)
        
    def forward(self, x, startpos):
        for layer in self._tf_layers:
            x = layer(x, self.freqs_cis, startpos)
        return self._rms(x)
    
class Decoder_layer(nn.Module):
    def __init__(self, vinput, hidden, q_heads, kv_heads,
                 cache_max_batch_size, cache_max_seq_size):
        super().__init__()
        
        self._attnrms = RMS(vinput)
        self._attn = Attention(vinput, q_heads, kv_heads, 
                    cache_max_batch_size, cache_max_seq_size)
        self._expert_rms = RMS(vinput)
        self._experts = SoftExperts(vinput, hidden, topk=2, num_experts=5)
    
    def forward(self, x, freqs_cis, startpos):
        _x = x
        _x = self._attnrms(_x)
        _x = self._attn(_x, freqs_cis, startpos)
        _x = _x + x
        
        _h = _x
        _h = self._expert_rms(_h)
        _h = self._experts(_h)
        return _h + _x
        
class Attention(nn.Module):
    def __init__(self, vinput, q_heads, kv_heads, 
                 cache_max_batch_size, cache_max_seq_size):
        super().__init__()
        
        self._qheads = q_heads
        self._scale = q_heads//kv_heads
        self._kvheads = kv_heads
        self._cache_batch = cache_max_batch_size
        self._cache_seq = cache_max_seq_size
        self._head_size = vinput // q_heads
        
        self._ql = nn.Linear(vinput, self._qheads * self._head_size)
        self._kl = nn.Linear(vinput, self._kvheads * self._head_size)
        self._vl = nn.Linear(vinput, self._kvheads * self._head_size)
        self._ol = nn.Linear(vinput, vinput)
        
        if self._cache_batch is not None:
            _cachek = torch.zeros(
                self._cache_batch,
                self._cache_seq,
                self._kvheads,
                self._head_size
            )
            
            _cachev = torch.zeros(
                self._cache_batch,
                self._cache_seq,
                self._kvheads,
                self._head_size
            )
            
            self.register_buffer("cachek", _cachek, persistent=False)
            self.register_buffer("cachev", _cachev, persistent=False)
    
    def forward(self, x, freqs_cis, startpos):
        _batch, _seq, _ = x.shape

        _q, _k, _v = self._ql(x), self._kl(x), self._vl(x)
        _q = _q.reshape(_batch, _seq, self._qheads, self._head_size)
        _k = _k.reshape(_batch, _seq, self._kvheads, self._head_size)
        _v = _v.reshape(_batch, _seq, self._kvheads, self._head_size)
        
        _q = apply_rotary_emb(_q, freqs_cis[startpos:_seq+startpos])
        _k = apply_rotary_emb(_k, freqs_cis[startpos:_seq+startpos])
        
        if self._cache_batch is not None:
            self.cachek[:_batch, startpos:_seq+startpos] = _k
            self.cachev[:_batch, startpos:_seq+startpos] = _v
            
            _k = self.cachek[:_batch, :_seq+startpos]
            _v = self.cachev[:_batch, :_seq+startpos]
            
        _q = _q.permute(0, 2, 1, 3)
        _k = _k.permute(0, 2, 1, 3)
        _v = _v.permute(0, 2, 1, 3)
        
        _k = _k[:,:,None].repeat(1, 1, self._scale, 1, 1).reshape(_batch, -1, startpos+_seq, self._head_size)
        _v = _v[:,:,None].repeat(1, 1, self._scale, 1, 1).reshape(_batch, -1, startpos+_seq, self._head_size)
        
        if startpos == 0:
            _score = F.scaled_dot_product_attention(_q, _k, _v,
                    attn_mask=None, is_causal=True)   ### is_causual = True 强覆盖attnmask, attnmask是自己写的注意力掩码
        else:
            _score = F.scaled_dot_product_attention(_q, _k, _v,
                    attn_mask=None, is_causal=False)
        # _q, _k, _v = _q.bfloat16(), _k.bfloat16(), _v.bfloat16()
        # if startpos == 0:
        #     _score = flash_attn_func(_q, _k, _v, causal=True)   ### is_causual = True 强覆盖attnmask, attnmask是自己写的注意力掩码
        # else:
        #     _score = flash_attn_func(_q, _k, _v, causal=False)
        # _score = _score.float()
        _output = _score.permute(0, 2, 1, 3).reshape(_batch, _seq, -1)
        return self._ol(_output)    
    
class SoftExperts(nn.Module):
    def __init__(self, vinput, hidden, topk, num_experts):
        super().__init__()
        
        self._topk = topk
        self._experts = nn.ModuleList(
            [MLP(vinput, hidden) for _ in range(num_experts)]
        )
        self._gate = nn.Linear(vinput, num_experts, bias=False)
        
    def forward(self, x):
        _x = x.reshape(-1, x.shape[-1])  ## [12, 256]  先转化为NV结构
        _gate_logits = self._gate(_x)    ## [12, 5]    转化成5个专家的概率分布 
        weights, selected_experts_index = torch.topk(_gate_logits, self._topk)  ## [12, 5] 2
        weights = torch.softmax(weights, dim=-1)                                ## weight,index =  [12, 2]
        result = torch.zeros_like(_x)    ## [12, 256]
        
        for _i, _selected_expert in enumerate(self._experts):                   ## 遍历专家列表
            # print(selected_experts_index == _i)
            batch_idx, nth_expert = torch.where(selected_experts_index == _i)   ## 从[12, 2] 里面选择某个专家负责的样本
            # print(batch_idx, nth_expert, _i)                                  ## 12个样本里有哪个里面有True， 选择的专家是哪一个
            _v = weights[batch_idx, nth_expert, None] * _selected_expert(_x[batch_idx])
            result[batch_idx] += _v
            
        result = result.reshape(*x.shape)
        return result
    
class MLP(nn.Module):
    """
    MoE软路由，连续，计算加权平均因此计算量增大，适合实验性研究，可以同时选择我们所规定数量专家来进行数据处理，对多个专家进行概率分布计算
    """
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
        "q_heads": 16,
        "kv_heads": 2,
        "voc_size": 30000,
        "cache_max_batch_size": None,
        "cahce_max_seq_size": None,
        "Rope_Length": 5000,
        "layers_num" : 3
    }
    moe = MoE(**config).cuda()
    data = torch.randint(0, 30000, (2, 6)).cuda()
    label = torch.randint(0, 30000, (2, 6)).reshape(-1).cuda()
    y = moe(data).reshape(-1, 30000)
    # print(y.shape)
    # lossfn = nn.CrossEntropyLoss()
    # loss = lossfn(y, label)
    # loss.backward()
    # for k, v in moe.named_modules():
    #     if k.startswith("_tfmr._tf_layers.0.") and isinstance(v, nn.Linear):
    #         # print(v)
    #         print(k)
    #         print(v.weight.grad)
    #         # print(v.weight.data.shape)
    #         # print(v.weight.grad)
    #         # print(v.cuda())
    #         # print(v.in_features)
    #         # print(v.requires_grad_())
            
        

            