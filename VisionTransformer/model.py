import torch
import torch.nn as nn
import torch.nn.functional as F

class VIT(nn.Module):
    def __init__(self, 
                layers_num = 5,
                input_dim = 768,
                hide_dim = 256,
                qheads_num = 12,
                kvheads_num = 2,
                max_pos_len = 16384):
        super().__init__()
        ### 数据进来是 NCHW, MINI-IMAGE 是 NHW
        self._patch_emb = nn.Linear(49, 768, bias=False)   ### 一般向量嵌入的时候把偏置关掉  
        # 也可用卷积
        self._encoder = TransformerEncoder(layers_num = layers_num, 
                                           input_dim = input_dim,
                                           hide_dim = hide_dim,
                                           qheads_num = qheads_num,
                                           kvheads_num = kvheads_num,
                                           max_pos_len = max_pos_len)
        self._out_put = nn.Linear(input_dim, 10, bias=False)
        
        self._class_token = nn.Parameter(torch.randn(768)[None,None])
        # self.register_buffer("class_token", _class_token, persistent=True)  不好，因为 clstoken需要被动态更新，而不是单纯保存在checkpoint中
        
        self.apply(self._init_weight)
        
    def forward(self, x):
        ### NHW 变成 N4h4w
        _bs, _ch, _h, _w = x.shape
        _x = x.reshape(_bs, _ch, 4, _h//4, 4, _w//4).permute(0, 2, 4, 1, 3, 5).reshape(_bs, 16, -1)
        _tokens = self._patch_emb(_x)
        _tokens = torch.concat((self._class_token.repeat(_bs, 1, 1), _tokens), dim=1)
        _features = self._encoder(_tokens)
        _feature = _features[:,0]
        return self._out_put(_feature)
    
    def _init_weight(self, model):
        if isinstance(model, nn.Linear):
            nn.init.xavier_normal_(model.weight)
            if model.bias is not None:
                nn.init.zeros_(model.bias)
        if isinstance(model, nn.Embedding):
            nn.init.xavier_normal_(model.weight)
    
class TransformerEncoder(nn.Module):
    def __init__(self, layers_num,
                 input_dim, hide_dim,
                 qheads_num, kvheads_num,
                 max_pos_len):
        super().__init__()
        
        self._layers = nn.ModuleList(
            TransformerLayer(input_dim = input_dim,
                            hide_dim = hide_dim,
                            qheads_num = qheads_num,
                            kvheads_num = kvheads_num) for _ in range(layers_num))
        self._layer_norm = LayerNormal(input_dim = input_dim)
        
        _freqs_cis = precompute_freqs_cis(input_dim // qheads_num, max_pos_len)  ## 不要忘记除以头数
        self.register_buffer("freqs_cis", _freqs_cis, persistent=False)
        
    def forward(self, x):
        _x = x
        for layer in self._layers:
            _x = layer(_x, self.freqs_cis)
        _x = self._layer_norm(_x)
        return _x
    
class TransformerLayer(nn.Module):
    def __init__(self, input_dim, 
                 hide_dim, qheads_num, 
                 kvheads_num):
        super().__init__()
        
        self._layer_norm = LayerNormal(input_dim)
        self._att_layer = Attention(input_dim=input_dim,
                                    qheads_num=qheads_num,
                                    kvheads_num=kvheads_num)
        self._mlp_layer = MLP(input_dim=input_dim, hide_dim=hide_dim)
        
    def forward(self, x, freqs_cis):
        _x = x
        _x = self._layer_norm(_x)
        _x = self._att_layer(_x, freqs_cis)
        _x = _x + x
        
        _h = _x
        _h = self._layer_norm(_h)
        _h = self._mlp_layer(_h)
        _h = _h + _x
        return _h

class Attention(nn.Module):
    def __init__(self, input_dim, 
                qheads_num, 
                kvheads_num):
        super().__init__()
        
        self._qheads_num = qheads_num
        self._kvheads_num = kvheads_num
        self._qkv_scale = self._qheads_num // self._kvheads_num
        self._head_size = input_dim // qheads_num
        self._dk = torch.sqrt(torch.tensor(self._head_size, dtype=torch.float32))
        
        self._ql = nn.Linear(input_dim, self._qheads_num * self._head_size)
        self._kl = nn.Linear(input_dim, self._kvheads_num * self._head_size)
        self._vl = nn.Linear(input_dim, self._kvheads_num * self._head_size)
        self._ow = nn.Linear(self._qheads_num * self._head_size, input_dim)
        
    def forward(self, x, freqs_cis):
        _bs, _seq, _ = x.shape
        _q, _k, _v = self._ql(x), self._kl(x), self._vl(x)
        _q = _q.reshape(_bs, _seq, self._qheads_num, self._head_size)
        _k = _k.reshape(_bs, _seq, self._kvheads_num, self._head_size)
        _v = _v.reshape(_bs, _seq, self._kvheads_num, self._head_size)

        _q = apply_rotary_emb(_q, freqs_cis[:_seq])
        _k = apply_rotary_emb(_k, freqs_cis[:_seq])
        
        _q = _q.permute(0, 2, 1, 3)
        _k = _k.permute(0, 2, 1, 3)
        _v = _v.permute(0, 2, 1, 3)
        
        # _causual = torch.ones(_seq, _seq)
        # _causual = torch.triu(_causual, diagonal=1)
        # _causual[_causual == 1] = -torch.inf
        # _causual = _causual.to(x.device)
        
        _k = _k[:,:,None].repeat(1,1,self._qkv_scale,1,1).reshape(_bs, -1, _seq, self._head_size)
        _v = _v[:,:,None].repeat(1,1,self._qkv_scale,1,1).reshape(_bs, -1, _seq, self._head_size)
        
        _score = _q @ _k.permute(0, 1, 3, 2) / self._dk
        _score = torch.softmax(_score, dim=-1)
        # _score = torch.softmax(_score + _causual, dim=-1)
        _o = _score @ _v
        _o = _o.permute(0, 2, 1, 3).reshape(_bs, _seq, -1)
        return self._ow(_o)

class MLP(nn.Module):
    def __init__(self, input_dim, hide_dim):
        super().__init__()
        self._l1 = nn.Linear(input_dim, hide_dim)
        self._l2 = nn.Linear(input_dim, hide_dim)
        self._l3 = nn.Linear(hide_dim, input_dim)
        self._gate = nn.SiLU()
        
    def forward(self, x):
        return self._l3(self._l1(x) * self._gate(self._l2(x)))

class LayerNormal(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        
        self._norm = nn.LayerNorm(input_dim)
        
    def forward(self, x):
        return self._norm(x)
    
# # 9/4下午进行了ROPE讲解，去看下#########################################################
def precompute_freqs_cis(dim, end, theta=50000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
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


if __name__ == "__main__":
    vit = VIT()
    print(vit(torch.randn(1, 1, 28, 28)).shape)
    print(vit)
    