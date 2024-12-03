import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F
from transformers import PretrainedConfig, PreTrainedModel, AutoConfig, GenerationConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.auto import modeling_auto

def precompute_freqs_cis(dim, end, theta=50000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)
                   [: (dim // 2)].float() / dim))
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

class Attention(nn.Module):
    def __init__(self,
                 input_dim,
                 hide_dim,
                 n_q_heads,
                 n_kv_heads,
                 cache_max_batch_size,
                 cache_max_seq_len
                 ):
        super().__init__()
        self._n_q_heads = n_q_heads
        self._n_kv_heads = n_kv_heads
        self._group = n_q_heads // n_kv_heads
        self._head_size = input_dim // self._n_q_heads

        self._qw = nn.Linear(input_dim, self._head_size*self._n_q_heads)
        self._kw = nn.Linear(input_dim, self._head_size*self._n_kv_heads)
        self._vw = nn.Linear(input_dim, self._head_size*self._n_kv_heads)
        self._ow = nn.Linear(input_dim, input_dim, bias=False)

        self._cache_max_batch_size = cache_max_batch_size
        if self._cache_max_batch_size is not None:
            _cache_k = torch.zeros((cache_max_batch_size,
                                    cache_max_seq_len,
                                    n_kv_heads,
                                    self._head_size))
            self.register_buffer("_cache_k", _cache_k, persistent=False)

            _cache_v = torch.zeros((cache_max_batch_size,
                                    cache_max_seq_len,
                                    n_kv_heads,
                                    self._head_size))
            self.register_buffer("_cache_v", _cache_v, persistent=False)

    def forward(self, x, freq_cis, start_pos):
        _bn, _seq, _ = x.shape
        _q, _k, _v = self._qw(x), self._kw(x), self._vw(x)
        _q = _q.reshape(_bn, _seq, self._n_q_heads, self._head_size)
        _k = _k.reshape(_bn, _seq, self._n_kv_heads, self._head_size)
        _v = _v.reshape(_bn, _seq, self._n_kv_heads, self._head_size)

        _q = apply_rotary_emb(_q, freq_cis[start_pos:start_pos+_seq])
        _k = apply_rotary_emb(_k, freq_cis[start_pos:start_pos+_seq])

        if self._cache_max_batch_size is not None:
            self._cache_k[:_bn, start_pos: start_pos + _seq] = _k
            self._cache_v[:_bn, start_pos: start_pos + _seq] = _v

            _k = self._cache_k[:_bn, : start_pos + _seq]
            _v = self._cache_v[:_bn, : start_pos + _seq]

        _q = _q.permute(0, 2, 1, 3)
        _k = _k.permute(0, 2, 1, 3)
        _v = _v.permute(0, 2, 1, 3)

        _k = _k[:, :, None].repeat(1, 1, self._group, 1, 1).reshape(
            _bn, -1, start_pos+_seq, self._head_size)
        _v = _v[:, :, None].repeat(1, 1, self._group, 1, 1).reshape(
            _bn, -1, start_pos+_seq, self._head_size)

        _o = F.scaled_dot_product_attention(
            _q, _k, _v, attn_mask=None, is_causal=True)

        _o = _o.permute(0, 2, 1, 3)
        _o = _o.reshape(_bn, _seq, -1)
        return self._ow(_o)


class FFN(nn.Module):
    def __init__(self, input_dim, hide_dim):
        super().__init__()

        self._w0 = nn.Linear(input_dim, hide_dim)
        self._w1 = nn.Linear(input_dim, hide_dim)
        self._w2 = nn.Linear(hide_dim, input_dim, bias=False)
        self._gate = nn.SiLU()

    def forward(self, x):
        _o0 = self._w0(x)
        _o1 = self._w1(x)
        _g = self._gate(_o1)
        _og = _o0*_g
        return self._w2(_og)


class RMSNormal(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self._w = nn.Parameter(torch.randn(input_dim))

    def forward(self, x):
        return self._w * x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True)+1e-6)


class TransformerLayer(nn.Module):
    def __init__(self,
                 input_dim,
                 hide_dim,
                 n_q_heads,
                 n_kv_heads,
                 cache_max_batch_size,
                 cache_max_seq_len):
        super().__init__()
        self._att_norm = RMSNormal(input_dim)
        self._att_layer = Attention(input_dim,
                                    hide_dim,
                                    n_q_heads,
                                    n_kv_heads,
                                    cache_max_batch_size,
                                    cache_max_seq_len)
        self._ffn_norm = RMSNormal(input_dim)
        self._ffn_layer = FFN(input_dim,
                              hide_dim)

    def forward(self, x, freq_cis, start_pos):
        _x = x
        _x = self._att_norm(_x)
        _x = self._att_layer(_x, freq_cis, start_pos)
        _x = x + _x

        _y = _x
        _y = self._ffn_norm(_y)
        _y = self._ffn_layer(_y)
        _y = _y + _x
        return _y


class TransformerDecoder(nn.Module):
    def __init__(self,
                 num_layers,
                 input_dim,
                 hide_dim,
                 n_q_heads,
                 n_kv_heads,
                 max_len,
                 cache_max_batch_size=None,
                 cache_max_seq_len=None
                 ):
        super().__init__()
        self._layers = nn.ModuleList(
            [TransformerLayer(input_dim,
                              hide_dim,
                              n_q_heads,
                              n_kv_heads,
                              cache_max_batch_size,
                              cache_max_seq_len) for _ in range(num_layers)]
        )
        self._out_norm = RMSNormal(input_dim)
        _freq_cis = precompute_freqs_cis(input_dim//n_q_heads, max_len)
        self.register_buffer("freq_cis", _freq_cis, persistent=False)

    def forward(self, x, start_pos):
        _x = x
        for _layer in self._layers:
            _x = _layer(_x, self.freq_cis, start_pos)
        return self._out_norm(_x)
    
class SkyerConfig(PretrainedConfig):
    model_type = "Skyer"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.layers_num = kwargs.get("layers_num")
        self.input_dim = kwargs.get("input_dim")
        self.hide_dim = kwargs.get("hide_dim")
        self.n_q_heads = kwargs.get("qheads_num")
        self.n_kv_heads = kwargs.get("kvheads_num")
        self.max_len = kwargs.get("max_pos_len")
        self.num_vocs = kwargs.get("vocab_size")
        self.cache_max_batch_size = kwargs.get("cache_max_batch_size")
        self.cache_max_seq_len = kwargs.get("cache_max_seq_len")
        
        self.pad_token_id = 0
        self.bos_token_id = 2
        self.eos_token_id = 3       ## 这三个参数只要继承了PretrainedConfig就会被写死
        self.auto_map = {
            "AutoModelForCausalLM": "model.SkyerModel",
            "AutoConfig": "model.SkyerConfig"
        }

class SkyerModel(PreTrainedModel):
    config_class = SkyerConfig
    ## Huggingface会自己调初始化权重的函数
    def __init__(self, config):
        super().__init__(config)

        self._cache_max_batch_size = config.cache_max_batch_size
        
        self._emb = nn.Embedding(config.vocab_size, config.input_dim)
        
        self._tf_layer = TransformerDecoder(
            num_layers = config.layers_num,
            input_dim = config.input_dim,
            hide_dim = config.hide_dim,
            n_q_heads = config.qheads_num,
            n_kv_heads = config.kvheads_num,
            max_len = config.max_pos_len,
            cache_max_batch_size = config.cache_max_batch_size,
            cache_max_seq_len = config.cache_max_seq_len)
    
    def _forward(self, input_ids, start_pos):
        _tokens = self._emb(input_ids)
        _features = self._tf_layer(_tokens, start_pos)
        return _features @ self._emb.weight.T
    
    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        # input_ids = prompt 你是谁 response 我是kkk
        # labels = -100, -100, ..., -100, 我是
        ### 如果模型基座不够好，那么将prompt也放入模型中去训练
        ### 如果模型基座够好，可以将prompt遮住来变成-100来进行训练，激发模型的能力
        ### 因此SFT训练有两个时期，初期和成熟期，初期是按照预训练的方式对整个指示的问题与答案模式进行学习与做损失
        ### 待模型明白了指示框架，就进入到成熟期的训练，也就是 训练时给模型问题让他去回答内容，标签隐去问题单纯看答案是否正确即可(做损失)。
        _labels = labels
        _loss = None
                            
        if _labels is not None:
            # 元婴期SFT
            _input_ids = input_ids[:, :-1]
            _labels = input_ids[:, 1: ]
            _logits = self._forward(_input_ids, start_pos = 0)
            # 成熟期SFT
            # _labels[_labels < 0] = self.config.pad_token_id 
            _logits = _logits - _logits.mean(dim=-1, keepdim=True)
            _o = _logits.reshape(-1, self.config.vocab_size)
            _t = _labels.reshape(-1)
            _loss = F.cross_entropy(_o, _t, ignore_index=self.config.pad_token_id)
        else:
            _logits = self._forward(input_ids, start_pos=0)
            
        return CausalLMOutputWithPast(
            loss = _loss,               
            logits = _logits,           
            past_key_values = None,     
            hidden_states = None,       
            attentions = None           
        )

    def _generate(self, ids, start_pos, prompt_len, genenration_config):     ##genenrate是Huggingface规定的推理专用接口
        if start_pos > prompt_len + genenration_config.max_new_tokens:
            return self.config.eos_token_id
        _outputs = self._forward(ids, start_pos)
        _output = _outputs[:, -1]
        _weight, _indices = torch.topk(_output, genenration_config.top_k, dim=-1)
        _probs = self._tsoftmax(_weight, genenration_config.temperature)
        _selected_indices = torch.multinomial(_probs, 1)
        _id = torch.gather(_indices, dim=-1, index=_selected_indices)
        return _id
    
    _generation_config = GenerationConfig(
        do_sample = True,
        temperature = 1,
        top_k = 10,
        top_p = 1,
        max_new_tokens = 50)
    
    def generate(self, input_ids, generation_config=_generation_config, **kwargs):  ## **kwargs的作用是能够接受attention mask
        _ids = input_ids
        prompt_len = len(_ids[0])
        _id = self._generate(_ids, 0, prompt_len, generation_config)

        for _start_pos in range(_ids.shape[1], self.config.cache_max_seq_len):
            _id = self._generate(_id, _start_pos, prompt_len, generation_config)
            if _id == self.config.eos_token_id:
                break
            # yield _id
            _ids = torch.cat((_ids, _id), dim=-1)
        return _ids
    
    def get_input_embeddings(self):
        return self._emb
    
    def set_input_embeddings(self, value):
        self._emb = value
        
    @staticmethod
    def _tsoftmax(x, temp):
        x = x - x.mean()
        return torch.exp(x/temp) / (torch.exp(x/temp).sum(-1) + 1e-4)
    
AutoConfig.register("Skyer", SkyerConfig)
modeling_auto.MODEL_FOR_CAUSAL_LM_MAPPING_NAMES["Skyer"] = "SkyerModel"

if __name__ == "__main__":
    import os
    config = SkyerConfig(
            layers_num = 48,
            input_dim = 768,
            hide_dim = 3072,
            qheads_num = 12,
            kvheads_num = 2,
            max_pos_len = 16384,
            vocab_size = 30000,
            cache_max_batch_size = None,
            cache_max_seq_len = None)
    
    model = SkyerModel(config)
    model.load_state_dict(torch.load(
        "/root/project/My_projects/MS_PEFT_RLHF_mymodel/pretrain_weights/mp_rank_00_model_states.pt", 
        weights_only=False)["module"])
    
    save_directory = "/root/project/My_projects/MS_PEFT_RLHF_mymodel/_cache"
    model.save_pretrained(save_directory)
    os.system(f"cp /root/project/My_projects/MS_PEFT_RLHF_mymodel/model.py {save_directory}")
    
    print("Model and configuration saved successfully.")
