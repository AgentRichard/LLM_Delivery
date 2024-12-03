from transformers import PreTrainedTokenizer
import sentencepiece as spm
import json
import os

class SentencepieceTokenizer(PreTrainedTokenizer):
    def __init__(self, model_file, vocab_file, **kwargs):
        self._model_file = model_file
        self._vocab_file = vocab_file
        
        self.sp = spm.SentencePieceProcessor(model_file = model_file)
        
        with open(vocab_file, "r", encoding="utf-8") as fr:
            self.vocab = {line.strip().split("\t")[0] : i for i, line in enumerate(fr)}
            ## [str, index]
        self.id_to_token = {v:k for k, v in self.vocab.items()}
            ## [index, str]
            
        super().__init__(**kwargs)
        
        self.chat_template = """
        {%- for message in messages %}
            {%- if (message.role == "system") %}{{- '<s>system:\n'+ message.content + '</s>\n' }}{%- endif %}
            {%- if (message.role == "user") %}{{- '<s>user:\n'+ message.content + '</s>\n' }}{%- endif %}
            {%- endfor %}
        {{- '<s>assistant:\n' }}
        """
        ## 上面这个模板是千问的模板，研究一下
    
    @property
    def vocab_size(self):
        return len(self.vocab)
    
    def get_vocab(self):
        return self.vocab    
    
    def _tokenize(self, text):
        return self.sp.encode(text, out_type=str)
        ### 父类中的代码重写
    
    def _convert_token_to_id(self, token):
        return self.vocab.get(token, self.vocab["<unk>"])
    
    def _convert_id_to_token(self, index):
        return self.id_to_token.get(index, "<unk>")
    
    def convert_tokens_to_string(self, tokens):
        return self.sp.decode(tokens)
    
    def save_vocabulary(self, save_directory: str, filename_prefix: str = None):
        os.makedirs(save_directory, exist_ok=True)
        os.system(f"cp /root/project/My_projects/MS_PEFT_RLHF_mymodel/tokenizer.py {save_directory}")
        os.system(f"cp {self._model_file} {save_directory}")
        os.system(f"cp {self._vocab_file} {save_directory}")

        tokenizer_config = {
            "bos_token": "<s>",
            "eos_token": "</s>",
            "pad_token": "<pad>",
            "unk_token": "<unk>",
            "tokenizer_class": "SentencepieceTokenizer",
            "model_file": self._model_file,
            "vocab_file": self._vocab_file,
            "auto_map": {
                "AutoTokenizer": [None, "tokenizer.SentencepieceTokenizer"]
            }
        }
        config_file = os.path.join(save_directory, "tokenizer_config.json")
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(tokenizer_config, f, ensure_ascii=False, indent=2)

        return self._vocab_file, self._model_file


if __name__ == '__main__':
    # 使用自定义的 Tokenizer
    tokenizer = SentencepieceTokenizer(
        model_file='/root/project/My_projects/MS_PEFT_RLHF_mymodel/tools/tokenizer.model', 
        vocab_file='/root/project/My_projects/MS_PEFT_RLHF_mymodel/tools/tokenizer.vocab')

    # 测试编码
    text = '<s>这是一个测试句子</s>'
    tokens = tokenizer.tokenize(text)       ### tokenizer.tokenize(text) == tokens      ## 调用上面自定义的 _tokenize() output = str
    print(tokens)
    
    tokens2 = tokenizer([text])             ### tokenizer([text]) ==> {"input_ids":..., "token_type_ids":..., "attention_mask":...}
    print(tokens2)                          ### 自动对 text 中的每个内容 调用 _convert_token_to_id， 因此运行11次， 自动添加 mask 和type的维度
    # 测试解码
    decoded_text = tokenizer.convert_tokens_to_string(tokens)
    print("Decoded text:", decoded_text)

    tokenizer.save_pretrained("/root/project/My_projects/MS_PEFT_RLHF_mymodel/_cache")  ## 自动调用 save_vocabulary
