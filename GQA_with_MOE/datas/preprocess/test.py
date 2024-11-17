import sentencepiece as sp
import json

data_dir = '/root/project/My_projects/Transformer/GQA_with_MOE/datas/preprocess/sample.jsonl'
model_dir = '/root/project/My_projects/Transformer/GQA_with_MOE/tokenizer.model'

spm = sp.SentencePieceProcessor()
spm.Load(model_dir)

with open(data_dir, mode="r", encoding="utf-8") as fr:
    data = fr.readline()
    print(spm.Encode(data)[0])

print(spm.Encode("我爱北京天安门", out_type=str))
print(spm.Encode("我爱北京天安门", out_type=int))