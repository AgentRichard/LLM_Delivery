import os
import torch
import sentencepiece as sp
import pandas as pd
import pickle

class Preprocess:
    def __init__(self, model_dir):
        self._spm = sp.SentencePieceProcessor()
        self._spm.Load(model_dir)
        
    def __call__(self, raw_dir, save_dir):
        for data_file in os.listdir(raw_dir):
            data_path = f"{raw_dir}/{data_file}"
            data_name = os.path.basename(data_path).split(".")[0]
            datas = pd.read_parquet(data_path)
            
            vocs = []
            for _, _rows in datas.iterrows():
                prompt = _rows["prompt"]
                chosen = _rows["chosen"]
                rejected = _rows["rejected"]
                
                prompt = f"<s>system\n以下用中文回答</s><s>human\n{prompt}</s><s>assistant"
                chosen = f"{chosen}</s>"
                rejected = f"{rejected}</s>"
                
                prompt_ids = self._spm.Encode(prompt)
                chosen_ids = self._spm.Encode(chosen)
                rejected_ids = self._spm.Encode(rejected)
                
                if len(prompt_ids) > 150:
                    ### 暴力截取前150个字，因为本地电脑跑不起来，所以减小长度，仅作测试
                    prompt_ids = prompt_ids[:151]
                else:
                    prompt_ids = prompt_ids + [0,]*(150-len(prompt_ids))
                
                chosen_input_ids = prompt_ids + chosen_ids
                chosen_label_ids = len(prompt_ids)*[0,] + chosen_ids
                
                rejected_input_ids = prompt_ids + rejected_ids
                rejected_label_ids = len(prompt_ids)*[0,] + rejected_ids
                
                vocs.append([chosen_input_ids, chosen_label_ids, rejected_input_ids, rejected_label_ids])
        
            with open(f"{save_dir}/{data_name}.bin", mode="wb") as fw:
                pickle.dump(vocs, fw)    
    
if __name__ == "__main__":
    preprocess = Preprocess("/root/project/My_projects/Transformer/GQA_RLHF/tokenizer.model")
    preprocess("/root/project/My_projects/Transformer/GQA_RLHF/datas/preprocess/rawdata",
               "/root/project/My_projects/Transformer/GQA_RLHF/datas/data")