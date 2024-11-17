import json
import torch
import pickle
import os
import sentencepiece as sp
from tqdm import tqdm

class Preprocess:
    def __init__(self, model_dir):
        self._spm = sp.SentencePieceProcessor()
        self._spm.Load(model_dir)
        
    def __call__(self, data_dir, save_dir):
        
        for data_file in os.listdir(data_dir):
            file_path = f"{data_dir}/{data_file}"
            file_name = os.path.basename(file_path).split(".")[0]
            with open(file_path, mode="r+", encoding="utf-8") as fr:
                voc = []
                data = json.load(fr)
                for _i, data_dict in tqdm(enumerate(data), total=len(data)):
                    human = data_dict["instruction"]
                    assistant = data_dict["output"]
                    
                    input = f"<s>system\n以下用中文回答问题</s><s>human\n{human}</s><s>assistant"
                    output = f"{assistant}</s>"
                    
                    input_ids = self._spm.Encode(input)
                    output_ids = self._spm.Encode(output)
                    
                    label = len(input_ids) * [0,] + output_ids
                    prompt = input_ids + output_ids
                    
                    voc.append((prompt, label))
            
            save_dir = f"{save_dir}/{file_name}.bin"
            with open(save_dir, mode="wb") as fw:
                pickle.dump(voc, fw)
    
                
if __name__ == "__main__":
    preprocess = Preprocess("/root/project/My_projects/Transformer/GQA_SFT/tokenizer.model")
    preprocess("/root/project/My_projects/Transformer/GQA_SFT/data/preprocess/raw_data", "/root/project/My_projects/Transformer/GQA_SFT/data/datas")
        