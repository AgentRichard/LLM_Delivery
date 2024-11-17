import sentencepiece as sp
import json
import os
from tqdm import tqdm
import torch
from multiprocessing import Pool, cpu_count

class Tokenizer:
    def __init__(self, model_dir):
        self._spm = sp.SentencePieceProcessor()
        self._spm.Load(model_dir)
        
    def preprocess(self, file_path, save_dir):
        file_name = os.path.basename(file_path).split(".")[0]
        print(file_name)
        _vocs = []
        with open(file_path, mode="r", encoding="utf-8") as fr:
            for line in tqdm(fr):
                text = json.loads(line)
                text = text["text"]
                _ids = self._spm.Encode(text)
                _vocs.append(2)
                _vocs.extend(_ids)
                _vocs.append(3)
        
            _vocs = torch.tensor(_vocs, dtype=torch.int16)
            torch.save(_vocs, f"{save_dir}/{file_name}")
            
    def __call__(self, raw_data_dir, save_dir):
        file_paths = [os.path.join(raw_data_dir, file) for file in os.listdir(raw_data_dir)]
        print(file_paths)
        with Pool(cpu_count()) as pool:
            pool.starmap(self.preprocess, [(file_path, save_dir) for file_path in file_paths])
            
            
if __name__ == "__main__":
    tokenizer = Tokenizer("/root/project/My_projects/Transformer/GQA_with_MOE/tokenizer.model")
    tokenizer("/root/project/My_projects/Transformer/GQA_with_MOE/datas/preprocess/raw_data", 
              "/root/project/My_projects/Transformer/GQA_with_MOE/datas/data")
    