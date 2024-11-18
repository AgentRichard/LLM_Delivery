import pandas as pd
import pickle
import os
import sentencepiece as sp

class Preprocess:
    def __init__(self, model_dir):
        self._spm = sp.SentencePieceProcessor()
        self._spm.Load(model_dir)
        # print(self._spm.Encode("[CLS]"))
        
    def __call__(self, data_dir, save_dir):
        for data_file in os.listdir(data_dir):
            data_path = f"{data_dir}/{data_file}"
            file_name = os.path.basename(data_path).split(".")[0]
            vocs = []
            datas = pd.read_csv(data_path)
            # print(datas[:3])
            for _, _row in datas.iterrows():
                sentence1 = _row["sentence1"]
                sentence2 = _row["sentence2"]
                label = int(_row["label"])
                
                prompt = "[CLS]" + sentence1 + "[SEP]" + sentence2 + "[SEP]"
                prompt_ids = self._spm.Encode(prompt,enable_sampling=False)
                vocs.append([prompt_ids, label])
            with open(f"{save_dir}/{file_name}", mode="wb") as fw:
                pickle.dump(vocs, fw)                
    
if __name__ == "__main__":
    preprocess = Preprocess("/root/project/My_projects/Transformer/Bert/tokenizer.model")
    preprocess("/root/project/My_projects/Transformer/Bert/datas/rawdata",
               "/root/project/My_projects/Transformer/Bert/datas/data")
        
    