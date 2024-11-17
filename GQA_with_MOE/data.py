import torch
import os
from torch.utils.data import Dataset

class Data(Dataset):
    def __init__(self, data_dir, seqL):
        """文本变成 NS结构即可"""
        super().__init__()
        self._ids = torch.load(data_dir)
        self._ids = self._ids[:(len(self._ids) // seqL) * seqL] ## 将整个数据集划分成 可被SeqL 等分的长度
        self._ids = self._ids.reshape(-1, seqL).to(torch.long)
        
    def __len__(self):
        return self._ids.shape[0]
    
    def __getitem__(self, index):
        return self._ids[index]
    
if __name__ == "__main__":
    data = Data("/root/project/My_projects/Transformer/GQA_with_MOE/datas/data/Skypile_RawData", 300)
    print(len(data))
    print(data[0])
    