import pickle
import torch
from torch.utils.data import Dataset

class Data(Dataset):
    def __init__(self, file_path, seq_len):
        super().__init__()
        with open(file_path, mode="rb") as fr:
            self._data = pickle.load(fr)
        self._seq_len = seq_len
            
    def __len__(self):
        return len(self._data)
    
    def __getitem__(self, index):
        _prompt,_label = self._data[index]
        
        if len(_prompt) < self._seq_len:
            _fill_zero = [0,] * (self._seq_len - len(_prompt))
            _prompt = _prompt + _fill_zero
        else:
            ### 这里为了方便直接 截取，但实际上要保证最后一个是 sep
            _prompt = _prompt[:self._seq_len]
        return (torch.tensor(_prompt, dtype=torch.int16), 
                torch.tensor(_label, dtype=torch.int16))

if __name__ == "__main__":
    data = Data("/root/project/My_projects/Transformer/Bert/datas/data/train", 200)
    print(len(data))
    print(data[0][0].shape, data[0][1])
    print(data[0][0])
    