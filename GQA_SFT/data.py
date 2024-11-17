import torch
import pickle
from torch.utils.data import Dataset

class Data(Dataset):
    def __init__(self, data_dir, max_seq_len):
        super().__init__()
        
        self._seq_len = max_seq_len
        
        with open(data_dir, mode="rb") as fr:
            self.data = pickle.load(fr)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        _prompt, _label = self.data[index]
        # print(len(_prompt), len(_label))
        if len(_prompt) < self._seq_len:
            _zero_padding = (self._seq_len - len(_prompt)) * [0,]
            _prompt = _prompt + _zero_padding
            _label = _label + _zero_padding
        else:
            _prompt = _prompt[:self._seq_len]
            _label = _label[:self._seq_len]
            
        return (torch.tensor(_prompt, dtype=torch.int16),
                torch.tensor(_label, dtype=torch.int16))
        
        
if __name__ == "__main__":
    data = Data("/root/project/My_projects/Transformer/GQA_SFT/data/datas/ruozhiba.bin", 200)
    print(len(data))
    print(data[0][0].shape, data[0][1].shape)
    print(data[0][1])

    