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
        chosen_input_ids, chosen_label_ids, rejected_input_ids, rejected_label_ids = self.data[index]
        # print(len(_prompt), len(_label))
        if len(chosen_input_ids) < self._seq_len:
            _zero_padding = (self._seq_len - len(chosen_input_ids)) * [0,]
            chosen_input_ids = chosen_input_ids + _zero_padding
            chosen_label_ids = chosen_label_ids + _zero_padding
        else:
            chosen_input_ids = chosen_input_ids[:self._seq_len]
            chosen_label_ids = chosen_label_ids[:self._seq_len]
            
        if len(rejected_input_ids) < self._seq_len:
            _zero_padding = (self._seq_len - len(rejected_input_ids)) * [0,]
            rejected_input_ids = rejected_input_ids + _zero_padding
            rejected_label_ids = rejected_label_ids + _zero_padding
        else:
            rejected_input_ids = rejected_input_ids[:self._seq_len]
            rejected_label_ids = rejected_label_ids[:self._seq_len]
            
        return (torch.tensor(chosen_input_ids, dtype=torch.int16),
                torch.tensor(chosen_label_ids, dtype=torch.int16),
                torch.tensor(rejected_input_ids, dtype=torch.int16),
                torch.tensor(rejected_label_ids, dtype=torch.int16),)
        
if __name__ == "__main__":
    data = Data("/root/project/My_projects/Transformer/GQA_RLHF/datas/data/Chinese_dpo_pairs.bin", 250)
    print(len(data))
    print(data[0][0].shape, data[0][1].shape)
    print(data[0][2].shape, data[0][3].shape)
    print(data[0][3])

    