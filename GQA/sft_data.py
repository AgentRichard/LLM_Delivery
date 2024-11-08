from torch.utils.data import Dataset
import torch
import pickle

class SFTDataSet(Dataset):
    def __init__(self, filepath, max_seqLen):
        self._max_seqLen = max_seqLen
        with open(filepath, "rb") as fr:
            self.datas = pickle.load(fr)
            
    def __len__(self):
        return len(self.datas)
    
    def __getitem__(self, index):
        _prompt, _tag = self.datas[index]
        _prompt_len = len(_prompt)
        
        if _prompt_len <= self._max_seqLen:
            _fill_zero = (self._max_seqLen - _prompt_len) * [0,]
            _prompt = _prompt + _fill_zero
            _tag = _tag + _fill_zero
        else:
            _prompt = _prompt[ : self._max_seqLen]
            _tag = _tag[ : self._max_seqLen]

        return (torch.tensor(_prompt, dtype = torch.long),
                torch.tensor(_tag, dtype = torch.long))
    
if __name__ == "__main__":
    _dataset = SFTDataSet("datas/SFT_preprocess/sft_data/ruozhiba_qa.bin", 100)
    print(len(_dataset))
    print(_dataset[0][0].shape)