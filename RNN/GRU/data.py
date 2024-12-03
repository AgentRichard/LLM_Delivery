import torch
from torch.utils.data import Dataset

class Data(Dataset):
    def __init__(self, data_dir, seqL):
        super().__init__()
        self._ids = torch.load(data_dir)
        self._ids = self._ids[ : (self._ids.shape[0] // seqL) * seqL]
        self._ids = self._ids.reshape(-1, seqL).to(torch.long)
        
    def __len__(self):
        return self._ids.shape[0]
    
    def __getitem__(self, index):
        return self._ids[index]
    
if __name__ == "__main__":
    data = Data("/root/project/My_projects/RNN/datas/data/sample", 200)
    print(len(data))
    print(data[0][0])