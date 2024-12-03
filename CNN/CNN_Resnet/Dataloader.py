from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

class Data(Dataset):
    def __init__(self, file_dir):
        self._data = []
        for _filedir in os.listdir(file_dir):
            _filepath = f"{file_dir}/{_filedir}"
            for _imgdir in os.listdir(_filepath):
                _imgpath = f"{_filepath}/{_imgdir}"
                _img = np.array(Image.open(_imgpath))
                _img = torch.tensor(_img, dtype=torch.float32) / 255.  ## 因为我们需要原图形的宽和高，因此不需要拉平
                # _tag = F.one_hot(torch.tensor(int(_filedir)), 10).to(torch.float32)
                _tag = torch.tensor(int(_filedir))
                ### 交叉熵损失自带One HOT 因此在数据处理时不需要 转成onehot也不需要转成浮点数！！！！！！！！！！！！！！！！！！
                self._data.append((_img, _tag))
        print("图片数据加载与预处理完毕！")

    def __len__(self):
        return len(self._data)
    
    def __getitem__(self, index):
        _x, _y = self._data[index]
        _x = _x[None]
        return _x, _y
            
    
if __name__ == "__main__":
    data = Data("/root/project/My_projects/FCNN/data/train")
    print(data[0][0].shape)
    