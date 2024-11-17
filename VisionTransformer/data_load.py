from PIL import Image
from torch.utils.data import Dataset
import os
import numpy as np
import torch.nn.functional as F
import torch

class Data(Dataset):
    def __init__(self, data_dir):
        self.data = []
        for data_file_name in os.listdir(data_dir):
            file_path = f"{data_dir}/{data_file_name}"
            for img_name in os.listdir(file_path):
                img_path = f"{file_path}/{img_name}"
                img = np.array(Image.open(img_path), dtype=np.int16) / 255.
                img = torch.tensor(img, dtype=torch.float16)
                label = torch.tensor(int(data_file_name), dtype=torch.int8)
                self.data.append((img, label))
        print("数据转换完毕")
    
    def __len__(self):
        ### 返回的是样本总数, 为 N
        return len(self.data)
    
    def __getitem__(self, index):
        _x, _t = self.data[index]
        return _x[None], _t
    
if __name__ == "__main__":
    data = Data("/root/project/My_projects/VisionTransformer/data/train")
    print(data[0][0].shape)
    print(len(data))
    