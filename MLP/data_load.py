from PIL import Image
import os
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset

class Data(Dataset):
    def __init__(self, data_dir):
        self.data = []
        for file_name in os.listdir(data_dir):
            img_path = f"{data_dir}/{file_name}"
            for img in os.listdir(img_path):
                img_name = f"{img_path}/{img}"
                img = np.array(Image.open(img_name))
                img = torch.tensor(img, dtype=torch.float32).reshape(-1) / 255.
                y = F.one_hot(torch.tensor(int(file_name)), 10).to(torch.float32)
                self.data.append((img, y))
        print("数据转换完毕")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        x, label = self.data[index]
        return x, label
    
if __name__ == "__main__":
    data = Data("/root/project/My_projects/FCNN/data/train")
    print(len(data))
    print(data[0][0].shape)
    