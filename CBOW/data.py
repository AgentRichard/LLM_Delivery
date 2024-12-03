import torch
import numpy as np
import torch.nn.functional as F

sentences = ["Kage is Teacher", "Mazong is Boss", "Niuzong is Boss",
             "Xiaobing is Student", "Xiaoxue is Student",]

vocs = " ".join(sentences)
vocs = list(set(vocs.split(" ")))
print(vocs)
word2index = {word:index for index, word in enumerate(vocs)}
index2word = {index:word for index, word in enumerate(vocs)}
print(word2index)
print(index2word)

### 创建训练集
def raw_train_dataset(sentences, window_size):
    '''未独热编码的训练集 [( ,[]), ...], window_size是控制当前单词左右关注多少个单词，设置的越大计算量越高，但是上下文语义了解的越好'''
    dataset = []
    for sentence in sentences:
        sentence = sentence.split(" ")
        for _index, word in enumerate(sentence):
            otherwords = sentence[max(_index - window_size, 0) : _index] + \
                        sentence[_index+1 : min(len(sentence), _index + window_size + 1)]
            dataset.append((word, otherwords))
    return dataset

raw_dataset = raw_train_dataset(sentences, 10)
print(raw_dataset)

### 创建训练集，独热编码过后
def one_hot_train_dataset(raw_dataset, vocs):
    '''onehot 独热编码, 这一步可以直接在训练时进行转换，否则这里还要存储'''
    train_dataset = []
    for label, context in raw_dataset:
        label = torch.tensor([word2index[label]], dtype=torch.long)
        x = torch.tensor([word2index[word] for word in context])
        x = torch.stack([F.one_hot(x, len(vocs))]).float()
        x = x.reshape(-1, 9)
        train_dataset.append((label, x))
    return train_dataset

train_dataset = one_hot_train_dataset(raw_dataset, vocs)
print(train_dataset)
print(train_dataset[0][1].shape)
print(train_dataset[0][0].shape)



        
    
    
