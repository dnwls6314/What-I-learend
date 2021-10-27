# -*- coding: utf-8 -*-
"""
Created on Mon May 24 12:34:42 2021

@author: Woojin
"""
# import some packages you need here
import torch
from torch.utils.data import Dataset, DataLoader

class Shakespeare(Dataset):
    """ Shakespeare dataset

        To write custom datasets, refer to
        https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

    Args:
        input_file: txt file

    Note:
        1) Load input file and construct character dictionary {index:character}.
					 You need this dictionary to generate characters.
				2) Make list of character indices using the dictionary
				3) Split the data into chunks of sequence length 30. 
           You should create targets appropriately.
    """

    def __init__(self, input_file, is_train=True):
        self.sq_length = 30
        self.encoded_line, self.int2char, self.char2int = integer_encoding(input_file)
        self.input, self.target = make_mini_batch(self.encoded_line, self.sq_length)

               
        if is_train == True : 
            self.input = delete_data(self.input[: int(self.input.size(0) * 0.9)])
            self.target = delete_data(self.target[: int(self.target.size(0) * 0.9)])
        else :
            self.input = delete_data(self.input[int(self.input.size(0) * 0.9) : ], is_train=False)
            self.target = delete_data(self.target[int(self.target.size(0) * 0.9) : ], is_train=False)
        

    def __len__(self):
        return len(self.input)
    

    def __getitem__(self, idx):
        
        input = self.input[idx]
        target = self.target[idx]

        return input, target

 
def integer_encoding(input_line) :
    lines = []
    
    # 대사에서 등장인물 삭제
    input_line = ''.join([word for word in list(filter(None, input_line.splitlines())) if ':' not in word])
    lines += [words for words in input_line] 
    
    # 등장인물
    characters = list(set(lines))
    
    # {등장인물:index} --> {index:등장인물}로 변경
    char2int = {char : idx for idx, char in enumerate(characters)}
    encoded_line = list(map(char2int.get, lines, lines))
    encoded_line = encoded_line
    int2char = dict((idx, char) for char, idx in char2int.items())
    
    return encoded_line, int2char, char2int
    


def make_mini_batch(data, sq_length) :
    # sequence data를 30 chunk로 변경, t 시점의 target은 (t+1)시점의 input과 같음
    input = []; target = []
    for index in range(len(data) - 1) :
        if index >= sq_length -1  :
            row_x = [] ; row_y = []
            row_x += [data[index-idx] for idx in range(sq_length)]
            row_x.reverse(); row_y = row_x.copy()
            row_y.pop(0); row_y.append(data[index+1])
            input.append(row_x); target.append(row_y)
    
            
    return torch.tensor(input, dtype=torch.float), torch.tensor(target, dtype=torch.float)
    


def one_hot_encoding(input_data) :
    # 원핫인코딩을 통해 2 dimension matrix(input dimension)를 three dimensional tensor(RNN input)로 변경
    dict_size = 59
    output = []

    
    for row in input_data.split(1) :
        row = row[0]
        row_to_matrix = []
        for x in row :
            one_hot_vector = torch.eye(dict_size, dtype=torch.float)[int(x)]
            one_hot_vector = one_hot_vector.tolist()
            row_to_matrix.append(one_hot_vector)
        
        output.append(row_to_matrix)
    output = torch.tensor(output)
    return output



def delete_data(x, is_train=True) :
    if is_train==True :
        remove_data = 38
    else :
        remove_data = 27
    x = x[:-remove_data]
    return x


if __name__ == '__main__':

    input_file = open('./data/shakespeare_train.txt', 'r').read()
    train_dataset = Shakespeare(input_file, is_train=True)
    test_dataset = Shakespeare(input_file, is_train=False)
    
    train_data = DataLoader(train_dataset, batch_size=100)
    test_data = DataLoader(test_dataset, batch_size=100)
    print(len(train_dataset))
    print(len(train_data))

