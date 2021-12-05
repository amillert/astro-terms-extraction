import lstm_BIO_train
import torch
import transformers
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import os
import torch.utils.data as Data

device = 'cuda'

def vectorize_data(data:str):
    vector_of_word = []
    data = data.split(' ')
    dict_word_bert = np.load('word_bert.npy')
    model = transformers.AutoModel.from_pretrained('distilbert-base-uncased')
    tokenizer = transformers.AutoTokenizer.from_pretrained('distilbert-base-uncased')
    vectorizor = transformers.pipeline('feature-extraction', model=model, tokenizer=tokenizer)

    for word in data:
        try:
            vector_of_word.append(dict_word_bert.item()[word])
        except:
            vector_of_word.append(vectorizor(word)[0][0])

    return  vector_of_word

def get_model():
    model = lstm_BIO_train.tagger_lstm()
    parameters = torch.load('tagger_lstm_model')
    model.load_state_dict(parameters)
    return model

def get_label(path):
    label_list = []
    label_num_dict = {'B':0, 'I':1, 'O':2}
    with open(path, 'r', encoding='utf-8') as file:
        for line in file.readlines():
            label = line.replace('\n', '').split('\t')[1]
            label_list.append(label_num_dict[label])

    return torch.tensor(label_list)

def get_dataloader(data_set):
    params = {'batch_size': 1,
              'shuffle': False,
              'num_workers': 0}
    data_loader = DataLoader(data_set, **params)
    return data_loader

def get_stroe_result(model, data_loader, doc):
    path = 'result/BIO_tagger/'
    result_num = []
    result_BIO = []
    for word_vector in data_loader:
        input_vector = word_vector[0]
        value = model(input_vector)
        max_value, max_index = torch.max(value, dim=1)
        result_num += max_index

    for result in result_num:
        if result == 0:
            result_BIO.append('B')
        elif result == 1:
            result_BIO.append('I')
        else:
            result_BIO.append('O')



    return result_num, result_BIO

if __name__ == "__main__":
    vectorize_data('darl matter')
    docs = list(map(lambda l: l.strip(), open("out-astro", "r", encoding="utf-8").readlines()))
    docs =docs[:10]
    model = get_model().to(device)
    words = []
    tag = []
    num =0
    for doc in docs:
        num += 1
        print(num)
        words += doc
        vectors = torch.tensor(vectorize_data(doc)).to(device)
        # print(vectors)
        # print(type(vectors)
        data_set = Data.TensorDataset(vectors)
        data_loader = get_dataloader(data_set)
        result_num, result_BIO = get_stroe_result(model, data_loader,docs[0])
        tag += result_BIO
    
    with open('result/BIO_tagger/result.txt', 'w', encoding='utf-8') as file:
        for index in range(len(tag)):
            file.write(words[index] + '\t' + tag[index] + '\n')



