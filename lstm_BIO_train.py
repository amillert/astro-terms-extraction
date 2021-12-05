import transformers
import torch
import pandas as pd
from torch.utils.data import DataLoader
import os
import torch.utils.data as Data
import sklearn as sk
import numpy as np

np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
main_path = 'annotated/BIO/'
file_list = os.listdir(main_path)
device = 'cuda'

def get_data_lable():
    data = []
    label = []

    for file_path in file_list:
        with open(main_path+file_path, 'r', encoding='utf-8') as file:
            for line in file.readlines():
                data.append(line.replace('\n', '').split('\t')[0])
                label.append(line.replace('\n', '').split('\t')[1])

    for index in range(len(label)):
        if label[index] == 'B':
            label[index] = 0
        elif label[index] == 'I':
                label[index] = 1
        else:
            label[index] = 2

    return data, label

def get_bert(word_single):
    model = transformers.AutoModel.from_pretrained('distilbert-base-uncased')
    tokenizer = transformers.AutoTokenizer.from_pretrained('distilbert-base-uncased')
    vectorizor = transformers.pipeline('feature-extraction', model=model, tokenizer=tokenizer)
    return vectorizor(word_single)[0][0]

def get_data_set(dict_word_bert, data):
    data_set = []

    for word in data:
        try:
            data_set.append(dict_word_bert.item()[word])
        except:
            data_set.append(get_bert(word))

    return torch.tensor(data_set)

def vectorize_data(data):
    model = transformers.AutoModel.from_pretrained('distilbert-base-uncased')
    tokenizer = transformers.AutoTokenizer.from_pretrained('distilbert-base-uncased')
    vectorizor = transformers.pipeline('feature-extraction', model=model, tokenizer=tokenizer)
    vector_of_word = [word[0] for word in vectorizor(data)]
    print(len(vector_of_word))
    vector_of_word = torch.tensor(vector_of_word)
    torch.save(vector_of_word, 'vector_of_word')
    return  vector_of_word


class tagger_lstm(torch.nn.Module):
    def __init__(self):
        super(tagger_lstm, self).__init__()
        self.lstm = torch.nn.LSTM(
            input_size=768,
            batch_first=True,
            hidden_size=768,
            num_layers=1
        )
        self.classifier = torch.nn.Linear(768, 3)

    def forward(self, input_data):
        input_data = input_data.view(len(input_data), 1, -1)
        input_data = self.lstm(input_data)[0]
        input_data = input_data.reshape(input_data.size(0), -1)
        out = self.classifier(input_data)
        return out

def train(model, data_set, num_epoch):
    params = {'batch_size': 1,
              'shuffle': False,
              'num_workers': 0}
    data_loader = DataLoader(data_set, **params)
    model = model.to(device)
    loss_function = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-05)
    model.train()

    for epoch in range(num_epoch):
        num_correct_epoch = 0
        total_epoch = 0
        num_correct_step =0
        total_step = 0
        step = 0
        for value, label in data_loader:
            total_epoch += len(label)
            total_step += len(label)
            step += 1
            value_pred = model(value.to(device)).float()
            max_val, max_index = torch.max(value_pred, dim=1)
            loss = loss_function(value_pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            for pos in range(len(label)):
                if label[pos] == max_index[pos]:
                    num_correct_epoch += 1
                    num_correct_step += 1
        print('correct rate per epoch:' +  '%.2f%%' % (num_correct_epoch/total_epoch * 100))

def valid(model, data_set):
    num_correct = 0
    num_sampel = 0
    params = {'batch_size': 1,
              'shuffle': False,
              'num_workers': 0}
    data_loader = DataLoader(data_set, **params)
    model = model.to(device)

    for value, label in data_loader:
        num_sampel += len(label)
        value_pred = model(value.to(device)).float()
        max_val, max_index = torch.max(value_pred, dim=1)

        for pos in range(len(label)):
            if label[pos] == max_index[pos]:
                num_correct += 1

    print('correct rate:' + '%.2f%%' % (num_correct / num_sampel * 100))


if __name__ == "__main__":
    data, label = get_data_lable()
    # print(data)
    # data = list(set(data))
    # dict_word_bert = dict()
    # num = 0
    # for word in data:
    #     num += 1
    #     print(num)
    #     dict_word_bert[word] = get_bert(word)
    #
    # np.save('word_bert.npy', dict_word_bert)

    dict_word_bert = np.load('word_bert.npy')
    print(type(dict_word_bert))
    print(dict_word_bert.item()['dark'])

    label = torch.tensor(label).to(device)
    vector_data = get_data_set(dict_word_bert, data)
    print(len(vector_data))
    data_train = vector_data[0: int(0.8 * len(label))]
    label_train = label[0: int(0.8 * len(label))]
    data_test = vector_data[int(0.8 * len(label)):]
    label_test = label[int(0.8 * len(label)):]
    data_set = Data.TensorDataset(vector_data, label)
    train_set = Data.TensorDataset(data_train, label_train)
    test_set = Data.TensorDataset(data_test, label_test)

    model = tagger_lstm()
    train(model, train_set, 2)
    valid(model, test_set)

    torch.save(model.state_dict(), 'tagger_lstm_model')
