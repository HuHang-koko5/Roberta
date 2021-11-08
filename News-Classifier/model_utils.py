import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoConfig, AutoModel, Trainer
from tqdm import tqdm, trange, tqdm_notebook
from tqdm.notebook import tqdm as notetqdm
from torch.autograd import Variable
from transformers import TrainingArguments
from transformers import RobertaTokenizer, RobertaForSequenceClassification, RobertaConfig
import random
import time
import os


# load json/csv data

def load_data(path, cate_size, type='JSON', percentage=1):
    if type == 'JSON':
        df = pd.read_json(path)
    else:
        df = pd.read_csv(path)
    df.iloc[np.random.permutation(len(df))]
    labels = df['category'].tolist()
    heads = df['headline'].tolist()
    descriptions = df['description'].tolist()
    contents = [h + d for h, d in zip(heads, descriptions)]
    label_dic = {}
    label_count = {}
    final_size = int(len(contents) * percentage)
    if percentage != 1:
        contents = contents[:final_size]
        labels = labels[:final_size]
    # itos                                        
    label_set = ['WORLD NEWS', 'ARTS & CULTURE', 'WEDDINGS', 'PARENTING',
                 'BUSINESS & FINANCES', 'HOME & LIVING', 'EDUCATION',
                 'WELLNESS', 'POLITICS', 'WOMEN', 'IMPACT', 'ENVIRONMENT',
                 'SPORTS', 'FOOD & DRINK', 'GROUPS VOICES', 'MEDIA',
                 'SCIENCE & TECH', 'CRIME', 'WEIRD NEWS', 'COMEDY',
                 'RELIGION', 'MISCELLANEOUS', 'DIVORCE', 'ENTERTAINMENT',
                 'STYLE & BEAUTY', 'TRAVEL']

    # stoi
    for idx, label in enumerate(label_set):
        label_dic[label] = idx
    flitered_labels = []
    flitered_contents = []
    for cate, cont in zip(labels, contents):
        if cate not in label_count.keys():
            label_count[cate] = 1
            flitered_labels.append(cate)
            flitered_contents.append(cont)
        elif label_count[cate] < cate_size:
            label_count[cate] += 1
            flitered_labels.append(cate)
            flitered_contents.append(cont)
    # shuffle
    idx_list = list(range(0, len(flitered_labels), 1))
    random.shuffle(idx_list)
    labels = []
    contents = []
    for idp in idx_list:
        labels.append(flitered_labels[idp])
        contents.append(flitered_contents[idp])
    print('Data loaded: ', len(flitered_labels), len(flitered_contents))
    return labels, contents, label_set, label_dic


def pre_encode_dic(model_name, contents, max_length=512):
    try:
        tokenizer = RobertaTokenizer.from_pretrained(model_name)
    except OSError:
        print('OSError! can not load tokenizer!')
    else:
        print('Tokenizer loaded...')
        return tokenizer(contents,
                         add_special_tokens=True,
                         padding=True,
                         max_length=max_length,
                         truncation=True,
                         return_tensors='pt')


def pre_encode_list(model_name, contents, max_length=512):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except OSError:
        print('OSError! can not load tokenizer!')
    else:
        res = []
        for content in tqdm(contents):
            res.append(tokenizer(content,
                                 add_special_tokens=True,
                                 padding='max_length',
                                 max_length=max_length,
                                 truncation=True,
                                 return_tensors='pt'))
        print('Tokenizer loaded...')
        return res


class NewsCategoryDataset(Dataset):
    def __init__(self,
                 labels, inputs, origins, label_dic,
                 mode='train',
                 balance=[0.7, 0.15, 0.15]):
        train_num = int(len(labels) * balance[0])
        val_num = int(len(labels) * balance[1])
        test_num = int(len(labels) * balance[2])
        # choose mode
        self.label_dic = label_dic
        if mode == 'train':
            self.inputs = inputs[:train_num]
            self.origins = origins[:train_num]
            self.labels = labels[:train_num]
        elif mode == 'val':
            self.inputs = inputs[train_num:-test_num]
            self.origins = origins[train_num:-test_num]
            self.labels = labels[train_num:-test_num]
        elif mode == 'test':
            self.inputs = inputs[-test_num:]
            self.origins = origins[-test_num:]
            self.labels = labels[-test_num:]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):

        y = self.labels[idx]
        y_encoded = torch.Tensor([self.label_dic.get(y, -1)]).long().squeeze(0)
        res = {'input_ids': self.inputs[idx]['input_ids'][0],
               'attention_mask': self.inputs[idx]['attention_mask'][0],
               'origin_contents': self.origins[idx],
               'targets': y_encoded}
        return res


class BertForSequenceClassification(nn.Module):
    def __init__(self, model_name, num_classes=None):
        super().__init__()

        config = RobertaConfig.from_pretrained(model_name, num_labels=num_classes)
        self.bert = AutoModel.from_pretrained(model_name, config=config)
        self.MLP = nn.Linear(config.hidden_size, num_classes)

    def forward(self, features, attention_mask=None, head_mask=None):
        assert attention_mask is not None, 'attention_mask is none'
        bert_output = self.bert(input_ids=features,
                                attention_mask=attention_mask,
                                head_mask=head_mask)

        hidden_state = bert_output[0]

        pool_output = hidden_state[:, 0]
        # print(pool_output)
        # print(pool_output.shape)
        logits = self.MLP(pool_output)
        # logits.unsqueeze(1)
        return logits


class RobertaForSequenceClassification(nn.Module):
    def __init__(self, model_name, num_classes=None):
        super().__init__()
        config = RobertaConfig.from_pretrained(model_name, num_labels=num_classes)

        self.bert = AutoModel.from_pretrained(model_name, config=config)
        self.MLP = nn.Linear(config.hidden_size, num_classes)

    def forward(self, features, attention_mask=None, head_mask=None):
        assert attention_mask is not None, 'attention_mask is none'
        bert_output = self.bert(input_ids=features,
                                attention_mask=attention_mask,
                                head_mask=head_mask)

        hidden_state = bert_output[0]

        pool_output = hidden_state[:, 0]
        # print(pool_output)
        # print(pool_output.shape)
        logits = self.MLP(pool_output)
        # logits.unsqueeze(1)
        return logits


def weight_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.constant_(m.bias, 0)


def train_classifier(model, epoch, lr, seq, iterator, criterion, date, optimizer=None,
                     scheduler=None, path='results/'
                     ):
    loss = 0
    tokens = 0
    accs = []
    losses = []
    epoch_losses = []
    epoch_accs = []
    for i in range(epoch):
        print('epoch {}'.format(i))
        model.train()
        for batch, data in enumerate(notetqdm(iterator['train'])):
            optimizer.zero_grad()
            features = data['input_ids'].cuda()
            # print(features.shape)
            att_mask = data['attention_mask'].cuda()
            y = data['targets'].cuda()

            # print(y.shape)

            y_pred = model(features, att_mask)

            # print(y_pred)
            # print(y_pred[0].shape)

            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()

            # print('---------------------------------')

            # print('---------------------------------')

            y = y.cpu().numpy()
            # print('target: ',y)

            y_pred = y_pred.cpu().detach().numpy()
            ss = np.argmax(y_pred, axis=1)
            ss = ss.tolist()
            '''
            # print screen
            for i in range(len(data)):
                print(data['origin_contents'][i])
                print('pred: ',lset[ss[i]])
                print('true: ',lset[y[i]])
                print('--------------')
            '''
            acc = (ss == y).sum().item() / len(y)
            accs.append(acc)
            losses.append(loss.item())
            print('Epoch {} batch {}, loss: {:.4} acc: {:.6} '.format(i, batch, loss, acc * 100), end='\r')
        epoch_losses.append(np.mean(np.array(losses)))
        epoch_accs.append(np.mean(np.array(accs)))
        print()
        # vaild
        print('validating...')
        true_labels = []
        pred_outputs = []
        model.eval()
        with torch.no_grad():
            for ba, data in enumerate(iterator['valid']):
                features = data['input_ids'].to('cuda')
                att_mask = data['attention_mask'].to('cuda')
                targets = data['targets']
                targets.numpy()
                true_labels += targets.tolist()
                outputs = model(features, att_mask)
                outputs = outputs.cpu().detach().numpy()
                outputs = np.argmax(outputs, axis=1)
                pred_outputs += outputs.tolist()
                '''
                # print screen
                for i in range(len(data)):
                    print(data['origin_contents'][i])
                    print('pred: ',lset[outputs[i]])
                    print('true: ',lset[targets[i]])
                    print('--------------')
                '''

            valid_acc = sum([1 if y == p else 0 for y, p in zip(pred_outputs, true_labels)]) / len(true_labels)
            print('After Epoch {} , valid acc: {}, avg loss{}  avg acc{}'.format(i, valid_acc, epoch_losses[-1],
                                                                                 epoch_accs[-1]))
        if scheduler is not None:
            scheduler.step(loss)
        torch.save(model.module.state_dict(), '{}/{}-epoch-{}.pth'.format(path, date, i))
        # torch.cuda.empty_cache()
    return accs, losses
