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

def load_data(path, cate_size,label_set, type='JSON', percentage=1):
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

def load_test_data(path,cate_size,type='JSON',percentage=1):
    # if already combined
    if type == "JSON":
        df = pd.read_json(path)
    else:
        df = pd.read_csv(path)
    df = df.iloc[np.random.permutation(len(df))]
    labels = df['category'].tolist()
    contents = df['content'].tolist()
    label_dic = {}
    label_count = {}
    final_size = int(len(contents) * percentage)
    if percentage != 1:
        contents = contents[:final_size]
        labels = labels[:final_size]

    # itos
    label_set = ['economia',  # economic
              'internacional',  # international
              'deportes',  # sports
              'cultura',  # culture
              'television',  # television
              'ciencia-y-salud',  # science and health
              'tecnologia',  # technology
               ]
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
        self.MLP.apply(weight_init)

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
        self.config = RobertaConfig.from_pretrained(model_name, num_labels=num_classes)
        self.model_name = model_name
        self.bert = AutoModel.from_pretrained(model_name, config=self.config)
        self.MLP = nn.Linear(self.config.hidden_size, num_classes)

    def load_dict(self,source,source_num,keep_mlp):
        source_model = RobertaForSequenceClassification(self.model_name,source_num)
        st_dict = torch.load(source)
        source_model.load_state_dict(st_dict)
        self.bert = source_model.bert
        if keep_mlp:
            self.MLP = source_model.MLP

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

'''
class FurtherPretrainClassifier(nn.Module):
    def __init__(self, model_name, source, target_num):
        super().__init__()
        config = RobertaConfig.from_pretrained(model_name, num_labels=target_num)
        self.model = RobertaForSequenceClassification(model_name, 26)
        state_dict = torch.load(source)
        self.model.load_state_dict(state_dict)
        print(self.model.MLP)
        self.model.MLP = nn.Linear(config.hidden_size, target_num)
        self.model.MLP.apply(weight_init)
        self.bert = self.model.bert
        self.MLP = self.model.MLP

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

class FurtherClassifier(nn.Module):
    def __init__(self, model_name, source_num, target_num):
        super().__init__()
        config = RobertaConfig.from_pretrained(model_name, num_labels=target_num)
        self.model = RobertaForSequenceClassification(model_name, source_num)
        self.model.MLP = nn.Linear(config.hidden_size, target_num)
        self.model.MLP.apply(weight_init)

    def forward(self, features, attention_mask=None, head_mask=None):
        assert attention_mask is not None, 'attention_mask is none'
        bert_output = self.model.bert(input_ids=features,
                                      attention_mask=attention_mask,
                                      head_mask=head_mask)

        hidden_state = bert_output[0]

        pool_output = hidden_state[:, 0]
        logits = self.model.MLP(pool_output)
        return logits
'''

def weight_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.constant_(m.bias, 0)


def train_classifier(model, epoch, lr, seq, iterator, criterion, date, optimizer=None,
                     scheduler=None, path=None
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
            y_pred = model(features, att_mask)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
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
        if path:
            torch.save(model.module.state_dict(), '{}/{}-epoch-{}.pth'.format(path, date, i))
        # torch.cuda.empty_cache()
    return accs, losses


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice


def para_compare(MODEL_NAME, pmodel, cmodels, criterion, cates):
    raw = RobertaForSequenceClassification(MODEL_NAME, 26).bert
    # Newscate pretrained model
    models = []
    pm = RobertaForSequenceClassification(MODEL_NAME, 26)
    pm.load_dict(pmodel, 26, True)
    models.append(raw)
    models.append(pm.bert)

    for idx in range(len(cmodels)):
        cmodel = RobertaForSequenceClassification(MODEL_NAME, cates[idx])
        cmodel.load_dict(cmodels[idx],cates[idx],True)
        models.append(cmodel.bert)
        print('compare model {} loaded'.format(idx))

    layer_num = len(raw.encoder.layer)
    print('models loaded')
    model_num = len(models)
    paras = [[] for _ in range(model_num)]

    for idx in range(model_num):
        for i in range(layer_num):
            layer_para = []
            att = models[idx].encoder.layer[i].attention
            para_query = list(att.self.query.parameters())
            para_key = list(att.self.key.parameters())
            para_value = list(att.self.value.parameters())
            layer_para.append(para_query[0].data)
            layer_para.append(para_query[1].data)
            layer_para.append(para_key[0].data)
            layer_para.append(para_key[1].data)
            layer_para.append(para_value[0].data)
            layer_para.append(para_value[1].data)
            paras[idx].append(layer_para)


    print("paras-size: {} * {} * {}: ".format(len(paras), len(paras[0]), len(paras[0][0])))

    matrix_loss = [[] for _ in range(model_num)]
    bias_loss = [[] for _ in range(model_num)]
    # criterion Loss
    for idx in range(layer_num):
        # print('for Layer {}:'.format(idx))
        for i in range(1, model_num):
            qml = criterion(paras[0][idx][0], paras[i][idx][0])
            qbl = criterion(paras[0][idx][1], paras[i][idx][1])
            kml = criterion(paras[0][idx][2], paras[i][idx][2])
            kbl = criterion(paras[0][idx][3], paras[i][idx][3])
            vml = criterion(paras[0][idx][4], paras[i][idx][4])
            vbl = criterion(paras[0][idx][5], paras[i][idx][5])
            matrix_loss[i].append([qml, kml, vml])
            bias_loss[i].append([qbl, kbl, vbl])
        '''
            print('Query Matrix Loss 1:{}'.format(qml1))
            print('Query Matrix Loss 2:{}'.format(qml2))
            print('  Query Bias Loss 1:{}'.format(qbl1))
            print('  Query Bias Loss 2:{}'.format(qbl2))
            print('  Key Matrix Loss 1:{}'.format(kml1))
            print('  Key Matrix Loss 2:{}'.format(kml2))
            print('    Key Bias Loss 1:{}'.format(kbl1))
            print('    Key Bias Loss 2:{}'.format(kbl2))
            print('Value Matrix Loss 1:{}'.format(vml1))
            print('Value Matrix Loss 2:{}'.format(vml2))
            print('  Value Bias Loss 1:{}'.format(vbl1))
            print('  Value Bias Loss 2:{}'.format(vbl2))
            print("Average Matrix Loss:{} - {}".format((qml1 + kml1 + vml1) / 3, (qml2 + kml2 + vml2) / 3))
            print("  Average Bias Loss:{} - {}".format((qbl1 + kbl1 + vbl1) / 3, (qbl2 + kbl2 + vbl2) / 3))
        '''
    return matrix_loss, bias_loss





    
    
