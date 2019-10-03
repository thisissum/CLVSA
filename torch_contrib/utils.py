from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import torch
import json

class SentenceReader(object):
    """A memory saving method using for-loop to read corpus
    args:
        path: str, path of corpus
        func: function, function to preprocess one line.
    """

    def __init__(self, path, func=None):
        self.path = path
        self.func = func

    def __iter__(self):
        if self.func is None:
            with open(self.path, 'r') as f:
                for sentence in f:
                    yield sentence
        else:
            with open(self.path, 'r') as f:
                for sentence in f:
                    yield self.func(sentence)


class BatchSentenceReader(object):
    """A memory saving method using for-loop to read corpus with specific batch
    args:
        path: str, path of corpus
        batch_size: int
    """
    def __init__(self, path, batch_size, encoding='utf-8'):
        self.path = path
        self.batch_size = batch_size
        self.encoding = encoding

    def __iter__(self):
        with open(self.path, 'r', encoding=self.encoding) as f:
            i = 0
            sentence_list = []
            for sentence in f:
                if i % self.batch_size == 0 and i != 0:
                    yield sentence_list
                    sentence_list = []
                i += 1
                sentence_list.append(sentence)
            yield sentence_list



class FlexibleDataset(Dataset):
    """A torch Dataset that support multi input
    """

    def __init__(self, *args):
        super(FlexibleDataset, self).__init__()
        self.args_num = len(args)
        if self.args_num == 1:
            self.inputs = args[0]
        else:
            self.inputs = [i for i in args]

    def __len__(self):
        if self.args_num == 1:
            return len(self.inputs)
        else:
            return len(self.inputs[0])

    def __getitem__(self, index):
        if self.args_num == 1:
            output = self.inputs[index]
        else:
            output = [data[index] for data in self.inputs]
        return output


def build_dataloader(*args, batch_size=64):
    """Using FlexibleDataset to build a DataLoader with specific batch_size
    args:
        *args: multi input tensor, data to load
        batch_size: int.
    """
    fdataset = FlexibleDataset(*args)
    dataloader = DataLoader(fdataset, batch_size=batch_size)
    return dataloader


class Coach(object):

    def __init__(self, train_config):

        self.train_config = train_config

    def train(self, model, train_dataloader, cv_dataloader=None):

        optimizer = self.train_config['optimizer']
        criterion = self.train_config['criterion']
        for epoch in range(self.train_config['epochs']):
            print(str(epoch+1) + '/' + str(self.train_config['epochs']))
            losses = []
            for seq, label in tqdm(train_dataloader):
                seq = seq.to(self.train_config['device'])
                label = label.to(self.train_config['device'])
                
                y_pred = model(seq)
                loss = criterion(y_pred, label)
                losses.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print('loss: ',np.mean(losses))

            if cv_dataloader is not None:
                self.validate(model,cv_dataloader)

        torch.save(model,'./models_weight/'+self.train_config['model_name'])
        print('done')

    def predict(self, model, test_dataloader):

        pred = []
        with torch.no_grad():
            for cvx in test_dataloader:
                cvx = cvx[0].to(self.train_config['device'])
                output = model(cvx)
                pred.append(output)

        pred = torch.cat(pred, dim=0)
        return pred
    
    def validate(self, model, cv_loader):
        losses = []
        errors = 0
        criterion = self.train_config['criterion']
        with torch.no_grad():
            for cvx, cvy in cv_loader:
                cvx = cvx.to(self.train_config['device'])
                cvy = cvy.to(self.train_config['device'])
                y_pred = model(cvx)
                loss = criterion(y_pred, cvy)
                losses.append(loss.item())
                ans = model.decode(model.compute_emission(cvx))
                errors += (ans != cvy).sum().item()

        print(np.mean(losses))
        print(errors)



def load_json(path, encoding='utf-8'):
    with open(path,'r',encoding=encoding) as f:
        data = json.load(f)
    return data


def save_json(path, data):
    with open(path,'w') as f:
        json.dump(data, f)


def load_npy(path):
    output = np.load(path)
    return output


def save_npy(path, data):
    np.save(path, data)



def pad_sequence(ids, padding=0, length=None):
    """
    """
    if length is None:
        length = max(map(lambda x:len(x), ids))
    
    for i, line in enumerate(ids):
        if len(line) > length:
            ids[i] = line[:length]
        elif len(line) < length:
            dif = length - len(line) 
            ids[i] = line + dif * [padding]
        else:
            pass
    
    return ids