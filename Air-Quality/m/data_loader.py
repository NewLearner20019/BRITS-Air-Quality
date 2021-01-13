import os
import time

import ujson as json
#import simplejson as json
#import json
import numpy as np
np.random.seed(1)
import pandas as pd

import torch
torch.manual_seed(1)
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class MySet(Dataset):
    def __init__(self):
        super(MySet, self).__init__()
        self.content = open('./json/json').readlines()

        indices = np.arange(len(self.content))
        # TODO instead of np.random we separate indices of set-a as a training set and set-b as a test set
        # set seed for np.random
        # np.random.seed(1)
        val_indices = np.random.choice(indices, len(self.content) // 5)
        #val_indices = np.arange(4880, len(self.content)) # TODO there are 1108 items in set-a after undersampling
                                                         # there are 3997 items in the original set-a
        print(len(val_indices))

        self.val_indices = set(val_indices.tolist())


    def __len__(self):
        return len(self.content)

    def __getitem__(self, idx):
        rec = json.loads(self.content[idx])
        if idx in self.val_indices:
            rec['is_train'] = 0
        else:
            rec['is_train'] = 1
        return rec

def collate_fn(recs):
    forward = list(map(lambda x: x['forward'], recs))
    backward = list(map(lambda x: x['backward'], recs))

    def to_tensor_dict(recs):
        values = torch.FloatTensor(list(map(lambda x: x['values'], recs)))
        masks = torch.FloatTensor(list(map(lambda x: x['masks'], recs)))
        deltas = torch.FloatTensor(list(map(lambda x: x['deltas'], recs)))
        forwards = torch.FloatTensor(list(map(lambda x: x['forwards'], recs)))

        evals = torch.FloatTensor(list(map(lambda x: x['evals'], recs)))
        eval_masks = torch.FloatTensor(list(map(lambda x: x['eval_masks'], recs)))

        return {'values': values, 'forwards': forwards, 'masks': masks, 'deltas': deltas, 'evals': evals, 'eval_masks': eval_masks}

    ret_dict = {'forward': to_tensor_dict(forward), 'backward': to_tensor_dict(backward)}

    #ret_dict['labels'] = torch.FloatTensor(list(map(lambda x: x['label'], recs)))
    ret_dict['is_train'] = torch.FloatTensor(list(map(lambda x: x['is_train'], recs)))

    return ret_dict


def get_loader(batch_size = 64, shuffle = True):
    data_set = MySet()
    data_iter = DataLoader(dataset = data_set, \
                              batch_size = batch_size, \
                              num_workers = 4, \
                              shuffle = shuffle, \
                              pin_memory = True, \
                              collate_fn = collate_fn
    )

    return data_iter
