# coding: utf-8

import os
import re
import numpy as np
import pandas as pd
import ujson as json
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv), data manipulation as in SQL
from datetime import datetime, timedelta
from pandas import DataFrame
import io
import matplotlib.pyplot as plt # this is used for the plot the graph 
import json


def shift(arr, num, fill_value=np.nan):
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)
                   


#fs = open('./json/train', 'w')
#ts = open('./json/test', 'w')
js = open('./json/test', 'w')

def parse_delta(masks, dir_):
    if dir_ == 'backward':
        masks = masks[::-1]

    deltas = []
    print('\n------------',len(masks))
    for h in range(len(masks)):
        if h == 0:
            deltas.append(np.ones(36))
        else:
            deltas.append(np.ones(36) + (1 - masks[h]) * deltas[-1])

    return np.array(deltas)


def parse_rec(values, masks, evals, eval_masks, dir_):
    deltas = parse_delta(masks, dir_)

    # only used in GRU-D
    # forwards = pd.DataFrame(values).fillna(method='ffill').fillna(0.0).as_matrix()
    forwards = pd.DataFrame(values).fillna(method='ffill').fillna(0.0).to_numpy()

    rec = {}

    rec['values'] = np.nan_to_num(values).tolist()
    rec['masks'] = masks.astype('int32').tolist()
    # imputation ground-truth
    rec['evals'] = np.nan_to_num(evals).tolist()
    rec['eval_masks'] = eval_masks.astype('int32').tolist()
    rec['forwards'] = forwards.tolist()
    rec['deltas'] = deltas.tolist()

    return rec


def train_data(sample,n):
    
    size = 36*36
    np.random.seed(n)
    r = np.random.randint(0, sample.shape[0]-size+1)
    df = sample[r:r+size].copy().reset_index()
    print(df)
    ar = df["value"].to_numpy()
    NUL = df["New_value"].to_numpy()
    values = ar.copy()
    shp = (36, 36)
    if df.date.isin([3,6,9,12]).any():
        indices = np.where(~np.isnan(NUL))[0].tolist()
        values[indices] = shift(values[indices], -1)
    else:
        pass
        #indices = np.where(~np.isnan(ar))[0].tolist()
        #np.random.seed(1)
        #indices = np.random.choice(indices, len(indices) // 10)
        #values[indices] = np.nan
 

    masks = ~np.isnan(values)
    eval_masks = (~np.isnan(values)) ^ (~np.isnan(ar))
   

    evals = ar.reshape(shp)
    values = values.reshape(shp)

    masks = masks.reshape(shp)
    eval_masks = eval_masks.reshape(shp)

    rec = {}

    # prepare the model for both directions
    rec['forward'] = parse_rec(values, masks, evals, eval_masks, dir_='forward')
    rec['backward'] = parse_rec(values[::-1], masks[::-1], evals[::-1], eval_masks[::-1], dir_='backward')

    #rec = json.dumps(rec)
    rec = json.dumps(rec,  cls=NpEncoder)

    js.write(rec + '\n')

def test_data(sample, n):

    size = 36*36
    np.random.seed(n)
    r = np.random.randint(0, sample.shape[0]-size+1)
    df = sample[r:r+size].copy().reset_index()
    print(df)
    NUL = df["New_value"].to_numpy()
    #print(NUL)
    ar = df["value"].to_numpy()
    values = ar.copy()
    shp = (36, 36)

    indices = np.where(~np.isnan(NUL))[0].tolist()
    values[indices] = np.nan


    masks = ~np.isnan(values)
    eval_masks = (~np.isnan(values)) ^ (~np.isnan(ar))
   

    evals = ar.reshape(shp)
    values = values.reshape(shp)

    masks = masks.reshape(shp)
    eval_masks = eval_masks.reshape(shp)

    rec = {}

    # prepare the model for both directions
    rec['forward'] = parse_rec(values, masks, evals, eval_masks, dir_='forward')
    rec['backward'] = parse_rec(values[::-1], masks[::-1], evals[::-1], eval_masks[::-1], dir_='backward')

    #rec = json.dumps(rec)
    rec = json.dumps(rec,  cls=NpEncoder)

    #ts.write(rec + '\n')
    js.write(rec + '\n')

data = pd.read_csv('./SampleData/so2_ground.txt')
data["date"] = pd.to_datetime(data['datetime']).dt.month
data['datehour'] = pd.to_datetime(data['datetime']).dt.hour
data = data.sort_values(by=["datehour"])
df = data.iloc[:, 0:37].copy().set_index('datetime').unstack().reset_index()
d = df.sort_values(by=['datetime'])
df = d.copy()

df.columns = ["Station", "datetime", "value"]
ar = df["value"].to_numpy()
df['date'] = pd.to_datetime(df['datetime']).dt.month
df['datehour'] = pd.to_datetime(df['datetime']).dt.hour
df['day'] = pd.to_datetime(df['datetime']).dt.day
df1 = df.copy().sort_values(by="datetime")


condition = df1.groupby(["datehour", "day"], as_index=False).apply(lambda x: x.loc[(x.date.isin([2, 5, 8,11]) &\
                                        (x.value.isnull()))]).reset_index().sort_values("datetime")
condition["value"] = -1
condition["date"] = condition["date"]+1
#print(condition["date"].unique())
test = df1[df1.date.isin([3,6,9,12])]
test_new = pd.merge(test.iloc[:, 0:7], condition[["date", "day", "datehour", "value", "Station"]],
         on=["Station", "datehour", "day","date"], how="left")

test_new.columns = ["Station", "datetime", "value", "date", "datehour", "day", "New_value"]

condition2 = df1.groupby(["datehour", "day"], as_index=False).apply(lambda x: x.loc[(x.date.isin([3, 6, 9,12]) &\
                                        (x.value.isnull()))]).reset_index().sort_values("datetime")

condition2["value"] = -1
condition2["date"] = condition["date"]-1
train = df1[~df1.date.isin([3,6,9,12])]
train_new = pd.merge(train.iloc[:, 0:7], condition2[["date", "day", "datehour", "value", "Station"]],
         on=["date", "Station", "datehour", "day"], how="left")
train_new.columns = ["Station", "datetime", "value", "date", "datehour", "day", "New_value"]
#dat = pd.concat([train, test_new])
dat = pd.concat([train_new, test_new])
dat = dat.sort_values(by=["datetime"])
n_train=300
#n_test = 100

for n in range(n_train):
    print('Processing sample {}'.format(n))
    try:
        train_data(dat, n)

    except Exception as e:
        print(e)
        continue

js.close() 
'''  
fs.close()

for n in range(n_test):
    print('Processing sample {}'.format(n))
    try:
        
        test_data(test_new, n)

    except Exception as e:
        print(e)
        continue

ts.close()
'''
