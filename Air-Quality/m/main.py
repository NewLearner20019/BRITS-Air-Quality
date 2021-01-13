import copy
import torch
torch.manual_seed(1)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from datetime import datetime

import numpy as np
np.random.seed(1)

import time
import utils
import models
import argparse
import data_loader
import pandas as pd
# import ujson as json

from sklearn import metrics

#from ipdb import set_trace

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--model', type=str)
parser.add_argument('--hid_size', type=int)
parser.add_argument('--impute_weight', type=float)
#parser.add_argument('--label_weight', type=float)
args = parser.parse_args()


def train(model):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    data_iter = data_loader.get_loader(batch_size=args.batch_size)

    auroc_auprc = []

    for epoch in range(args.epochs):
        model.train()

        run_loss = 0.0

        for idx, data in enumerate(data_iter):
            data = utils.to_var(data)
            ret = model.run_on_batch(data, optimizer, epoch)

            run_loss += ret['loss'].item()

            print("\r Progress epoch {}, {:.2f}%, average loss {}".format(epoch, (idx + 1) * 100.0 / len(data_iter), run_loss / (idx + 1.0))),

        auroc_auprc.append(evaluate(model, data_iter))

 

def evaluate(model, val_iter):
    model.eval()

    #labels = []
    #preds = []

    evals = []
    imputations = []

    save_impute = []
    #save_label = []
    #MAE = []

    for idx, data in enumerate(val_iter):
        data = utils.to_var(data)
        ret = model.run_on_batch(data, None)

        # save the imputation results which is used to test the improvement of traditional methods with imputed values
        save_impute.append(ret['imputations'].data.cpu().numpy())
        #save_label.append(ret['labels'].data.cpu().numpy())

        #pred = ret['predictions'].data.cpu().numpy()
        #label = ret['labels'].data.cpu().numpy()
        #is_train = ret['is_train'].data.cpu().numpy()

        eval_masks = ret['eval_masks'].data.cpu().numpy()
        eval_ = ret['evals'].data.cpu().numpy()
        imputation = ret['imputations'].data.cpu().numpy()

        evals += eval_[np.where(eval_masks == 1)].tolist()
        imputations += imputation[np.where(eval_masks == 1)].tolist()
        
        #M = np.abs(evals - imputations).mean()

        # collect test label & prediction
        #pred = pred[np.where(is_train == 0)]
        #label = label[np.where(is_train == 0)]

        #labels += label.tolist()
        #preds += pred.tolist()

    #labels = np.asarray(labels).astype('int32')
    #preds = np.asarray(preds)

    # compute auroc and auprc
    #auroc = metrics.roc_auc_score(labels, preds)
    #auprc = metrics.average_precision_score(labels, preds)


    #print('AUROC {}'.format(auroc), 'AUPRC {}'.format(auprc))
    # Compute average precision (AP) from prediction scores This score corresponds to the area under the precision-recall curve.
    # The worst AUPRC is 0, and the best AUPRC is 1.0. This is in contrast to AUROC, where the lowest value is 0.5.
    #print('AUPRC {}'.format(metrics.average_precision_score(labels, preds)))


    #tn, fp, fn, tp = metrics.confusion_matrix(labels, rfc_pred).ravel()


    #print("=== Confusion Matrix ===")
    #print(metrics.confusion_matrix(labels, rfc_pred))
    #print("tp = " + str(tp), "fn = " + str(fn), "fp = " + str(fp), "tn = " + str(tn))


    evals = np.asarray(evals)
    imputations = np.asarray(imputations)
    MAE = np.abs(evals - imputations).mean()

    print('MAE', np.abs(evals - imputations).mean())
    print('MRE', np.abs(evals - imputations).sum() / np.abs(evals).sum())

    save_impute = np.concatenate(save_impute, axis=0)
    #save_label = np.concatenate(save_label, axis=0)

    np.save('./result/{}_data'.format(args.model), save_impute)
    #np.save('./result/{}_label'.format(args.model), save_label)
    return(MAE)


def run():
    model = getattr(models, args.model).Model(args.hid_size, args.impute_weight)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total params is {}'.format(total_params))

    if torch.cuda.is_available():
        model = model.cuda()

    train(model)


if __name__ == '__main__':
    run()
    dateTimeObj = datetime.now()
    print(dateTimeObj)

