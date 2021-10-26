import itertools
from collections import Counter, OrderedDict

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import sys
import pickle
import copy
from shutil import copyfile
import os

sys.path.append('/home/hzp/CTRmodel')

from base_model.features import Number, Category, Sequence, Features
from base_model.transformers.column import (
    StandardScaler, CategoryEncoder, SequenceEncoder)

from base_model.pytorch.data import Dataset
from base_model.pytorch import WideDeep, DeepFM, DNN, DIN, DIEN, AttentionGroup
from base_model.pytorch.functions import fit, predict, create_dataloader_fn

pwd = '/home/hzp/CTRmodel/examples/aminer/'
with open(pwd + 'dataset.pkl', 'rb') as f:
    train_set = pickle.load(f)
    test_set = pickle.load(f)
    user_count, item_count = pickle.load(f)
    asin_map, revi_map = pickle.load(f)
asin_map_1 = dict(zip(list(asin_map.values()), list(asin_map.keys())))
revi_map_1 = dict(zip(list(revi_map.values()), list(revi_map.keys())))

SEQ_MAX_LEN = 100  # maximum sequence length
BATCH_SIZE = 128
EMBEDDING_DIM = 18
DNN_HIDDEN_SIZE = [200, 80]
DNN_DROPOUT = 0.0
TEST_RUN = False
EPOCH = 2
EPOCH = 2
SEED = 2021

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

train_df = pd.read_csv(
    pwd + "/local_train.csv", sep='\t')

valid_df = pd.read_csv(
    pwd + "/local_test.csv", sep='\t')

if TEST_RUN:
    train_df = train_df.sample(1000)
    valid_df = valid_df.sample(1000)


def scale_eda(df):
    print(df.shape)
    print(df.user.nunique())
    print(df.item.nunique())
    print(df.groupby('label', as_index=False).user.count())


scale_eda(train_df)
scale_eda(valid_df)

# All categorys also appear in history categorys.
unique_mids = Counter(train_df.item.values.tolist())
unique_mids_in_hist = Counter(
    itertools.chain(*train_df.hist_item.apply(lambda x: x.split('\t')).values.tolist()))
print(len(unique_mids), len(unique_mids_in_hist),
      len(np.intersect1d(list(unique_mids.keys()), list(unique_mids_in_hist.keys()))))

# Most mids appears in history mids.
print("There are {}% mid overlap between train and valid".format(
    100 * len(np.intersect1d(train_df.item.unique(), valid_df.item.unique())) / len(valid_df.item.unique())))


# # define features
# cat_enc = SequenceEncoder(sep='\x02', min_cnt=1, max_len=SEQ_MAX_LEN)
# cat_enc.fit(train_df.hist_cat.values)
# cat_word2idx, cat_idx2word = cat_enc.word2idx, cat_enc.idx2word
# mid_enc = SequenceEncoder(sep='\x02', min_cnt=1, max_len=SEQ_MAX_LEN)
# mid_enc.fit(np.vstack([train_df.item.values, train_df.hist_item.values]))
# mid_word2idx, mid_idx2word = mid_enc.word2idx, mid_enc.idx2word

def evaluation(model, df, dataloader):
    preds = predict(model, dataloader)
    return roc_auc_score(df['label'], preds.ravel())


def run(models):
    scores = OrderedDict()
    model_loss_curves = OrderedDict()
    for model_name, model in models:
        print(model_name)
        loss_func = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
        # optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

        # scores[model_name] = evaluation(model, valid_df, valid_loader)

        losses = fit(EPOCH, model, loss_func, optimizer,
                     train_loader, valid_loader, valid_df, notebook=True, auxiliary_loss_rate=1)
        scores[model_name] = evaluation(model, valid_df, valid_loader)
        model_loss_curves[model_name] = losses

        # save model
        model_save = copy.deepcopy(model).cpu()
        torch.save(model_save,
                   '/home/hzp/CTRmodel/model_serve/model_{}.pth'.format(model_name))
        print('----------------------save end----------------------------')
        os.system('rm -rf /home/hzp/CTRmodel/base_model_0/base_model')
        os.system('cp -r /home/hzp/CTRmodel/base_model/ /home/hzp/CTRmodel/base_model_0/')

    return scores, model_loss_curves


'''---------------------------------just DIN------------------------------------'''
number_features = []

category_features = [
    Category('item',
             CategoryEncoder(min_cnt=1, word2idx=asin_map, idx2word=asin_map_1),
             embedding_name='item')
]

sequence_features = [
    Sequence('hist_item',
             SequenceEncoder(sep='\t', min_cnt=1, max_len=SEQ_MAX_LEN,
                             word2idx=asin_map, idx2word=asin_map_1),
             embedding_name='item')
]

features, train_loader, valid_loader = create_dataloader_fn(
    number_features, category_features, sequence_features, BATCH_SIZE, train_df, 'label', valid_df, 0)

din_attention_groups = [
    AttentionGroup(
        name='group1',
        pairs=[{'ad': 'item', 'pos_hist': 'hist_item'}],
        hidden_layers=[80, 40], att_dropout=0.2)]

models = [
    ('DIN', DIN(features, din_attention_groups, 2, EMBEDDING_DIM, DNN_HIDDEN_SIZE,
                final_activation='sigmoid', dropout=DNN_DROPOUT, dnn_activation='sigmoid'))
]

scores2, model_loss_curves2 = run(models)
