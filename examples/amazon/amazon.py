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


sys.path.append('/raid/huaizepeng/DIN&DIEN_MIT')

from base_model.features import Number, Category, Sequence, Features
from base_model.transformers.column import (
    StandardScaler, CategoryEncoder, SequenceEncoder)

from base_model.pytorch.data import Dataset
from base_model.pytorch import WideDeep, DeepFM, DNN, DIN, DIEN, AttentionGroup

from base_model.pytorch.functions import fit, predict, create_dataloader_fn

SEQ_MAX_LEN = 100  # maximum sequence length
BATCH_SIZE = 128
EMBEDDING_DIM = 18
DNN_HIDDEN_SIZE = [200, 80]
DNN_DROPOUT = 0.0
TEST_RUN = False
EPOCH = 2
EPOCH = 200
SEED = 2021

pwd = '/raid/huaizepeng/DIN&DIEN_MIT/examples/amazon'

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

train_df = pd.read_csv(
    pwd + "/local_train.csv", sep='\t')

global valid_df
valid_df = pd.read_csv(
    pwd + "/local_test.csv", sep='\t')

if TEST_RUN:
    train_df = train_df.sample(1000)
    valid_df = valid_df.sample(1000)


def scale_eda(df):
    print(df.shape)
    print(df.uid.nunique())
    print(df.mid.nunique())
    print(df.groupby('label', as_index=False).uid.count())


scale_eda(train_df)
scale_eda(valid_df)

unique_cats = Counter(train_df.cat.values.tolist())
unique_cats_in_hist = Counter(
    itertools.chain(*train_df.hist_cats.apply(lambda x: x.split(chr(0x02))).values.tolist()))
print(len(unique_cats), len(unique_cats_in_hist),
      len(np.intersect1d(list(unique_cats.keys()), list(unique_cats_in_hist.keys()))))

# All categorys also appear in history categorys.
unique_mids = Counter(train_df.mid.values.tolist())
unique_mids_in_hist = Counter(
    itertools.chain(*train_df.hist_mids.apply(lambda x: x.split(chr(0x02))).values.tolist()))
print(len(unique_mids), len(unique_mids_in_hist),
      len(np.intersect1d(list(unique_mids.keys()), list(unique_mids_in_hist.keys()))))

# Most mids appears in history mids.
print("There are {}% mid overlap between train and valid".format(
    100 * len(np.intersect1d(train_df.mid.unique(), valid_df.mid.unique())) / len(valid_df.mid.unique())))

# define features
cat_enc = SequenceEncoder(sep="\x02", min_cnt=1, max_len=SEQ_MAX_LEN)
cat_enc.fit(train_df.hist_cats.values)
cat_word2idx, cat_idx2word = cat_enc.word2idx, cat_enc.idx2word
mid_enc = SequenceEncoder(sep="\x02", min_cnt=1, max_len=SEQ_MAX_LEN)
mid_enc.fit(np.vstack([train_df.mid.values, train_df.hist_mids.values]))
mid_word2idx, mid_idx2word = mid_enc.word2idx, mid_enc.idx2word

def evaluation(model, df, dataloader):
    preds = predict(model, dataloader)
    return roc_auc_score(df['label'], preds.ravel())

def run(models):
    scores = OrderedDict()
    model_loss_curves = OrderedDict()
    for model_name, model in models:
        print(model_name)
        loss_func = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

        # scores[model_name] = evaluation(model, valid_df, valid_loader)

        losses = fit(EPOCH, model, loss_func, optimizer,
                     train_loader, valid_loader,valid_df, notebook=True, auxiliary_loss_rate=1)
        scores[model_name] = evaluation(model, valid_df, valid_loader)
        model_loss_curves[model_name] = losses
    return scores, model_loss_curves

"""------------------------------------------all model---------------------------------------"""
# number_features = []
#
# category_features = [
#     Category('mid',
#              CategoryEncoder(min_cnt=1, word2idx=mid_word2idx, idx2word=mid_idx2word),
#              embedding_name='mid'),
#     Category('cat',
#              CategoryEncoder(min_cnt=1, word2idx=cat_word2idx, idx2word=cat_idx2word),
#              embedding_name='cat'),
# ]
#
# sequence_features = [
#     Sequence('hist_mids',
#              SequenceEncoder(sep="\x02", min_cnt=1, max_len=SEQ_MAX_LEN,
#                              word2idx=mid_word2idx, idx2word=mid_idx2word),
#              embedding_name='mid'),
#     Sequence('hist_cats',
#              SequenceEncoder(sep="\x02", min_cnt=1, max_len=SEQ_MAX_LEN,
#                              word2idx=cat_word2idx, idx2word=cat_idx2word),
#              embedding_name='cat')
# ]
#
# features, train_loader, valid_loader = create_dataloader_fn(
#     number_features, category_features, sequence_features, BATCH_SIZE, train_df, 'label', valid_df, 4)
#
# din_attention_groups = [
#     AttentionGroup(
#         name='group1',
#         pairs=[{'ad': 'mid', 'pos_hist': 'hist_mids'},
#                {'ad': 'cat', 'pos_hist': 'hist_cats'}],
#         hidden_layers=[80, 40], att_dropout=0.0)]
#
# gru_attention_groups = [
#     AttentionGroup(
#         name='group1',
#         pairs=[{'ad': 'mid', 'pos_hist': 'hist_mids'},
#                {'ad': 'cat', 'pos_hist': 'hist_cats'}],
#         hidden_layers=[80, 40], att_dropout=0.0, gru_type='GRU')]
#
# aigru_attention_groups = [
#     AttentionGroup(
#         name='group1',
#         pairs=[{'ad': 'mid', 'pos_hist': 'hist_mids'},
#                {'ad': 'cat', 'pos_hist': 'hist_cats'}],
#         hidden_layers=[80, 40], att_dropout=0.0, gru_type='AIGRU')]
#
# agru_attention_groups = [
#     AttentionGroup(
#         name='group1',
#         pairs=[{'ad': 'mid', 'pos_hist': 'hist_mids'},
#                {'ad': 'cat', 'pos_hist': 'hist_cats'}],
#         hidden_layers=[80, 40], att_dropout=0.0, gru_type='AGRU')]
#
# augru_attention_groups = [
#     AttentionGroup(
#         name='group1',
#         pairs=[{'ad': 'mid', 'pos_hist': 'hist_mids'},
#                {'ad': 'cat', 'pos_hist': 'hist_cats'}],
#         hidden_layers=[80, 40], att_dropout=0.0, gru_type='AUGRU')]
#
# models = [
#     ('DNN', DNN(features, 2, EMBEDDING_DIM, DNN_HIDDEN_SIZE,
#                 final_activation='sigmoid', dropout=DNN_DROPOUT)),
#
#     ('WideDeep', WideDeep(features,
#                           wide_features=['mid', 'hist_mids', 'cat', 'hist_cats'],
#                           deep_features=['mid', 'hist_mids', 'cat', 'hist_cats'],
#                           cross_features=[('mid', 'hist_mids'), ('cat', 'hist_cats')],
#                           num_classes=2, embedding_size=EMBEDDING_DIM, hidden_layers=DNN_HIDDEN_SIZE,
#                           final_activation='sigmoid', dropout=DNN_DROPOUT)),
#
#     ('DeepFM', DeepFM(features, 2, EMBEDDING_DIM, DNN_HIDDEN_SIZE,
#                       final_activation='sigmoid', dropout=DNN_DROPOUT)),
#
#     ('DIN', DIN(features, din_attention_groups, 2, EMBEDDING_DIM, DNN_HIDDEN_SIZE,
#                 final_activation='sigmoid', dropout=DNN_DROPOUT)),
#
#     ('DIEN_gru', DIEN(features, gru_attention_groups, 2, EMBEDDING_DIM, DNN_HIDDEN_SIZE,
#                       final_activation='sigmoid', dropout=DNN_DROPOUT)),
#
#     ('DIEN_aigru', DIEN(features, aigru_attention_groups, 2, EMBEDDING_DIM, DNN_HIDDEN_SIZE,
#                         final_activation='sigmoid', dropout=DNN_DROPOUT)),
#
#     ('DIEN_agru', DIEN(features, agru_attention_groups, 2, EMBEDDING_DIM, DNN_HIDDEN_SIZE,
#                        final_activation='sigmoid', dropout=DNN_DROPOUT)),
#
#     ('DIEN_augru', DIEN(features, augru_attention_groups, 2, EMBEDDING_DIM, DNN_HIDDEN_SIZE,
#                         final_activation='sigmoid', dropout=DNN_DROPOUT))
# ]
#
# scores1, model_loss_curves1 = run(models)

# '''---------------------------------just dien------------------------------------'''
# number_features = []
#
# category_features = [
#     Category('mid',
#              CategoryEncoder(min_cnt=1, word2idx=mid_word2idx, idx2word=mid_idx2word),
#              embedding_name='mid'),
#     Category('cat',
#              CategoryEncoder(min_cnt=1, word2idx=cat_word2idx, idx2word=cat_idx2word),
#              embedding_name='cat'),
# ]
#
# sequence_features = [
#     Sequence('hist_mids',
#              SequenceEncoder(sep="\x02", min_cnt=1, max_len=SEQ_MAX_LEN,
#                              word2idx=mid_word2idx, idx2word=mid_idx2word),
#              embedding_name='mid'),
#     Sequence('hist_cats',
#              SequenceEncoder(sep="\x02", min_cnt=1, max_len=SEQ_MAX_LEN,
#                              word2idx=cat_word2idx, idx2word=cat_idx2word),
#              embedding_name='cat'),
#     Sequence('neg_hist_mids',
#              SequenceEncoder(sep="\x02", min_cnt=1, max_len=SEQ_MAX_LEN,
#                              word2idx=mid_word2idx, idx2word=mid_idx2word),
#              embedding_name='mid'),
#     Sequence('neg_hist_cats',
#              SequenceEncoder(sep="\x02", min_cnt=1, max_len=SEQ_MAX_LEN,
#                              word2idx=cat_word2idx, idx2word=cat_idx2word),
#              embedding_name='cat')
# ]
#
# features, train_loader, valid_loader = create_dataloader_fn(
#     number_features, category_features, sequence_features, BATCH_SIZE, train_df, 'label', valid_df, 0)
#
# augru_attention_groups_with_neg = [
#     AttentionGroup(
#         name='group1',
#         pairs=[{'ad': 'mid', 'pos_hist': 'hist_mids', 'neg_hist': 'neg_hist_mids'},
#                {'ad': 'cat', 'pos_hist': 'hist_cats', 'neg_hist': 'neg_hist_cats'}],
#         hidden_layers=[80, 40], att_dropout=0.0, gru_type='AUGRU')]
#
# din_attention_groups = [
#     AttentionGroup(
#         name='group1',
#         pairs=[{'ad': 'mid', 'pos_hist': 'hist_mids'},
#                {'ad': 'cat', 'pos_hist': 'hist_cats'}],
#         hidden_layers=[80, 40], att_dropout=0.0)]
#
# models = [
#     ('DIEN', DIEN(features, augru_attention_groups_with_neg, 2, EMBEDDING_DIM, DNN_HIDDEN_SIZE,
#                   final_activation='sigmoid', dropout=DNN_DROPOUT, use_negsampling=True))
# ]
#
# scores2, model_loss_curves2 = run(models)

'''---------------------------------just DIN------------------------------------'''
number_features = []

category_features = [
    Category('mid',
             CategoryEncoder(min_cnt=1, word2idx=mid_word2idx, idx2word=mid_idx2word),
             embedding_name='mid'),
    Category('cat',
             CategoryEncoder(min_cnt=1, word2idx=cat_word2idx, idx2word=cat_idx2word),
             embedding_name='cat'),
]

sequence_features = [
    Sequence('hist_mids',
             SequenceEncoder(sep="\x02", min_cnt=1, max_len=SEQ_MAX_LEN,
                             word2idx=mid_word2idx, idx2word=mid_idx2word),
             embedding_name='mid'),
    Sequence('hist_cats',
             SequenceEncoder(sep="\x02", min_cnt=1, max_len=SEQ_MAX_LEN,
                             word2idx=cat_word2idx, idx2word=cat_idx2word),
             embedding_name='cat')
]

features, train_loader, valid_loader = create_dataloader_fn(
    number_features, category_features, sequence_features, BATCH_SIZE, train_df, 'label', valid_df, 0)

din_attention_groups = [
    AttentionGroup(
        name='group1',
        pairs=[{'ad': 'mid', 'pos_hist': 'hist_mids'},
               {'ad': 'cat', 'pos_hist': 'hist_cats'}],
        hidden_layers=[80, 40], att_dropout=0.0)]

models = [
    ('DIN', DIN(features, din_attention_groups, 2, EMBEDDING_DIM, DNN_HIDDEN_SIZE,
                     final_activation='sigmoid', dropout=DNN_DROPOUT))
]

scores2, model_loss_curves2 = run(models)
