import itertools
from collections import Counter

import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import pickle
# /home/hzp/CTRmodel
# os.path.join('/raid/huaizepeng/DIN&DIEN_MIT/examples/amazon_e')

# pwd = '/raid/huaizepeng/DIN&DIEN_MIT/examples/amazon_e/'
# pwd1 = '/raid/huaizepeng/DIN&DIEN_MIT/examples/amazon/'

pwd = '/home/hzp/CTRmodel/examples/amazon_e/'
pwd1 = '/home/hzp/CTRmodel/examples/amazon/'

with open(pwd + 'dataset.pkl', 'rb') as f:
    train_set = pickle.load(f)
    test_set = pickle.load(f)
    cate_list = pickle.load(f)
    user_count, item_count, cate_count = pickle.load(f)
    asin_map, cate_map, revi_map = pickle.load(f)
    item2cat = pickle.load(f)

cate_map_1 = dict(zip(list(cate_map.values()), list(cate_map.keys())))
asin_map_1 = dict(zip(list(asin_map.values()), list(asin_map.keys())))
revi_map_1 = dict(zip(list(revi_map.values()), list(revi_map.keys())))
item2cat_1 = dict(zip(list(item2cat.values()), list(item2cat.keys())))
"""-----------将训练样本组合成pandas---------------"""
train_df = pd.DataFrame(columns=('label', 'user', 'item', 'cat', 'hist_item', 'hist_cat'))
train_set_dict_all = []
for batch in tqdm(train_set):
    train_set_dict = {}
    train_set_dict['label'] = batch[3]
    train_set_dict['user'] = revi_map_1[batch[0]]
    train_set_dict['item'] = asin_map_1[batch[2]]
    train_set_dict['cat'] = cate_map_1[cate_list[batch[2]]]
    train_set_dict['hist_item'] = '\t'.join([asin_map_1[ii] for ii in batch[1]])
    train_set_dict['hist_cat'] = '\t'.join([cate_map_1[cate_list[ii]] for ii in batch[1]])
    train_set_dict_all.append(train_set_dict)
train_df = train_df.append(train_set_dict_all, ignore_index=True)

valid_df = pd.DataFrame(columns=('label', 'user', 'item', 'cat', 'hist_item', 'hist_cat'))
train_set_dict_all = []
for batch in tqdm(test_set):
    train_set_dict = {}
    train_set_dict['label'] = 1
    train_set_dict['user'] = revi_map_1[batch[0]]
    train_set_dict['item'] = asin_map_1[batch[2][0]]
    train_set_dict['cat'] = cate_map_1[cate_list[batch[2][0]]]
    train_set_dict['hist_item'] = '\t'.join([asin_map_1[ii] for ii in batch[1]])
    train_set_dict['hist_cat'] = '\t'.join([cate_map_1[cate_list[ii]] for ii in batch[1]])
    train_set_dict_all.append(train_set_dict)

    train_set_dict = {}
    train_set_dict['label'] = 0
    train_set_dict['user'] = revi_map_1[batch[0]]
    train_set_dict['item'] = asin_map_1[batch[2][1]]
    train_set_dict['cat'] = cate_map_1[cate_list[batch[2][1]]]
    train_set_dict['hist_item'] = '\t'.join([asin_map_1[ii] for ii in batch[1]])
    train_set_dict['hist_cat'] = '\t'.join([cate_map_1[cate_list[ii]] for ii in batch[1]])
    train_set_dict_all.append(train_set_dict)

valid_df = valid_df.append(train_set_dict_all, ignore_index=True)

TEST_RUN = False

# train_df = pd.read_csv(
#     pwd1 + "data/local_train_splitByUser", sep='\t',
#     names=['label', 'uid', 'mid', 'cat', 'hist_mids', 'hist_cats'])
#
# valid_df = pd.read_csv(
#     pwd1 + "data/local_test_splitByUser", sep='\t',
#     names=['label', 'uid', 'mid', 'cat', 'hist_mids', 'hist_cats'])

if TEST_RUN:
    train_df = train_df.sample(1000)
    valid_df = valid_df.sample(1000)


def prepare_neg(df):
    # records = df['hist_mids'].apply(lambda x: x.split(chr(0x02)))
    records = df['hist_item'].apply(lambda x: x.split('\t'))
    # candidates = list(mid_cat_map.keys())
    candidates = list(asin_map.keys())
    max_len = len(candidates)

    def neg_sampling(filters, length):
        mids = []
        cats = []
        for i in range(length):
            while (1):
                # c = candidates[np.random.randint(0, max_len)]
                c1 = np.random.randint(0, max_len)
                c = candidates[c1]
                if c not in filters:
                    mids.append(c)
                    # cats.append(mid_cat_map[c])
                    cats.append(cate_map_1[item2cat[c1]])
                    filters.add(c)
                    break
        return mids, cats

    total_neg_mids = []
    total_neg_cats = []
    for record in tqdm(records):
        neg_mids, neg_cats = neg_sampling(set(record), len(record))
        total_neg_mids.append('\t'.join(neg_mids))
        total_neg_cats.append('\t'.join(neg_cats))

    return total_neg_mids, total_neg_cats


total_neg_mids, total_neg_cats = prepare_neg(train_df)
train_df['neg_hist_item'] = total_neg_mids
train_df['neg_hist_cat'] = total_neg_cats

total_neg_mids, total_neg_cats = prepare_neg(valid_df)
valid_df['neg_hist_item'] = total_neg_mids
valid_df['neg_hist_cat'] = total_neg_cats

train_df.to_csv(pwd+'local_train.csv', sep='\t', index=False)
valid_df.to_csv(pwd+'local_test.csv', sep='\t', index=False)

print('end')
