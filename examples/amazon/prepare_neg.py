import itertools
from collections import Counter

import numpy as np
import pandas as pd
import os
from tqdm import tqdm

os.path.join('/raid/huaizepeng/DIN&DIEN_MIT/examples/amazon')
pwd = '/raid/huaizepeng/DIN&DIEN_MIT/examples/amazon'
# os.system('wget --no-check-certificate https://raw.githubusercontent.com/mouna99/dien/master/data.tar.gz')
# os.system('wget --no-check-certificate https://raw.githubusercontent.com/mouna99/dien/master/data1.tar.gz')
# os.system('wget --no-check-certificate https://raw.githubusercontent.com/mouna99/dien/master/data2.tar.gz')
#
# os.system('tar jxvf ./data.tar.gz && tar jxvf ./data1.tar.gz && tar jxvf ./data2.tar.gz')

TEST_RUN = False

train_df = pd.read_csv(
    pwd + "/data/local_train_splitByUser", sep='\t',
    names=['label', 'uid', 'mid', 'cat', 'hist_mids', 'hist_cats'])

valid_df = pd.read_csv(
    pwd + "/data/local_test_splitByUser", sep='\t',
    names=['label', 'uid', 'mid', 'cat', 'hist_mids', 'hist_cats'])
item_info_df = pd.read_csv(pwd + "/data2/item-info", sep='\t', names=['mid', 'cat'])

item_info_df.head()

reviews_info_df = pd.read_csv(pwd+"/data1/reviews-info", sep='\t', names=['c1', 'mid', 'c3', 'c4'])

reviews_info_df.head()

reviews_info_df = reviews_info_df[['mid']].merge(item_info_df, on='mid', how='inner').drop_duplicates()

reviews_info_df.head()

mid_cat_map = reviews_info_df.set_index('mid').to_dict()['cat']

if TEST_RUN:
    train_df = train_df.sample(1000)
    valid_df = valid_df.sample(1000)


def prepare_neg(df):
    records = df['hist_mids'].apply(lambda x: x.split(chr(0x02)))
    candidates = list(mid_cat_map.keys())
    max_len = len(candidates)

    def neg_sampling(filters, length):
        mids = []
        cats = []
        for i in range(length):
            while (1):
                c = candidates[np.random.randint(0, max_len)]
                if c not in filters:
                    mids.append(c)
                    cats.append(mid_cat_map[c])
                    filters.add(c)
                    break
        return mids, cats

    total_neg_mids = []
    total_neg_cats = []
    for record in tqdm(records):
        neg_mids, neg_cats = neg_sampling(set(record), len(record))
        total_neg_mids.append(chr(0x02).join(neg_mids))
        total_neg_cats.append(chr(0x02).join(neg_cats))

    return total_neg_mids, total_neg_cats


total_neg_mids, total_neg_cats = prepare_neg(train_df)
train_df['neg_hist_mids'] = total_neg_mids
train_df['neg_hist_cats'] = total_neg_cats

total_neg_mids, total_neg_cats = prepare_neg(valid_df)
valid_df['neg_hist_mids'] = total_neg_mids
valid_df['neg_hist_cats'] = total_neg_cats

train_df.to_csv('local_train.csv', sep='\t', index=False)
valid_df.to_csv('local_test.csv', sep='\t', index=False)

print('end')