import pickle
import pandas as pd

def to_df(file_path):
  with open(file_path, 'r') as fin:
    df = {}
    i = 0
    for line in fin:
      df[i] = eval(line)
      i += 1
    df = pd.DataFrame.from_dict(df, orient='index')
    return df


# reviews_df = to_df('/raid/huaizepeng/DIN&DIEN_MIT/examples/amazon_e/reviews_Electronics_5.json')
# reviews_df = to_df('/home/hzp/CTRmodel/examples/amazon_e/reviews_Electronics_5.json')
reviews_df = to_df('/home/hzp/CTRmodel/examples/amazon_e/Electronics_5.json')
# with open('/raid/huaizepeng/DIN&DIEN_MIT/examples/amazon_e/reviews.pkl', 'wb') as f:
with open('/home/hzp/CTRmodel/examples/amazon_e/reviews.pkl', 'wb') as f:
  pickle.dump(reviews_df, f, pickle.HIGHEST_PROTOCOL)

# meta_df = to_df('/raid/huaizepeng/DIN&DIEN_MIT/examples/amazon_e/meta_Electronics.json')
meta_df = to_df('/home/hzp/CTRmodel/examples/amazon_e/meta_Electronics.json')
meta_df = meta_df[meta_df['asin'].isin(reviews_df['asin'].unique())]
meta_df = meta_df.reset_index(drop=True)
# with open('/raid/huaizepeng/DIN&DIEN_MIT/examples/amazon_e/meta.pkl', 'wb') as f:
with open('/home/hzp/CTRmodel/examples/amazon_e/meta.pkl', 'wb') as f:
  pickle.dump(meta_df, f, pickle.HIGHEST_PROTOCOL)

print('finish')