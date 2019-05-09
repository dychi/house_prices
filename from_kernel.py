#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

#%%
import warnings
warnings.filterwarnings('ignore')

#%%
train_df = pd.read_csv('./data/train.csv')
test_df = pd.read_csv('./data/test.csv')

#%%
train_df.head()

#%%
NAs = pd.concat([train_df.isnull().sum(), test_df.isnull().sum()], axis=1, keys=['Train', 'Test'])
# 欠損値が含まれているものを出力
NAs[NAs.sum(axis=1) > 0]

#%%
train_labels = train_df.SalePrice
ax = sns.distplot(train_labels)

#%% 
# 特徴量設計


#%%
# 学習データ、テストデータを分割

#%%
