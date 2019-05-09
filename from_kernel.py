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
# 標準化
num_df = train_df.select_dtypes(exclude='O')
num_df.head()

#%%
num_features = num_df.iloc[:,['LotFrontage', 'LotArea', 'GrLivArea', 'TotalSF']]

#%%
from sklearn.preprocessing import StandardScaler

#%%
stdsc = StandardScaler()
std_vec = stdsc.fit_transform(num_df.LotFrontage.values[:,np.newaxis])
std_vec

#%% 
feat = num_df.LotFrontage.values
feat_std = (feat - feat.mean()) / feat.std()
feat_std

#%%
# 学習データ、テストデータを分割

#%%
