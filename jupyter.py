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
train_df.plot(kind='scatter', x=u'LotFrontage', y=u'SalePrice')

#%%
from sklearn.preprocessing import StandardScaler, MinMaxScaler


#%%
mms = MinMaxScaler()
mms.fit_transform(train_df[["GrLivArea"]].values)

#%%
stdsc = StandardScaler()
GrLivArea_std = stdsc.fit_transform(train_df[['GrLivArea']].values)
grlivarea_df = pd.DataFrame({'x': train_df.GrLivArea.values, 'y': GrLivArea_std[:,0]})
grlivarea_df.plot(kind='hist')

#%%
display(GrLivArea_std[:,0].shape)
display(train_df.GrLivArea.values.shape)


#%%
train_df.plot(kind='hist', x=u'LotFrontage', y=u'SalePrice', bins=200)

#%%
train_df.LotFrontage[train_df.LotFrontage.isnull()]

#%%
fig = plt.figure(figsize=(10,10))
ax = fig.gca()
train_df.hist(ax=ax)
plt.tight_layout()
plt.show()

#%%
train_df.isnull().sum()

#%%
plt.figure(figsize=(10,10))
sns.heatmap(train_df.corr(), cmap='coolwarm_r', vmin=-1, vmax=1)

#%%
train_df.plot(kind='scatter', x='OverallQual', y='SalePrice')

#%%
train_df.plot(kind='scatter', x='GrLivArea', y='SalePrice')


#%%
# 数字以外の特徴量
df_obj = train_df.select_dtypes(include=['O'])
df_obj.head()

#%%
# 数字の特徴量
df_num = train_df.select_dtypes(exclude=['O']).drop('Id', axis=1)
display(df_num.isnull().sum())
df_num.head()


#%%
cp = sns.countplot('Neighborhood', data=df_obj, orient=20)
plt.xticks(rotation=30)

#%%
fig = plt.figure(figsize=(10,10))
ax = fig.gca()
df_num.plot(kind='hist', bins=50, subplots=True, ax=ax)

#%%
from sklearn.
from sklearn.model_selection import KFold

#%%
X = x.reshape(-1,1) # scikit-learnに入力するために整形
n_split = 5 # グループ数を設定（今回は5分割）

cross_valid_mae = 0
split_num = 1

# テスト役を交代させながら学習と評価を繰り返す
for train_idx, test_idx in KFold(n_splits=n_split, random_state=None).split(X, y):
    X_train, y_train = X[train_idx], y[train_idx] #学習用データ
    X_test, y_test = X[test_idx], y[test_idx]     #テスト用データ
    
    # 学習用データを使って線形回帰モデルを学習
    regr = LinearRegression(fit_intercept=False)
    regr.fit(X_train, y_train)

    # テストデータに対する予測を実行
    y_pred_test = regr.predict(X_test)
    
    # テストデータに対するMAEを計算
    mae = mean_absolute_error(y_test, y_pred_test)
    print("Fold %s"%split_num)
    print("MAE = %s"%round(mae, 3))
    print()
    
    cross_valid_mae += mae #後で平均を取るためにMAEを加算
    split_num += 1

# MAEの平均値を最終的な汎化誤差値とする
final_mae = cross_valid_mae / n_split
print("Cross Validation MAE = %s"%round(final_mae, 3))