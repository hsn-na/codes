#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import datetime
import random
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.svm import SVR
from mlxtend.regressor import StackingCVRegressor
import lightgbm as lgb
from xgboost import XGBRegressor
from scipy.stats import skew, norm
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA

pd.set_option('display.max_columns', None)

import warnings
warnings.filterwarnings(action="ignore")
pd.options.display.max_seq_items = 8000
pd.options.display.max_rows = 8000


# In[2]:


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")


# In[3]:


train.head()


# In[4]:


#sale price is the variable we are trying to predict


# In[5]:


f, ax = plt.subplots(figsize=(16, 14))

sns.set_style('darkgrid')
sns.distplot(train['SalePrice'])
ax.set(ylabel="Frequency")
ax.set(xlabel="SalePrice")
ax.set(title="SalePrice distribution")

plt.show()


# In[6]:


print("Skewness: %f" % train['SalePrice'].skew())
print("Kurtosis: %f" % train['SalePrice'].kurt())


# In[7]:


corr = train.corr()
plt.subplots(figsize=(15,12))
sns.heatmap(corr, vmax=0.9, cmap="YlGnBu", square=True)
plt.show()


# In[8]:


print("Find most important features relative to target")
corr = train.corr()
corr.sort_values(["SalePrice"], ascending = False, inplace = True)
print(corr.SalePrice)


# In[9]:


plt.subplots(figsize=(15,12))
plt.scatter(train['OverallQual'],train['SalePrice'])
plt.show()


# In[10]:


plt.subplots(figsize=(15,12))
plt.scatter(train['YearBuilt'],train['SalePrice'])
plt.show()


# In[11]:


sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(train[cols], size = 4)
plt.show();


# In[12]:


#feature engineering


# In[13]:


train["SalePrice_log"] = np.log1p(train["SalePrice"])


# In[14]:


f, ax = plt.subplots(figsize=(16, 14))

sns.set_style('darkgrid')
sns.distplot(train['SalePrice_log'], fit=norm)
ax.set(ylabel="Frequency")
ax.set(xlabel="SalePrice")
ax.set(title="SalePrice normal distribution")

plt.show()


# In[15]:


(mu, sigma) = norm.fit(train['SalePrice_log'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))


# In[16]:


#removing outliers

train.drop(train[(train['OverallQual']<5) & (train['SalePrice_log']>200000)].index, inplace=True)
train.drop(train[(train['GrLivArea']>4500) & (train['SalePrice_log']<300000)].index, inplace=True)
train.reset_index(drop=True, inplace=True)


# In[17]:


train_labels = train['SalePrice_log'].reset_index(drop=True)
train_features = train.drop(['SalePrice_log'], axis=1)
test_features = test

# Combine train and test features in order to apply the feature transformation pipeline to the entire dataset
all_features = pd.concat([train_features, test_features]).reset_index(drop=True)
all_features.shape


# In[18]:


def perc_missing(df):
    data = pd.DataFrame(df)
    df_cols = list(pd.DataFrame(data))
    dict_x = {}
    for i in range(0, len(df_cols)):
        dict_x.update({df_cols[i]: round(data[df_cols[i]].isnull().mean()*100,2)})
    
    return dict_x

missing = perc_missing(all_features)
df_miss = sorted(missing.items(), key=lambda x: x[1], reverse=True)
print('Percent of missing data')
del df_miss[4]
#can delete [4], as we dont need non-log price
df_miss[0:10]


# In[19]:


sns.set_style('darkgrid')
f, ax = plt.subplots(figsize=(8, 7))

missing = round(train.isnull().mean()*100,2)
missing = missing[missing > 0]
missing.sort_values(inplace=True)
missing.plot.bar(color="b")
# Tweak the visual presentation
ax.xaxis.grid(False)
ax.set(ylabel="Percent of missing values")
ax.set(xlabel="Features")
ax.set(title="Percent missing data by feature")
sns.despine(trim=True, left=True)


# In[20]:


#convert predictors to strings, so we can impute missing values

all_features['MSSubClass'] = all_features['MSSubClass'].apply(str)
all_features['YrSold'] = all_features['YrSold'].astype(str)
all_features['MoSold'] = all_features['MoSold'].astype(str)


# In[21]:


def input_missing(features):
    #description of data replaced by data type
    features['Functional'] = features['Functional'].fillna('Typ')
    
    #missing valuess replaced by most common feature
    features['Electrical'] = features['Electrical'].fillna("SBrkr")
    features['KitchenQual'] = features['KitchenQual'].fillna("TA")
    features['Exterior1st'] = features['Exterior1st'].fillna(features['Exterior1st'].mode()[0])
    features['Exterior2nd'] = features['Exterior2nd'].fillna(features['Exterior2nd'].mode()[0])
    features['SaleType'] = features['SaleType'].fillna(features['SaleType'].mode()[0])
    features['MSZoning'] = features.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))
    
    #missing value assumes None or 0
    
    features["PoolQC"] = features["PoolQC"].fillna("None")
    
    for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
        features[col] = features[col].fillna(0)
    
    for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:
        features[col] = features[col].fillna('None')
    
    for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
        features[col] = features[col].fillna('None')
    
    #missing values replaced by median of neighbourhood
    
    features['LotFrontage'] = features.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
    
    #the rest is filled by None as no more assumptions can be made
    
    objects = []
    for i in features.columns:
        if features[i].dtype == object:
            objects.append(i)
    features.update(features[objects].fillna('None'))
    
    numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numeric = []
    for i in features.columns:
        if features[i].dtype in numeric_dtypes:
            numeric.append(i)
    features.update(features[numeric].fillna(0))    
    return features

all_features = input_missing(all_features)


# In[22]:


missing = perc_missing(all_features)
df_miss = sorted(missing.items(), key=lambda x: x[1], reverse=True)
print('Percent of missing data')
#df_miss is ID which isnt needed
df_miss[1:11]


# In[23]:


numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numeric = []
for i in all_features.columns:
    if all_features[i].dtype in numeric_dtypes:
        numeric.append(i)


# In[24]:


all_features = all_features[numeric[1:34]]


# In[25]:


# find skewed values

skewed = all_features.apply(lambda x: skew(x)).sort_values(ascending=False)
high_skew = skewed[abs(skewed) > 0.5]
skew_index = high_skew.index

print("There are {} numerical features with -0.5 < Skew > 0.5 :".format(high_skew.shape[0]))
skewness = pd.DataFrame({'Skew' :high_skew})
skewed


# In[26]:


for i in skew_index:
    all_features[i] = boxcox1p(all_features[i], boxcox_normmax(all_features[i] + 1))


# In[27]:


#new table is of normalised variables

all_features = all_features.loc[:,~all_features.columns.duplicated()]


# In[28]:


X = all_features.iloc[:len(train_labels), :]
X_test = all_features.iloc[len(train_labels):, :]
X.shape, train_labels.shape, X_test.shape


# In[30]:


kf = KFold(n_splits=12, random_state=42, shuffle=True)


# In[31]:


def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

def cv_rmse(model, X=X):
    rmse = np.sqrt(-cross_val_score(model, X, train_labels, scoring="neg_mean_squared_error", cv=kf))
    return (rmse)


# In[32]:


xgboost = XGBRegressor(learning_rate=0.01,
                       n_estimators=6000,
                       max_depth=4,
                       min_child_weight=0,
                       gamma=0.6,
                       subsample=0.7,
                       colsample_bytree=0.7,
                       objective='reg:squarederror',
                       nthread=-1,
                       scale_pos_weight=1,
                       seed=27,
                       reg_alpha=0.00006,
                       random_state=42)

ridge_alphas = [1e-15, 1e-10, 1e-8, 9e-4, 7e-4, 5e-4, 3e-4, 1e-4, 1e-3, 5e-2, 1e-2, 0.1, 0.3, 1, 3, 5, 10, 15, 18, 20, 30, 50, 75, 100]
ridge = make_pipeline(RobustScaler(), RidgeCV(alphas = ridge_alphas, cv = kf))

svr = make_pipeline(RobustScaler(), SVR(C = 20,
                                        epsilon = 0.008,
                                        gamma = 0.0003))


# In[35]:


scores = {}


# In[36]:


score = cv_rmse(xgboost)
print("xgboost: {:.4f} ({:.4f})".format(score.mean(), score.std()))
scores['xgb'] = (score.mean(), score.std())


# In[37]:


score = cv_rmse(svr)
print("SVR: {:.4f} ({:.4f})".format(score.mean(), score.std()))
scores['svr'] = (score.mean(), score.std())


# In[38]:


score = cv_rmse(ridge)
print("ridge: {:.4f} ({:.4f})".format(score.mean(), score.std()))
scores['ridge'] = (score.mean(), score.std())


# In[67]:


y = [scores['xgb'][0], scores['svr'][0], scores['ridge'][0]]
x = list(scores.keys())


# In[72]:


plt.figure(figsize=(10,10))
plt.plot(x, y, 'o-')
plt.ylabel('Score (RMSE)', size=20, labelpad=12.5)
plt.xlabel('Model', size=20, labelpad=12.5)
plt.tick_params(axis='x', labelsize=13.5)
plt.tick_params(axis='y', labelsize=12.5)

plt.title('Scores of Models', size=20)
plt.show()


# In[73]:


#best model is svr


# In[76]:


svr_model_full_data = svr.fit(X, train_labels)


# In[77]:


svr_model_full_data


# In[79]:


def predict(X):
    return svr_model_full_data.predict(X)


# In[89]:


X_test['SalePrice_predict'] = np.floor(np.expm1(predict(X_test)))


# In[91]:


X_test.head()


# In[ ]:




