
# coding: utf-8

# In[3]:


import os
os.chdir('C:/Users/Dell/Documents/MLDS_final_data/MLDS_final_data')


# In[4]:


os.getcwd()


# In[5]:


import pandas as pd 
import matplotlib as plt
get_ipython().magic('matplotlib inline')
import seaborn as sns
import numpy as np


# In[6]:


train = pd.read_csv('final_train_data.csv')


# In[7]:


train.head()


# In[8]:


train.tail()


# In[11]:


test = pd.read_csv('final_test_data.csv')

test.head()
# In[9]:


sns.distplot(train['Balance'],bins=70)


# In[13]:


train.describe(include='all')


# In[14]:


test.describe(include='all')


# In[15]:


train['gdp_growth']=train['GDP']-train['Inflation']


# In[16]:


train['export_tradeoff']=(1-train['Trade']/100)*train['Exports']


# In[17]:


test['gdp_growth']=train['GDP']-train['Inflation']


# In[18]:


test['export_tradeoff']=(1-train['Trade']/100)*train['Exports']


# In[19]:


train['Year']=train['Year'].astype(np.object)
test['Year']=train['Year'].astype(np.object)


# In[20]:


train['gdp_growth']


# In[2]:


train.head()


# In[25]:


test.head()


# In[26]:


train.shape


# In[27]:


test.shape


# In[22]:


train.shape
n_cols=['Inflation', 'GDP', 'Exports', 'Trade', 'gdp_growth', 'export_tradeoff']


# In[23]:


for i in n_cols:
    train[i].fillna(train[i].mean(),inplace=True)


# In[24]:


for i in n_cols:
    test[i].fillna(test[i].mean(),inplace=True)


# In[25]:


train.head()


# In[32]:


train.head()


# In[33]:


#from sklearn.preprocessing import RobustScaler, StandardScaler, Normalizer
from sklearn.preprocessing import RobustScaler #(with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0)


# In[34]:


#from sklearn.preprocessing import RobustScaler
#data[n_cols]=RobustScaler.fit_transform(data[n_cols])
#data1[n_cols]=RobustScaler.fit_transform(data1[n_cols])
from sklearn.preprocessing import StandardScaler
s=StandardScaler()
train[n_cols]=s.fit_transform(train[n_cols])
test[n_cols]=s.transform(test[n_cols])


# In[35]:


#from sklearn.preprocessing import Normalizer
#N=Nomalizer()
#data[n_cols]=N.fit_transform(data[n_cols])
#data1[n_cols]=N.fit_transform(data1[n_cols])
from sklearn.preprocessing import RobustScaler
r=RobustScaler()
train[n_cols]=r.fit_transform(train[n_cols])
test[n_cols]=r.fit_transform(test[n_cols])


# In[36]:


train[n_cols].isnull().sum()


# In[83]:


test.info


# In[ ]:


from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold,train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,roc_auc_score
import warnings
import seaborn as sns
import matplotlib.pyplot as plt 
import itertools
from scipy.stats import mode


# In[72]:


X_train,X_val,y_train,y_val = train_test_split(train.drop(['Unnamed: 0','Balance'],axis=1),train['Balance'],
                                                 test_size=0.25,random_state = 1994)


# In[73]:


categorical_features_indices = np.where(X_train.dtypes =='object')[0]
categorical_features_indices


# In[74]:


X,y=train.drop(['Unnamed: 0','Balance'],axis=1),train['Balance']
Xtest=test.drop(['Unnamed: 0'],axis=1)


# In[75]:


X,y=train.drop(['Unnamed: 0','Balance'],axis=1),train['Balance']
Xtest=test.drop(['Unnamed: 0'],axis=1)


# In[76]:


import math
def rmsle(h, y): 
    """
    #Compute the Root Mean Squared Log Error for hypthesis h and targets y

    #Args:
        h - numpy array containing predictions with shape (n_samples, n_targets)
        y - numpy array containing targets with shape (n_samples, n_targets)
    """
    return 1/(1+math.exp(np.sqrt(np.square(np.log(h + 1) - np.log(y + 1)).mean())))

def runCatBoost(x_train, y_train,x_test, y_test,test,depth):
    model=CatBoostRegressor(n_estimators=1000,
                            learning_rate=0.1,
                            loss_function='RMSE',
                            eval_metric='RMSE',
                            random_seed=1994,

                           )
                           
    model.fit(x_train, y_train,cat_features=categorical_features_indices, eval_set=(x_test, y_test), use_best_model=True, verbose=150)
    y_pred_train=model.predict(x_test)
    rmsle_result = rmsle(y_pred_train,y_test)
    y_pred_test=model.predict(test)
    return y_pred_train,rmsle_result,y_pred_test
#     return y_pred_train,y_pred_test


# In[77]:


from sklearn import model_selection
pred_full_test_cat_feen = 0
mse_cat_list_feen=[]
kf = model_selection.KFold(n_splits=2, shuffle=True, random_state=30)
for dev_index, val_index in kf.split(X):
    dev_X, val_X = X.loc[dev_index], X.loc[val_index]
    dev_y, val_y = y.loc[dev_index], y.loc[val_index]
    y_pred_feen,rmsle_feen,y_pred_test_feen=runCatBoost(dev_X, dev_y, val_X, val_y,Xtest,depth=4)
    print('fold score :',rmsle_feen)
    mse_cat_list_feen.append(rmsle_feen)
    pred_full_test_cat_feen = pred_full_test_cat_feen + y_pred_test_feen
mse_cat_feen_mean=np.mean(mse_cat_list_feen)
print("Mean cv score : ", np.mean(mse_cat_feen_mean))
y_pred_test_feen=pred_full_test_cat_feen/2


# In[59]:



sns.distplot(y_pred_test_feen)


# In[1]:


test1 = y_pred_test_feen
test1['Balance']=y_pred_test_feen
test1.drop('Unnamed: 0',axis=1,inplace=True)



# In[1]:


test1=(y_pred_test_feen)


# In[65]:


y_pred_test_feen = test1


# In[82]:


y_pred_test_feen
s=pd.DataFrame({'Balance':y_pred_test_feen})


# In[89]:


#y_pred_test_feen
#s=pd.DataFrame({'Balance':y_pred_test_feen})
#test1.to_csv('2foldcb8.csv',index=False)
#test1.head()

