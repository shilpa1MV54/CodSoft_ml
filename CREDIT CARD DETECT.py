#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
train=pd.read_csv('fraudTrain.csv.zip')
test=pd.read_csv("fraudTest.csv.zip")
train.head()


# In[5]:


test.head()


# In[6]:


train.columns


# In[7]:


test.columns


# In[8]:


train.shape


# In[9]:


test.shape


# In[10]:


frames = [train,test]
df = pd.concat(frames)
df.shape


# In[11]:


df.reset_index(inplace=True)
df.info()


# In[13]:


from sklearn.preprocessing import OrdinalEncoder
cols = ['trans_date_trans_time', 'merchant', 'category', 'first', 'last',
        'gender', 'street', 'city', 'state', 'job', 'dob', 'trans_num']
encoder = OrdinalEncoder()
df[cols] = encoder.fit_transform(df[cols])


# In[14]:


df.duplicated().sum()


# In[15]:


df.isnull().sum()


# In[16]:


df['is_fraud'].value_counts()


# In[17]:


from sklearn.model_selection import train_test_split 
x=df.drop(['is_fraud'],axis=1)
y=df['is_fraud']


# In[18]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)


# In[19]:


x_train.shape


# In[20]:


from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(x_train,y_train)


# In[21]:


from sklearn.metrics import accuracy_score


# In[22]:


pred_train = model.predict(x_train)
pred_test  = model.predict(x_test)

print('Training Accuracy : ', accuracy_score(y_train, pred_train))
print('Testing  Accuracy : ', accuracy_score(y_test, pred_test))


# In[ ]:




