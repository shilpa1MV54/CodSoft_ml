#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score
from sklearn.preprocessing import StandardScaler


# In[2]:


df = pd.read_csv('Churn_Modelling.csv')
df.head()


# In[3]:


df.describe()


# In[4]:


df.info()


# In[5]:


df.isnull().sum()


# In[6]:


df.duplicated().sum()


# In[7]:


df.columns


# In[8]:


drop_cols = ['CustomerId','Surname','RowNumber']
df.drop(drop_cols, axis=1, inplace=True)
df.head()


# In[9]:


print(df['Geography'].unique())
df['Geography'].value_counts()


# In[10]:


sns.countplot(x='Geography',hue='Exited',data=df)
plt.show()


# In[11]:


sns.countplot(x='Gender',hue='Exited',data=df)
plt.show()


# In[12]:


df['Exited'].value_counts().plot(kind='bar')
plt.xlabel('Exited')
plt.ylabel('Count')
plt.show()


# In[13]:


df = pd.get_dummies(df, columns=['Gender','Geography'])
order = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts',
       'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Gender_Female',
       'Gender_Male', 'Geography_France', 'Geography_Germany', 'Geography_Spain', 'Exited']
df = df[order]
df.head()


# In[14]:


X = df.drop(columns=['Exited'])
y = df['Exited']


# In[15]:


scaler = StandardScaler()
X = scaler.fit_transform(X)


# In[16]:


print(X.shape)
y.shape


# In[17]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=2)


# In[18]:


lg = LogisticRegression()
rf = RandomForestClassifier(n_estimators=50, random_state=2)
gb = GradientBoostingClassifier(n_estimators=50,random_state=2)


# In[19]:


clfs = {
    'lg':lg,
    'rf':rf,
    'gb':gb
}


# In[20]:


def train_clfs_and_predict(clfs,X_train,X_test,y_train,y_test):
    acc = []
    prec = []
    conf_mat = []

    for clf in clfs:
        model = clfs[clf]
        model.fit(X_train,y_train)
        y_pred = model.predict(X_test)
        acc.append(accuracy_score(y_test,y_pred))
        prec.append(precision_score(y_test,y_pred))
        conf_mat.append(confusion_matrix(y_test,y_pred))

    return acc, prec, conf_mat


# In[21]:


accuracy, precision, conf_mat = train_clfs_and_predict(clfs,X_train,X_test,y_train,y_test)


# In[22]:


performance = {
    'classifiers':list(clfs.keys()),
    'accuracy':accuracy,
    'precision':precision,
    'confusion_matrix':conf_mat,
}


# In[23]:


perf_df = pd.DataFrame(performance).sort_values(by='accuracy',ascending=False)
perf_df.head()


# In[24]:


num_classifiers = len(conf_mat)

fig, axes = plt.subplots(1, num_classifiers, figsize=(20, 5))  # Adjusting figsize 

for i, (matrix, classifier) in enumerate(zip(conf_mat, list(clfs.keys()))):
    sns.set(font_scale=1)  # Adjusting the font size 
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Predicted Negative", "Predicted Positive"],
                yticklabels=["Actual Negative", "Actual Positive"],
                ax=axes[i])
    axes[i].set_title(f"Confusion Matrix for {classifier}")
    axes[i].set_xlabel("Predicted Label")
    axes[i].set_ylabel("True Label")


# In[25]:


sns.set(style="whitegrid")
sns.lineplot(x=perf_df.classifiers, y=perf_df.accuracy, marker='o', label='Accuracy', data=perf_df)
sns.lineplot(x=perf_df.classifiers, y=perf_df.precision, marker='o', label='Precision', data=perf_df)

plt.title("Accuracy and Precision by Classifiers")
plt.xlabel("Classifiers")
plt.ylabel("Value")
plt.legend()
plt.show()


# In[ ]:




