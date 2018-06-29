
# coding: utf-8

# # Iris Dataset
# 
# I will be beginning a raw approach to the dataset i.e no external help and try to build up.

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as skl
import seaborn as sns


# In[2]:


data = pd.read_csv('iris.data.txt' , header = None)
data.columns = ['Sepal-Length','Sepal-Width','Petal-Length','Petal-Width','Class']


# Visualising the data

# In[3]:


data.head(10)


# In[4]:


data.shape


# There are 150 rows and 5 columns.
# 
# According to the dataset description, the columns stand for - 
# 1. Sepal length in cm
# 2. Sepal width in cm
# 3. Petal length in cm
# 4. Petal width in cm
# 5. Class - Sentosa, Virgicolour, Virginica

# In[5]:


data['Class'].unique().tolist()


# In[6]:


data.describe()


# In[7]:


plt.subplots(figsize=(10,5))
sns.set(style="darkgrid")
sns.countplot(x='Class', data=data)


# In[8]:


sns.lmplot(x='Sepal-Length', y='Sepal-Width', data=data, fit_reg=False, hue='Class', markers=["o","x","^"], aspect=2, scatter_kws={"s": 100})


# Sepal length and Sepal width is not giving much help since they are overlapping. While Iris-sentosa is comfortably separated from the two, Versicolour and Virginica is not.
# 
# Let us now plot the Petal Plot.

# In[9]:


sns.lmplot(x='Petal-Length', y='Petal-Width', data=data, fit_reg=False, hue='Class', markers=["o","x","^"], aspect=2, scatter_kws={"s": 100})


# This is better. All the three species can be separated but the Versicolour and Virginica still needs some work.

# ## Machine Learning Algorithms
# 
# Now we will start by importing the Sci-kit Learn's modules and try different models.

# In[10]:


from sklearn.linear_model import LogisticRegression #Logistic Regression
from sklearn import svm #Support Vector Machines
from sklearn import metrics #Scoring of our model
from sklearn.ensemble import RandomForestClassifier #RandomForest to check the feature importances
from sklearn.neighbors import KNeighborsClassifier #K-NN
from sklearn.model_selection import train_test_split #splitting the dataset


# Splitting the dataset into 30% test and 70% training set.

# In[11]:


train, test = train_test_split(data, test_size=0.3)
print(train.shape)
print(test.shape)


# Now, creating the Training Set and the Test Set

# In[12]:


trainX = train[['Sepal-Length','Sepal-Width','Petal-Length','Petal-Width']]
trainY = train[['Class']].values

testX = test[['Sepal-Length','Sepal-Width','Petal-Length','Petal-Width']]
testY = test[['Class']].values


# Testing the feature importances using RandomForestClassifier

# In[13]:


rndF = RandomForestClassifier(n_estimators = 50, max_features='sqrt')
rndF = rndF.fit(trainX, trainY.ravel())

feat = pd.DataFrame()
feat['Feature'] = trainX.columns
feat['Importance'] = rndF.feature_importances_
feat.sort_values(by=['Importance'], ascending=True, inplace=True)
feat.set_index('Feature', inplace=True)


# In[14]:


feat.plot(kind='barh', figsize=(10,5))


# Petal length and Petal width have the most importance in the Random Forest Classification model.

# ### Logistic Regression

# In[15]:


model = LogisticRegression()
model.fit(trainX,trainY.ravel())

predictY = model.predict(testX)

#Classification Report and Confusion Matrix
print(metrics.classification_report(testY,predictY))

print(metrics.confusion_matrix(testY,predictY))

#print the accuracy
print("\nAccuracy of the model using Logistic Regression is {0}".format(metrics.accuracy_score(testY,predictY)))


# ### Support Vector Machine

# In[16]:


model = svm.SVC()
model.fit(trainX,trainY.ravel())

predictY = model.predict(testX)

#Classification Report and Confusion Matrix
print(metrics.classification_report(testY,predictY))

print(metrics.confusion_matrix(testY,predictY))

#print the accuracy
print("\nAccuracy of the model using Support Vector Machine is {0}".format(metrics.accuracy_score(testY,predictY)))


# ### K Nearest Neighbour

# In[17]:


model = KNeighborsClassifier(n_neighbors=8)
model.fit(trainX,trainY.ravel())

predictY = model.predict(testX)

#Classification Report and Confusion Matrix
print(metrics.classification_report(testY,predictY))

print(metrics.confusion_matrix(testY,predictY))

#print the accuracy
print("\nAccuracy of the model using K Neighbours Classifier is {0}".format(metrics.accuracy_score(testY,predictY)))


# ### Random Forest Classifier

# In[18]:


model = RandomForestClassifier(n_estimators = 100, max_features='sqrt')
model.fit(trainX,trainY.ravel())

predictY = model.predict(testX)

#Classification Report and Confusion Matrix
print(metrics.classification_report(testY,predictY))

print(metrics.confusion_matrix(testY,predictY))

#print the accuracy
print("\nAccuracy of the model using Random Forest Classifier is {0}".format(metrics.accuracy_score(testY,predictY)))


# ## Conclusion
# 
# We have seen that out of all the models, Support Vector Machines, K-Neighbours and Random Forest gives the perfect score.
# This was a beginner level dataset with no data preprocessing required.
