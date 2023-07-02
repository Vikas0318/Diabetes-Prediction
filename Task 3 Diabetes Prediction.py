#!/usr/bin/env python
# coding: utf-8

# ## Task 3 : Diabetes Prediction

# In[1]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")


# In[2]:


# Loading the dataset
diabetes_dataset = pd.read_csv("C:\\Users\\Vikas\\OneDrive\\Desktop\\DATASET\\TechnoHacks Edutech\\diabetes.csv")
diabetes_dataset.head()


# In[3]:


# Number of rows and columns in dataset
diabetes_dataset.shape


# In[4]:


# Statistical measures of the data
diabetes_dataset.describe()


# In[5]:


diabetes_dataset['Outcome'].value_counts()

0 --> Non diabetic people
1 --> Diabetic people
# In[6]:


diabetes_dataset.groupby('Outcome').mean()


# In[7]:


# Separating the data and labels
X = diabetes_dataset.drop(columns = 'Outcome',axis=1)
Y = diabetes_dataset['Outcome']


# In[8]:


X


# In[9]:


Y


# Data Standardization

# In[10]:


scaler = StandardScaler()


# In[11]:


scaler.fit(X)


# In[12]:


standardized_data = scaler.transform(X)


# In[13]:


print(standardized_data)


# In[14]:


X = standardized_data
Y = diabetes_dataset['Outcome']


# In[15]:


print(X)
print(Y)


# Train_Test_Split

# In[16]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=2,stratify=Y)


# In[17]:


print(X.shape, X_train.shape, X_test.shape)


# Training the Model

# In[18]:


classifier = svm.SVC(kernel='linear')


# In[19]:


# Training the Support Vector Machine Classifier


# In[20]:


classifier.fit(X_train, Y_train)


# Model Evaluation

# Accuracy Score

# In[21]:


# Accuracy score on the training data


# In[22]:


X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction,Y_train)


# In[23]:


print('Accuracy score of the training data : ',training_data_accuracy)


# In[24]:


# Accuracy score on the test data


# In[25]:


X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction,Y_test)


# In[26]:


print('Accuracy score of the test data : ',test_data_accuracy)


# Making a Predictive System

# In[27]:


input_data = (7,196,90,0,0,39.8,0.451,41)

# Changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# Reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# standardize the input data
std_data = scaler.transform(input_data_reshaped)
print(std_data)

prediction = classifier.predict(std_data)
print(prediction)


# In[28]:


if (prediction[0] == 0): 
    print('The person is not diabetic')
else : 
    print('The person is diabetic')

