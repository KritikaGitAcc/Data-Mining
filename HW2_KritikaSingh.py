#!/usr/bin/env python
# coding: utf-8

# ***

# <body style="background-color:Gray;">
# 
# <h1 style=" text-align:center ; ">HomeWork 2</h1>
# <h2 style= "text-align:center"><b> Kritika Singh</b></h2>
# <h3 style=" text-align:center; ">CS 5310 -- Data Mining</h3>    
# 
# 
# </body>

# ****

# #### We are exploring the features available to us in the data set from 'Data11tumors.csv'. We will also try to apply the Machine Learning algorithms KNN and Gaussian Naïve Bayes and evaluate the performance without and with parameters optimization, discuss the results obtained.

# ****

# #### Importing Packages and Loading Libraries

# In[1]:


import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn import model_selection
import seaborn as sns
import matplotlib.pyplot as plt


# ##### Loading Data and performing Exploratory Analysis

# In[2]:


dataset = pd.read_csv('C:/Users/KritikaSingh/Documents/College/Data11tumors.csv')

# Summarize the Dataset

# shape
print(dataset.shape)
print('\nOur data set contains', dataset.shape[0], 'rows and', dataset.shape[1], 'columns.')
print('\nThe names of our', dataset.shape[1], 'variables (features) are:')

[print(i) for i in dataset.columns]


# In[3]:


# head
print('\nWe can get a snapshot of the first 20 rows of the data available using head().')
print('From this small sample we can estimate how each feature was recorded.')
print(dataset.head(20))


# ###### Now we will check statistical summary of the data

# In[4]:


# Statistical Summary

# descriptions
# Summarizing the Data
print('\nWe can get a snapshot of the statistical summary for the same data')
print('From this small sample we can estimate how each feature was recorded.')
print(dataset.describe())


# In[5]:


#Counts of classes in data

print('We can see that for class feauture, these are the distinctive values and their count')
dataset['Classes'].value_counts()


# ####  Now we will be replacing Class 0,1,2,3..11 with Class1, Class2, Class3....

# In[6]:


# replacing values

dataset['Classes'].replace([0,1,2,3,4,5,6,7,8,9,10],['Class1', 'Class2', 'Class3', 'Class4', 'Class5', 'Class6', 'Class7', 'Class8', 'Class9', 'Class10', 'Class11'], inplace=True)


# In[7]:


print("Rows, Columns:",dataset.shape)
dataset['Classes'].value_counts()


# In[8]:


Clases = dataset.groupby('Classes').size()
labels = Clases.index.values
sizes = Clases.values
muestra = []
for k in range(0,labels.size):
  texto = labels[k]+': '+str(sizes[k])+' samples\n({:,.2f} %)'.format((100*sizes[k]/sum(sizes)))
  muestra.append(texto)
colors = ['#E6B0AA','#D7BDE2','#A9CCE3','#A3E4D7','#F9E79F','#D5DBDB','#AEB6BF','#EDBB99','#5DADE2','#F4D03F','#27AE60']
bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
kw = dict(arrowprops=dict(arrowstyle="-"),bbox=bbox_props, zorder=0, va="center")
fig,ax1 = plt.subplots(figsize=(18,9),ncols=1,nrows=1)
wedges, texts = ax1.pie(sizes, shadow=True, colors=colors, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

for i, p in enumerate(wedges):
    ang = (p.theta2 - p.theta1)/2. + p.theta1
    y = np.sin(np.deg2rad(ang))
    x = np.cos(np.deg2rad(ang))
    horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
    connectionstyle = "angle,angleA=0,angleB={}".format(ang)
    kw["arrowprops"].update({"connectionstyle": connectionstyle})
    ax1.annotate(muestra[i], xy=(x, y), xytext=(1.35*np.sign(x), 1.4*y),
                horizontalalignment=horizontalalignment, **kw)
plt.show()


# #### We can see that we have replaced the class name and also checked the classs and their sample coount in pie chart above. Now we check the features and label determination 

# In[ ]:





# In[9]:


x = dataset.drop(['Classes'], axis=1)
y = dataset['Classes'].values
print(x.shape)
print(y.shape)


# #### No we need to check for missing values

# In[10]:


#Missing values (in percent)
missing = (x.isnull().sum() / len(x)).sort_values(ascending = False)
missing = missing.index[missing > 0.5]
all_missing = list(set(missing))
print('There are %d columns with more than 50%% missing values' % len(all_missing))


# ### Splitting the dataset into the Training set and Test set

# In[11]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test =  model_selection.train_test_split(x, y, test_size = 0.2, random_state = 1)


# In[12]:


print(x_train)


# In[13]:


print (x_test)


# In[14]:


print (y_train)


# In[15]:


print(y_test)


# #### Feature scaling (Normalizing data)

# In[16]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# ### Training the K-NN model on the Training set

# In[17]:


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(x_train, y_train)


# #### Predicting a new result

# In[18]:


print(classifier.predict(sc.transform(x_test)))


# #### Predicting the Test set results

# In[19]:


y_pred = classifier.predict(x_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


# #### Making the Confusion matrix

# In[20]:


from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)


# ### Training the Naive Bayes model on the Training set

# In[21]:


from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(x_train, y_train)


# #### Predicting a new result

# In[22]:


y_pred = classifier.predict(x_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


# #### Making the Confusion Matrix

# In[23]:


from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)


# <b> We are using the metric of ‘accuracy‘ to evaluate models. This is a ratio of the number of correctly predicted instances in divided by the total number of instances in the dataset multiplied by 100 to give a percentage (e.g. 95% accurate). We will be using the scoring variable when we run build and evaluate each model next.
# 
# #### WE will now build models
# #### We are now goin to evalauate both the algorithms </b>

# In[24]:


from sklearn.metrics import accuracy_score
models = []
models.append(('GB', GaussianNB()))
models.append(('KNN', KNeighborsClassifier()))

# evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = model_selection.cross_val_score(model, x_train, y_train, cv=kfold)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# We can also create a plot of the model evaluation results and compare the spread and the mean accuracy of each model. There is a population of accuracy measures for each algorithm because each algorithm was evaluated 10 times (10 fold cross validation).

# Compare Algorithms Accuracy
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


# We now have 2 models and accuracy estimations for each. We need to compare the models to each other and select the most accurate.
# 
# In this case, we can see that it looks like Gaussian NB has the largest estimated accuracy score.
# 
# We can also create a plot of the model evaluation results and compare the spread and the mean accuracy of each model. There is a population of accuracy measures for each algorithm because each algorithm was evaluated 10 times (10 fold cross validation).

# In[ ]:





# In[ ]:





# In[ ]:




