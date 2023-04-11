#!/usr/bin/env python
# coding: utf-8

# # Importing the Libraries Needed to Visualize the Data

# In[42]:


import numpy as np   #linear algebra
import seaborn as sns
import pandas as pd      #data processing, reading a csv file

import matplotlib.pyplot as plt
plt.style.use("ggplot")

get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.graph_objects as go
import plotly.express as px


# # Reading the Data Set and Creating a Data Table for it

# In[43]:


adapt_level = pd.read_csv('students_adaptability_level_online_education.csv')


# In[44]:


adapt_level.head()


# In[45]:


adapt_level.tail()


# In[46]:


adapt_level.describe().T


# In[47]:


adapt_level.info()


# In[48]:


adapt_level.isnull().sum()


# # Visualizing the Different Columns in the Data Set

# In[49]:


plt.figure(figsize = (12,8))
adapt_level['Adaptivity Level'].value_counts().plot.pie(autopct ='%1.1f%%', shadow = True)


# In[50]:


plt.figure(figsize = (12,8))
adapt_level['Gender'].value_counts().plot.pie(autopct ='%1.1f%%', shadow = True)


# In[10]:


#visualizing the effects of different factors against students adaptivity level
i = 1
plt.figure(figsize = (20, 60))
for factors in [col for col in adapt_level.columns if col!= 'Adaptivity Level']:
    plt.subplot(6,3,i),
    sns.countplot(x = factors, hue = 'Adaptivity Level', data = adapt_level,)
    i +=1


# # Importing Libraries for Graph/Data Analysis

# In[11]:


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import OrdinalEncoder    #helps transform the data into numeric numbers for the computer to easily understand
from sklearn.preprocessing import LabelEncoder,StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score,recall_score
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
from sklearn.metrics import ConfusionMatrixDisplay


# In[12]:


age_col = adapt_level['Age'].apply(lambda x: x.split("-")[0])
update_adapt_level = adapt_level.join(age_col.to_frame(name = "Lower Limit Age"))
update_adapt_level.drop(['Age'], axis = 1, inplace = True)

update_adapt_level


# In[13]:


update_adapt_level['Lower Limit Age'] = update_adapt_level['Lower Limit Age'].astype(int)


# In[14]:


scaler = OrdinalEncoder()
labels = update_adapt_level.columns
d = scaler.fit_transform(update_adapt_level)

scaler_df = pd.DataFrame(d, columns = labels)
scaler_df.head(10)


# # Machine Learning Algorithm : DecisionTreeClassifier()

# In[15]:


update_dataset = scaler_df
x = update_dataset.drop(columns = ['Adaptivity Level'])
y = update_dataset['Adaptivity Level']


# In[16]:


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 60)

print('x train : ',x_train.shape)
print('x test : ', x_test.shape)
print('y train : ', y_train.shape)
print('y test : ', y_test.shape)


# In[17]:


DTC = DecisionTreeClassifier()
model = DTC.fit(x_train, y_train)
predictions = DTC.predict(x_test)
print (DTC, ':', accuracy_score(y_test,predictions)*100)


# # Machine Learning Algorithm : LogisticRegression()

# In[18]:


update_dataset = scaler_df
x = update_dataset.drop(columns = ['Adaptivity Level'])
y = update_dataset['Adaptivity Level']


# In[19]:


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 60)

print('x train : ',x_train.shape)
print('x test : ', x_test.shape)
print('y train : ', y_train.shape)
print('y test : ', y_test.shape)


# In[20]:


LR = LogisticRegression()
model = LR.fit(x_train, y_train)
predictions = LR.predict(x_test)
print (LR, ':', accuracy_score(y_test,predictions)*100)


# # Machine Learning Algorithm : RandomForestClassifier()

# In[21]:


update_dataset = scaler_df
x = update_dataset.drop(columns = ['Adaptivity Level'])
y = update_dataset['Adaptivity Level']


# In[22]:


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 60)

print('x train : ',x_train.shape)
print('x test : ', x_test.shape)
print('y train : ', y_train.shape)
print('y test : ', y_test.shape)


# In[23]:


RFC = RandomForestClassifier()
model = RFC.fit(x_train, y_train)
predictions = RFC.predict(x_test)
print (RFC, ':', accuracy_score(y_test,predictions)*100)


# # Confusion Matrix using the RandomForestClassifier()

# In[24]:


cm = confusion_matrix(RFC.predict(x_test),y_test)
display = ConfusionMatrixDisplay(cm, display_labels = ["High", "Low", "Moderate"])
display.plot()
plt.title("Confusion Matrix")
plt.show()


# # Determing Which Factors are Important to Make Meaningful Conclusions

# In[38]:


feature_importances=RFC.feature_importances_
feature_importances_df=pd.DataFrame({'Variable':list(x_train), 'Variable importance':feature_importances})

feature_importances_df.sort_values('Variable importance',ascending=False)

feat_importances = pd.Series(RFC.feature_importances_, index=x.columns)

feat_importances.nlargest(5).plot(kind='barh')
plt.title("5 most important features")
