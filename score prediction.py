#!/usr/bin/env python
# coding: utf-8

# # Name:Nikita Tiwari

# # The spark foundation GRIP may2024

# # Data Science and Busniss Analystics intern at the spark foundation

# ## Task1:Prediction using supervised learning

# ## predict the % of a student based on the number of study hour

# ### import required library

# In[1]:


#import requied librarie used in this task
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ### loading the dataset

# In[2]:


#reading data from link
url = "http://bit.ly/w-data"
data = pd.read_csv(url)


# In[3]:


data.head(7)


# In[4]:


data.describe()


# In[5]:


data.isnull().sum()


# In[23]:


#plotting the distribution of scores 
data.plot(x='Hours', y='Scores', style='o',color='blue')  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()


# #### from tha above graph,we can see that there is a linear relationship between the number of hours studied and percentage of score

# In[7]:


X = data.iloc[:, :-1].values  
y = data.iloc[:, 1].values 


# In[8]:


X


# In[10]:


y


# ### Here we are spliting the data into 2 parts first one is training data 80%and the another one is testing data 20%

# In[11]:


from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                            test_size=0.2, random_state=0)


# ### TRAINING ALGORITHM

# In[12]:


from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit(X_train, y_train) 
print("Training complete.")


# In[24]:


#plotting the regression line
line = regressor.coef_*X+regressor.intercept_
plt.title("Linear regression vs trained model")
plt.scatter(X, y)
plt.xlabel('Hours studied')
plt.ylabel('Percentage Score')
plt.plot(X, line);
plt.show()


# In[14]:


#testing data
print(X_test)
#prediction the score
y_pred = regressor.predict(X_test)


# In[15]:


y_pred


# In[16]:


#compare actual vs prediction
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
df


# ## what will be the predicted score if a student studies for 9.25 hour per day?

# In[18]:


hours = 9.25
test=np.array([hours])
test=test.reshape(-1,1)
pred = regressor.predict([[9.5]])
print("No of Hours = {}".format(hours))
print("Predicted Score = {}".format(pred[0]))

