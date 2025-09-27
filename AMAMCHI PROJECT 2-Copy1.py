#!/usr/bin/env python
# coding: utf-8

# ## TIPS PREDICTION USING A REGRESSOR MODEL
# 
# ### BY OKOJI AMAMCHINEFE 
# 
# The objective of the regression task is to predict the amount of tip (gratuity in Nigeria naira) given to a food server based on these factors( total_bill, gender, smoker (whether they smoke in the party or not), day (day of the week for the party), time (time of the day whether for lunch or dinner), and size (size of the party)) in Mama Tee restaurant.
# 

# ## Importing the libraries

# In[1]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# ## Loading the dataset/viewing the first 5 records

# In[2]:


Tips_df = pd.read_csv("tips.csv") 

Tips_df.head()


# In[3]:


#numbers of row and colums
Tips_df.shape


# In[4]:


#checking for nonempty columns and datatype
Tips_df.info()


# In[5]:


#summarise the numeric datatype and count of each column
Tips_df.describe()


# ### Clean the dataset

# In[6]:


#show the column names
Tips_df.columns


# In[7]:


# Check for missing value
Tips_df.isnull().sum()


# ### Data Visualisation

# In[8]:


# Histogram
sns.histplot(data=Tips_df, x='tip', bins=20, kde=True) 
plt.title('Distribution of Tip Amounts')
plt.xlabel('Tip')
plt.ylabel('Frequency')
plt.show()

#bins=20 controls the number of intervals.
#kde=True adds a smooth curve to show the density.


# In[9]:


# Distribution of tips across gender, smoker, day, and time
sns.catplot(data=Tips_df, x="gender", y="tip", col="smoker", row="time", kind="box", hue="day")
plt.suptitle("Tip Distribution by Gender, Smoker, Day, and Time", y=1.02)
plt.show()


# In[10]:


sns.scatterplot(data=Tips_df, x="total_bill", y="tip", hue="gender", style="smoker")
plt.title("Relationship Between Total Bill and Tip")
plt.show()


# In[11]:


print(Tips_df.groupby(["gender", "smoker"])["tip"].describe())


# In[12]:


sns.boxplot(data=Tips_df, x="gender", y="tip", hue="smoker")
plt.title("Tip Amount by Gender and Smoking Status")
plt.show()


# In[13]:


# Box plot: tip by gender
sns.boxplot(data=Tips_df, x='gender', y='tip')
plt.title('Tip Distribution by Gender')
plt.show()


# In[14]:


# Bar chart: average tip by day
sns.barplot(data=Tips_df, x='day', y='tip', estimator=lambda x: sum(x)/len(x))
plt.title('Average Tip by Day')
plt.show()


# In[15]:


# Histogram: total_bill
sns.histplot(data=Tips_df, x='total_bill', bins=20, kde=True)
plt.title('Distribution of Total Bill')
plt.show()


# In[16]:


# Pie chart: smoker proportion
smoker_counts = Tips_df['smoker'].value_counts()
plt.pie(smoker_counts, labels=smoker_counts.index, autopct='%1.1f%%', startangle=90)
plt.title('Smoker vs Non-Smoker Proportion')
plt.show()


# ## Data Preparation

# In[17]:


y = Tips_df["tip"] # Add Target name
X = Tips_df.drop("tip", axis=1) 


# In[18]:


# Target variable
y = Tips_df["tip"]

# Feature matrix (excluding the target)
X = Tips_df.drop("tip", axis=1)

# One-hot encode categorical variables
X = pd.get_dummies(X, drop_first=True)

# Optional: Check the transformed feature set
print(X.head())


# ### DataSegmentation

# ### Load The data into the Linear Regression model i.e Model Building

# In[19]:


# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Fit model
Lreg = LinearRegression()
Lreg.fit(X_train, y_train)

# Evaluate
print("Lreg score:", Lreg.score(X_test, y_test))
print("Coefficients:", pd.Series(Lreg.coef_, index=X.columns))


# ### Make predictions

# In[20]:


Lreg.predict(X_test)


# ## Evaluate Linear regression model

# In[21]:


from sklearn.metrics import mean_absolute_error

# TODO: Evaluate your model using the following
# - R Square/Adjusted R Square
# - Mean Square Error(MSE)/Root Mean Square Error(RMSE)
# - Mean Absolute Error(MAE)

y_pred = Lreg.predict(X_test)

print("MAE",mean_absolute_error(y_test,y_pred))


# # Other Machine Learning Algorithms

# ## Decision Tree Model

# In[22]:


from sklearn.tree import DecisionTreeRegressor


# In[23]:


# Decision Tree
dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)


# In[24]:


print("MAE",mean_absolute_error(y_test,y_pred_dt))


# ## Random Forest Regressor

# In[25]:


from sklearn.ensemble import RandomForestRegressor


# In[26]:


# Random Forest
rf = RandomForestRegressor(random_state=42, n_estimators=100)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)


# In[27]:


print("MAE",mean_absolute_error(y_test,y_pred_rf))


# ## Evaluate/Compare models

# In[28]:


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# In[29]:


results = pd.DataFrame({
    "Model": ["Linear Regression", "Decision Tree", "Random Forest"],
    "MAE": [
        mean_absolute_error(y_test, y_pred),
        mean_absolute_error(y_test, y_pred_dt),
        mean_absolute_error(y_test, y_pred_rf)
    ],
    "RMSE": [
        np.sqrt(mean_squared_error(y_test, y_pred)),
        np.sqrt(mean_squared_error(y_test, y_pred_dt)),
        np.sqrt(mean_squared_error(y_test, y_pred_rf))
    ],
    "R2 Score": [
        r2_score(y_test, y_pred),
        r2_score(y_test, y_pred_dt),
        r2_score(y_test, y_pred_rf)
    ]
})

print("\n===== Model Comparison Table =====")
print(results)


# ## Visualize Predictions

# In[30]:


plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.xlabel("Actual Tips")
plt.ylabel("Predicted Tips")
plt.title("Linear Regression: Actual vs Predicted")
plt.plot([0, max(y_test)], [0, max(y_test)], 'r--')
plt.show()


# In[39]:


plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred_dt, alpha=0.7)
plt.xlabel("Actual Tips")
plt.ylabel("Predicted Tips")
plt.title("Decision Tree: Actual vs Predicted")
plt.plot([0, max(y_test)], [0, max(y_test)], 'r--')
plt.show()


# In[31]:


plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred_rf, alpha=0.7)
plt.xlabel("Actual Tips")
plt.ylabel("Predicted Tips")
plt.title("Random Forest: Actual vs Predicted")
plt.plot([0, max(y_test)], [0, max(y_test)], 'r--')
plt.show()


# ## Save the best performing Model

# In[32]:


import pickle


# In[33]:


# Save LR model
with open("model.pkl", "wb") as f:
    pickle.dump(Lreg, f)

print("✅ Model trained and saved as model.pkl")


# In[34]:


feature_importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print(feature_importance)


# ## Insights:
# Total_bill is the strongest predictor — since tips often scale with the bill amount.
# 
# size (number of people) can influence tipping behavior, especially in larger groups.
# 
# Gender and smoker have subtle effects, possibly due to cultural or behavioral patterns.
# 
# Day and Time may reflect different service expectations (e.g., weekend dinners vs weekday lunches).
# 

# In[ ]:





# In[ ]:




