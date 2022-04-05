#!/usr/bin/env python
# coding: utf-8

# In[2]:


import sklearn.gaussian_process as gp
from sklearn.gaussian_process.kernels import RBF
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
get_ipython().run_line_magic('matplotlib', 'inline')
#https://towardsdatascience.com/quick-start-to-gaussian-process-regression-36d838810319


# In[3]:


kernel = gp.kernels.ConstantKernel(1.0, (1e-1, 1e3)) * gp.kernels.RBF(10.0, (1e-3, 1e3))


# In[5]:


prntdata = pd.read_csv('Datamatrix_two.csv')
X = prntdata.drop(['MFR_transient','MFR1(under blade)','20*20*D','20*20*2D','20*20*3'], axis = 1)
y = prntdata['20*20*D']
prntdata.head()


# In[4]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# In[5]:


model = gp.GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=0.1, normalize_y=True)


# In[6]:


model.fit(X_train, y_train)
params = model.kernel_.get_params()


# In[7]:


y_pred, std = model.predict(X_test, return_std=True)


# plt.plot(X.iloc[:,0], y, 'r.', markersize=5, label=u'Observation') 
# plt.plot(X_test.iloc[:,0], y_pred, 'b-',linewidth=1, label=u'Prediction')
# plt.fill_between(X.iloc[:,0], y_pred - 1.96*std, y_pred.iloc[:,0] +1.96*std, alpha = 0.2, color='k',label=u'95 % confidence interval')
# plt.xlabel('(a)')
# plt.legend (loc='upper right', fontsize=10)

# In[8]:


#Make negative values 0 
for x in range(len(y_pred)):
    if y_pred[x] <= 0:
        y_pred[x] = 0


# ## Accuracy

# In[9]:


corr_matrix = np.corrcoef(y_test, y_pred)
corr = corr_matrix[0,1]
R_sq = corr**2
R_sq


# In[10]:


#Kinda usless but Ill leave it here
#mean_squared_error(y_test,y_pred)


# ## Kernel accuracy

# Just to keep track of which kerneral is best record below. Let everyone know if you are able to achieve a higher accuracy
# 
# kernel = gp.kernels.ConstantKernel(1.0, (1e-1, 1e3)) * gp.kernels.RBF(10.0, (1e-3, 1e3))
# R_sq for MF_transient = 0.9447329483802405
# R_sq for 20*20*D = 0.9432225425453269
