#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sklearn.gaussian_process as gp
from sklearn.gaussian_process.kernels import RBF
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import normalize
get_ipython().run_line_magic('matplotlib', 'inline')
#https://towardsdatascience.com/quick-start-to-gaussian-process-regression-36d838810319


# In[2]:


kernel = gp.kernels.RationalQuadratic(1.0,1.0,(1e-5,1e5),(1e-5,1e5))+gp.kernels.RBF(10.0,(1e-3,1e3))


# In[3]:


prntdata = pd.read_csv('Data matrix_gpd.csv')
X = prntdata.drop(['MFR_transient','20*20*D'], axis = 1)
y = prntdata['20*20*D']
prntdata.head()


# In[4]:


n_splits = 6
X_new = X.to_numpy()
y_new = y.to_numpy()

y_new = np.reshape(y_new,(-1,1))

X_new = normalize(X_new)
y_new = normalize(y_new)
y_new = y_new.flatten()
print(X_new)
print(y_new)


# In[7]:


kf = KFold(n_splits = n_splits, shuffle = True)
R_sqavg = 0

for train_index, test_index in kf.split(X):
    print("Train:", train_index, "Test:", test_index)
    X_train, X_test = X_new[train_index], X_new[test_index]
    y_train, y_test = y_new[train_index], y_new[test_index]
    ## Include all the below in the for loop to actually get the average
    model = gp.GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer = 10, alpha = 0.05)
    model.fit(X_train,y_train)
    params = model.kernel_.get_params()
    y_pred, std = model.predict(X_test,return_std = True)
    for x in range(len(y_pred)):
        if y_pred[x] <=0:
            y_pred[x] = 0
    corr_matrix = np.corrcoef(y_test, y_pred)
    corr = corr_matrix[0,1]
    R_sq = corr**2
    print("R_sq = ", R_sq)
    R_sqavg = R_sqavg+R_sq
R_sqavg = 1/n_splits*R_sqavg
print("R_sq Average = ", R_sqavg)


# ## Accuracy

# In[10]:


#Kinda usless but Ill leave it here
#mean_squared_error(y_test,y_pred)


# ## Kernel accuracy

# Just to keep track of which kerneral is best record below. Let everyone know if you are able to achieve a higher accuracy
# 
# kernel = gp.kernels.ConstantKernel(1.0, (1e-1, 1e3)) * gp.kernels.RBF(10.0, (1e-3, 1e3))
# R_sq for MF_transient = 0.9447329483802405
# R_sq for 20*20*D = 0.9432225425453269

# kernel = gp.kernels.DotProduct(1.0,(1e-3,1e3))*gp.kernels.RBF(10.0,(1e-3,1e3)) R_sq for 2020D = 0.9417821402796684
# 
# kernel = gp.kernels.DotProduct(1.0,(1e-3,1e3))*gp.kernels.RBF(10.0,(1e-3,1e3)) R_sq for MF_transient = 0.9440155736724359
# 
# kernel = gp.kernels.ExpSineSquared(1,1)*gp.kernels.RBF(10.0,(1e-3,1e3)) R_sq for MF_Transient = 0.9454176276635292
# 
# kernel = gp.kernels.ExpSineSquared(1,1)*gp.kernels.RBF(10.0,(1e-3,1e3)) R_sq for 2020D = 0.9436000165127506
# 
# kernel = gp.kernels.WhiteKernel(2)*gp.kernels.RBF(10.0,(1e-3,1e3)) R_sq for 2020D = 2.469030114314386e-32
# 
# kernel = gp.kernels.WhiteKernel(2)*gp.kernels.RBF(10.0,(1e-3,1e3)) R_sq for MFR_transient = nan
# 
# kernel = gp.kernels.WhiteKernel(1)*gp.kernels.RBF(10.0,(1e-3,1e3)) R_sq for 2020D = 2.469030114314386e-32
# 
# kernel = gp.kernels.Matern(1.0,(1e-5,1e5),1.5) R_sq for 2020D = 0.944840712797477
# 
# kernel = gp.kernels.Matern(1.0,(1e-5,1e5),1.5) R_sq for MFR_transient = 0.946145548318504
# 
# kernel = gp.kernels.RationalQuadratic(1.0,1.0,(1e-5,1e5),(1e-5,1e5))*gp.kernels.RBF(10.0,(1e-3,1e3)) R_sq for MFR_transient  = 0.94626321413892
# 
# kernel = gp.kernels.RationalQuadratic(1.0,1.0,(1e-5,1e5),(1e-5,1e5))*gp.kernels.RBF(10.0,(1e-3,1e3)) R_sq for 2020D  = 0.9448073657611128
# 
# From this point forward, K-fold cross validation is being used: (n_fold = 6), n_points = 2401
# 
# kernel = gp.kernels.ExpSineSquared(1,1)+gp.kernels.RBF(10.0,(1e-3,1e3)) R_sq for 2020D  = 0.980251945470363
# 
# kernel = gp.kernels.ExpSineSquared(1,1)+gp.kernels.RBF(10.0,(1e-3,1e3)) R_sq for MFR_Transient = 0.9584261678414293
# 
# kernel = gp.kernels.ExpSineSquared(1,1)*gp.kernels.RBF(10.0,(1e-3,1e3)) R_sq for MFR_Transient = 0.958949586878824
# 
# kernel = gp.kernels.ExpSineSquared(1,1)*gp.kernels.RBF(10.0,(1e-3,1e3)) R_sq for 2020D = 0.980381948552475
# 
# kernel = gp.kernels.DotProduct(1.0,(1e-5,1e5))*gp.kernels.RBF(10.0,(1e-5,1e3)) R_sq for 2020D = 0.9787512761810472
# 
# kernel = gp.kernels.DotProduct(1.0,(1e-5,1e5))*gp.kernels.RBF(10.0,(1e-5,1e3)) R_sq for MFR_transient = 0.9503969654257823
# 
# kernel = gp.kernels.WhiteKernel(2)*gp.kernels.RBF(10.0,(1e-3,1e3)) R_sq for MFR_transient = 6.418925610598585e-34
# 
# kernel = gp.kernels.WhiteKernel(2)*gp.kernels.RBF(10.0,(1e-3,1e3)) R_sq for 2020D = nan
# 
# kernel = gp.kernels.WhiteKernel(1)*gp.kernels.RBF(10.0,(1e-3,1e3)) R_sq for 202D = nan
# 
# kernel = gp.kernels.Matern(1.0,(1e-5,1e5),1.5) R_sq for 2020D = 0.9809589228670174
# 
# kernel = gp.kernels.Matern(1.0,(1e-5,1e5),1.5) R_sq for MFR_transient = 0.9633850265567655
# 
# kernel = gp.kernels.RationalQuadratic(1.0,1.0,(1e-5,1e5),(1e-5,1e5))*gp.kernels.RBF(10.0,(1e-3,1e3)) R_sq for MFR_transient = 0.9636922251152922
# 
# kernel = gp.kernels.RationalQuadratic(1.0,1.0,(1e-5,1e5),(1e-5,1e5))*gp.kernels.RBF(10.0,(1e-3,1e3)) R_sq for 2020D = 0.9812302388847343, 0.9812302390385466
# 
# kernel = gp.kernels.RationalQuadratic(1.0,1.5,(1e-5,1e5),(1e-5,1e5))*gp.kernels.RBF(10.0,(1e-3,1e3)) R_sq for 2020D = 
# 0.981230238504128
# 
# kernel = gp.kernels.RationalQuadratic(1.0,1.5,(1e-5,1e5),(1e-5,1e5))*gp.kernels.RBF(10.0,(1e-3,1e3)) R_sq for MFR_transient = 0.96369223000484
# 
# kernel = gp.kernels.RationalQuadratic(1.0,0.5,(1e-5,1e5),(1e-5,1e5))*gp.kernels.RBF(10.0,(1e-3,1e3)) R_sq for MFR_transient = 
# 0.963692228699684
# 
# kernel = gp.kernels.RationalQuadratic(1.0,0.5,(1e-5,1e5),(1e-5,1e5))*gp.kernels.RBF(10.0,(1e-3,1e3)) R_sq for 2020D = 0.9812302385716457
# 
# kernel = gp.kernels.RationalQuadratic(1.5,1.0,(1e-5,1e5),(1e-5,1e5))*gp.kernels.RBF(10.0,(1e-3,1e3)) R_sq for 202D = 0.9812302385381078
# 
# kernel = gp.kernels.RationalQuadratic(1.5,1.0,(1e-5,1e5),(1e-5,1e5))*gp.kernels.RBF(10.0,(1e-3,1e3)) R_sq for MFR_transient = 0.9636922292736972
# 
# kernel = gp.kernels.RationalQuadratic(0.5,1.0,(1e-5,1e5),(1e-5,1e5))*gp.kernels.RBF(10.0,(1e-3,1e3)) R_sq for MFR_transient = 0.9636922292026304
# 
# kernel = gp.kernels.RationalQuadratic(0.5,1.0,(1e-5,1e5),(1e-5,1e5))*gp.kernels.RBF(10.0,(1e-3,1e3)) R_sq for 2020D = 0.9812302385313526
# 
# kernel = gp.kernels.RationalQuadratic(1.0,1.0,(1e-5,1e5),(1e-5,1e5))+gp.kernels.RBF(10.0,(1e-3,1e3)) R_sq for 2020D = 0.9812327370588839
# 
# kernel = gp.kernels.RationalQuadratic(1.0,1.0,(1e-5,1e5),(1e-5,1e5))+gp.kernels.RBF(10.0,(1e-3,1e3)) R_sq for MFR_transient =
# 0.963708131601245
# 
# From this point forward, both X and y data are being normalized:
# 
# kernel = gp.kernels.RationalQuadratic(1.0,1.0,(1e-5,1e5),(1e-5,1e5))*gp.kernels.RBF(10.0,(1e-3,1e3)) R_sq for 2020D = 0.9765297729546001
# 
# kernel = gp.kernels.RationalQuadratic(1.0,1.0,(1e-5,1e5),(1e-5,1e5))+gp.kernels.RBF(10.0,(1e-3,1e3)) R_sq for 2020D = 0.9764099687534774

# 
