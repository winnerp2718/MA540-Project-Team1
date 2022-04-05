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
get_ipython().run_line_magic('matplotlib', 'inline')
#https://towardsdatascience.com/quick-start-to-gaussian-process-regression-36d838810319


# In[158]:


kernel = gp.kernels.RationalQuadratic(1.0,1.0,(1e-5,1e5),(1e-5,1e5))*gp.kernels.RBF(10.0,(1e-3,1e3))


# In[159]:


prntdata = pd.read_csv('Datamatrix_two.csv')
X = prntdata.drop(['MFR_transient','MFR1(under blade)','20*20*D','20*20*2D','20*20*3'], axis = 1)
y = prntdata['20*20*D']
prntdata.head()


# In[160]:


print(X)
kf = KFold(n_splits = 6)
X_new = X.to_numpy()
y_new = y.to_numpy()
print(X_new)
print(y_new)

for train_index, test_index in kf.split(X):
    print("Train:", train_index, "Test:", test_index)
    X_train, X_test = X_new[train_index], X_new[test_index]
    y_train, y_test = y_new[train_index], y_new[test_index]


# In[161]:


model = gp.GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=0.1, normalize_y=True)


# In[162]:


model.fit(X_train, y_train)
params = model.kernel_.get_params()


# In[163]:


y_pred, std = model.predict(X_test, return_std=True)


# plt.plot(X.iloc[:,0], y, 'r.', markersize=5, label=u'Observation') 
# plt.plot(X_test.iloc[:,0], y_pred, 'b-',linewidth=1, label=u'Prediction')
# plt.fill_between(X.iloc[:,0], y_pred - 1.96*std, y_pred.iloc[:,0] +1.96*std, alpha = 0.2, color='k',label=u'95 % confidence interval')
# plt.xlabel('(a)')
# plt.legend (loc='upper right', fontsize=10)

# In[164]:


#Make negative values 0 
for x in range(len(y_pred)):
    if y_pred[x] <= 0:
        y_pred[x] = 0


# ## Accuracy

# In[165]:


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
# 
# From this point forward, K-fold cross validation is being used: (n_fold = 6), n_points = 625
# 
# kernel = gp.kernels.DotProduct(1.0,(1e-5,1e3))*gp.kernels.RBF(10.0,(1e-3,1e3)) R_sq for 2020D = 0.9380104377144214
# 
# kernel = gp.kernels.DotProduct(1.0,(1e-5,1e3))*gp.kernels.RBF(10.0,(1e-3,1e3)) R_sq for MFR_transient = 0.9098988001562424
# 
# kernel = gp.kernels.ExpSineSquared(1,1)*gp.kernels.RBF(10.0,(1e-3,1e3)) R_sq for MFR_transient = 0.9350470547903161
# 
# kernel = gp.kernels.ExpSineSquared(1,1)*gp.kernels.RBF(10.0,(1e-3,1e3)) R_sq for 2020D = 0.9392785337232372
# 
# kernel = gp.kernels.WhiteKernel(2)*gp.kernels.RBF(10.0,(1e-3,1e3)) R_sq for 2020D = 1.7660188495464636e-33
# 
# kernel = gp.kernels.WhiteKernel(2)*gp.kernels.RBF(10.0,(1e-3,1e3)) R_sq for MFR_transient = nan
# 
# kernel = gp.kernels.WhiteKernel(1)*gp.kernels.RBF(10.0,(1e-3,1e3)) R_sq for 2020D = 1.7660188495464636e-33
# 
# kernel = gp.kernels.Matern(1.0,(1e-5,1e5),1.5) R_sq for 2020D = 0.9388230928778297
# 
# kernel = gp.kernels.Matern(1.0,(1e-5,1e5),1.5) R_sq for MFR_transient = 0.9242899696209131
# 
# kernel = gp.kernels.RationalQuadratic(1.0,1.0,(1e-5,1e5),(1e-5,1e5))*gp.kernels.RBF(10.0,(1e-3,1e3)) R_sq for MFR_transient = 0.9220780977448653
# 
# kernel = gp.kernels.RationalQuadratic(1.0,1.0,(1e-5,1e5),(1e-5,1e5))*gp.kernels.RBF(10.0,(1e-3,1e3)) R_sq for 2020D = 0.9397696026560236, 0.9397696026560236, 0.9397715802328483 
# 
# kernel = gp.kernels.ExpSineSquared(1,1)+gp.kernels.RBF(10.0,(1e-3,1e3)) R_sq for 2020D  = 0.9397049097518393
# 
# kernel = gp.kernels.ExpSineSquared(1,1)+gp.kernels.RBF(10.0,(1e-3,1e3)) R_sq for MFR_Transient = 0.9191484120084201
# 
# kernel = gp.kernels.RationalQuadratic(1.0,1.0,(1e-5,1e7),(1e-5,1e7))+gp.kernels.RBF(10.0,(1e-3,1e3)) R_sq for MFR_transient = 0.9204857223606515
# 
# kernel = gp.kernels.RationalQuadratic(1.0,1.0,(1e-5,1e7),(1e-5,1e7))+gp.kernels.RBF(10.0,(1e-3,1e3)) R_sq for 2020D = 0.9397944008640747 
# 
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
# kernel = gp.kernels.RationalQuadratic(1.0,1.0,(1e-5,1e5),(1e-5,1e5))*gp.kernels.RBF(10.0,(1e-3,1e3)) R_sq for 2020D = 0.9812302388847343
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

# 
