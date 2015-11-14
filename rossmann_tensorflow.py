import os
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
from pandas import *
from numpy import *
from sklearn.linear_model import *
from sklearn.preprocessing import OneHotEncoder

import time

DATA_DIR = '/home/guerzhoy/Desktop/UTDS/rossmann'
os.chdir(DATA_DIR)

random.seed(0)

def convert_date(date_str):
    pattern = '%Y-%m-%d'
    epoch = int(time.mktime(time.strptime(date_str, pattern)))
    return epoch//86400


def get_data_categ(data):
    data.loc[:, "StateHoliday"] = data.loc[:,"StateHoliday"].astype(str)
    data.loc[data.loc[:,"StateHoliday"] != '0', "StateHoliday"]  = 'a'
    data.loc[:, "Promo"] = data.loc[:, "Promo"].astype(int)
    data.loc[:, "Open"] = data.loc[:, "Promo"].astype(int)
    data_categ = DataFrame()
    #data_categ = concat([data_categ, get_dummies(data.loc[:,"Store"], prefix = ["STORE"])], axis = 1)
    data_categ = concat([data_categ, get_dummies(data.loc[:,"DayOfWeek"], prefix = ["DAY"])], axis = 1)
    data_categ = concat([data_categ, get_dummies(data.loc[:,"StateHoliday"], prefix = ["StH"])], axis = 1)
    data_categ = concat([data_categ, get_dummies(data.loc[:,"SchoolHoliday"], prefix = ["SchH"])], axis = 1)
    data_categ = concat([data_categ, get_dummies(data.loc[:,"Promo"], prefix = ["PROMO"])], axis = 1)
    data_categ = concat([data_categ, get_dummies(data.loc[:,"Open"], prefix = ["OPEN"])], axis = 1)
    #data_categ = concat([data_categ, get_dummies(data.loc[:,"month"], prefix = ["MONTH"])], axis = 1)
    data_categ = concat([data_categ, get_dummies(data.loc[:,"season"], prefix = ["SEASON"])], axis = 1)
    data_categ = data_categ.fillna(0)
    return data_categ
    




data = read_csv("data/train.csv", low_memory=False)
data = data.loc[data["Sales"]!=0,:]

################################################################################
data = data.loc[logical_and(120 <= data["Store"], data["Store"] < 121),:]
################################################################################




store_means = DataFrame(data.groupby("Store").median()["Sales"])
store_means["Store"] = store_means.index
store_means.columns = ["SalesMean", "Store"]
data = data.merge(store_means, on = ["Store"])


data["epoch"] = data["Date"].apply(convert_date)
data["season"] = (data["epoch"]/91).astype(int)%4

#read in the store data
store_data = read_csv("data/store.csv")


#Not doing anything with store_data for now...
data_categ = get_data_categ(data)
#data_categ = concat([data_categ, data["SalesMean"]], axis = 1)
data_categ.index = range(len(data_categ))
data_cols = list(data_categ.columns)




data_x = data_categ.loc[:, data_cols]
data_y = data.loc[:, ["Sales"]]

#random.seed(0)
idx = random.permutation(range(len(data_x)))
idx_train = idx[:int(.8*len(idx))]

##############################################################################
#idx_train = idx_train[where(data.loc[idx_train, "Store"] == 50)]
##############################################################################

idx_valid =  idx[int(.8*len(idx))+1:int(.9*len(idx))]
idx_valid2 = idx[int(.9*len(idx))+1:]

###############################################################################
#idx_valid = idx_valid[where(data.loc[idx_valid, "Store"] == 50)]
###############################################################################


train_x = data_x.loc[idx_train,:]
train_x.index = range(len(train_x))
train_y = data_y.loc[idx_train,:]
train_y.index = range(len(train_y))

valid_x = data_x.loc[idx_valid, :]
valid_x.index = range(len(valid_x))
valid_y = data_y.loc[idx_valid, :]
valid_y.index = range(len(valid_y))

valid2_x = data_x.loc[idx_valid2, :]
valid2_x.index = range(len(valid2_x))
valid2_y = data_y.loc[idx_valid2, :]
valid2_y.index = range(len(valid2_y))


# store_ind = 120
# train_x.to_csv("store%d_train_x.csv" % store_ind, index=False)
# train_y.to_csv("store%d_train_y.csv" % store_ind, index=False)
# valid_x.to_csv("store%d_valid_x.csv" % store_ind,  index=False)
# valid_y.to_csv("store%d_valid_y.csv" % store_ind, index=False)
# valid2_x.to_csv("store%d_valid2_x.csv" % store_ind,  index=False)
# valid2_y.to_csv("store%d_valid2_y.csv" % store_ind,  index=False)


import tensorflow as tf

################################################################################

def get_train_batch(train_x, train_y, batch_size):
    idx = random.permutation(range(len(train_x)))[:batch_size]
    return train_x.loc[idx,:].values, train_y.loc[idx,:].values
    
################################################################################


#Linear regression for a single store, with different kinds of errors

data_dim = train_x.shape[1]   #17
output_dim = 1
hidden_dim = 5



#Linear Model###################################################################

x = tf.placeholder("float", [None, data_dim])
W = tf.Variable(tf.zeros([data_dim, output_dim]))
b = tf.Variable(tf.zeros([output_dim]))
y = tf.matmul(x, W) + b

################################################################################

#Hidden layer model

# hidden_dim = 1
# x = tf.placeholder("float", [None, data_dim])
# W1 = tf.Variable(tf.zeros([data_dim, hidden_dim]))
# b1 = tf.Variable(tf.zeros([hidden_dim]))
# #h = tf.nn.tanh(tf.matmul(x, W1) + b1)
# 
# W2 = tf.Variable(tf.zeros([hidden_dim, output_dim]))
# b2 = tf.Variable(tf.zeros([output_dim]))
# y =  tf.matmul(h, W2) + b2






################################################################################




#Training
y_ = tf.placeholder("float", [None, output_dim])
MSE = tf.reduce_mean(  tf.pow((y_-y), 2)  )
RMR = tf.reduce_mean( tf.pow( (y_-y)/y_, 2 ))


train_step_MSE = tf.train.GradientDescentOptimizer(0.01).minimize(MSE)
train_step_RMR = tf.train.GradientDescentOptimizer(1000.).minimize(RMR)

sess = tf.Session()
sess.run(tf.initialize_all_variables())
for i in range(10000):
    batch_xs, batch_ys = get_train_batch(train_x, train_y, 500)
    if i < 100:
        sess.run(train_step_MSE, feed_dict={x:batch_xs, y_:batch_ys})
    else:
        print "Now optimizing RMR"
        sess.run(train_step_RMR, feed_dict={x:batch_xs, y_:batch_ys})
    #sess.run(train_step_RMR, feed_dict={x:batch_xs, y_:batch_ys})
    
    print "sqrt(MSE) Training: %g" % (sqrt(MSE.eval(feed_dict={x:train_x.values, y_:train_y.values}, session = sess)))
    print "sqrt(MSE) Validation: %g" % (sqrt(MSE.eval(feed_dict={x:valid_x.values, y_:valid_y.values}, session = sess)))
    
    print "sqrt(RMR) Training: %g" % (sqrt(RMR.eval(feed_dict={x:train_x.values, y_:train_y.values}, session = sess)))
    print "sqrt(RMR) Validation: %g" % (sqrt(RMR.eval(feed_dict={x:valid_x.values, y_:valid_y.values}, session = sess)))
    
    print("====================================================================")
    
    
        

