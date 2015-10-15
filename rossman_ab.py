import os
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
from pandas import *
from numpy import *
from sklearn.linear_model import *
from sklearn.preprocessing import OneHotEncoder

DATA_DIR = 'C:/Users/Guerzhoy/Desktop/utds/rossmann'
os.chdir(DATA_DIR)

random.seed(0)


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
    data_categ = data_categ.fillna(0)
    return data_categ
    


data = read_csv("data/train.csv", low_memory=False)
store_means = DataFrame(data.groupby("Store").mean()["Sales"])
store_means["Store"] = store_means.index
store_means.columns = ["SalesMean", "Store"]
data = data.merge(store_means, on = ["Store"])


store_data = read_csv("data/store.csv")


#Not doing anything with store_data for now...
data_categ = get_data_categ(data)
data_categ = concat([data_categ, data["SalesMean"]], axis = 1)
data_categ.index = range(len(data_categ))
data_cols = list(data_categ.columns)



data_test = read_csv("data/test.csv")
data_test = data_test.merge(store_means, on = ["Store"])

data_test_categ = get_data_categ(data_test)
data_test_categ = concat([data_test_categ, data_test["SalesMean"]], axis = 1)
data_test_x = data_test_categ.loc[:, data_cols]




data_x = data_categ.loc[:, data_cols]
data_y = data.loc[:, ["Sales"]]

idx = random.permutation(range(len(data_x)))
idx_train = idx[:int(.9*len(idx))]
idx_valid =  idx[int(.9*len(idx)):]

train_x = data_x.loc[idx_train,:]
train_x.index = range(len(train_x))
train_y = data_y.loc[idx_train,:]
train_y.index = range(len(train_y))

valid_x = data_x.loc[idx_valid, :]
valid_x.index = range(len(valid_x))
valid_y = data_y.loc[idx_valid, :]
valid_y.index = range(len(valid_y))

train_x["SalesMean"] = log(1+train_x["SalesMean"])
valid_x["SalesMean"] = log(1+valid_x["SalesMean"])
data_test_x["SalesMean"] = log(1+data_test_x["SalesMean"])

lr = RidgeCV(alphas= [.75, 1, 1.25, 1.5, 2, 10, 20, 100])
lr.fit(train_x.as_matrix(), log(1+train_y.as_matrix().reshape((len(train_y),))))

df = DataFrame()
df["var"] = valid_x.columns
df["coef"] = lr.coef_


pred_valid = exp(lr.predict(valid_x.as_matrix()))-1
actual_valid = valid_y.as_matrix().reshape((len(valid_y),))

pred_valid = pred_valid[actual_valid>0]
actual_valid = actual_valid[actual_valid>0]
#validation RMSPE
sqrt(mean( ((pred_valid-actual_valid)/actual_valid)**2 ))

#sqrt(mean((lr.predict(train_x.as_matrix()).flatten()-train_y.as_matrix().flatten())**2))
#sqrt(mean((lr.predict(valid_x.as_matrix()).flatten()-valid_y.as_matrix().flatten())**2))

submit = True
if submit:
    submission = DataFrame()
    submission["Id"] = data_test["Id"]
    pred = lr.predict(data_test_x.as_matrix())
    pred = around(pred.astype(double)).astype(int32)
    pred = maximum(0, pred)
    
    submission["Sales"] = pred
    submission.to_csv("submission.csv", index=False)

