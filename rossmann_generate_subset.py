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

DATA_DIR = 'C:/Users/Guerzhoy/Desktop/utds/rossmann/'
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
data_categ = concat([data_categ, data["SalesMean"]], axis = 1)
data_categ.index = range(len(data_categ))
data_cols = list(data_categ.columns)



data_test = read_csv("data/test.csv")
data_test = data_test.merge(store_means, on = ["Store"])
data_test["epoch"] = data_test["Date"].apply(convert_date)
data_test["season"] = (data["epoch"]/91).astype(int)%4
#data_test["month"] = data_test["epoch"]//12


data_test_categ = get_data_categ(data_test)
data_test_categ = concat([data_test_categ, data_test["SalesMean"]], axis = 1)
#data_test_x = data_test_categ.loc[:, data_cols]




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


store_ind = 120
train_x.to_csv("store%d_train_x.csv" % store_ind, index=False)
train_y.to_csv("store%d_train_y.csv" % store_ind, index=False)
valid_x.to_csv("store%d_valid_x.csv" % store_ind,  index=False)
valid_y.to_csv("store%d_valid_y.csv" % store_ind, index=False)
valid2_x.to_csv("store%d_valid2_x.csv" % store_ind,  index=False)
valid2_y.to_csv("store%d_valid2_y.csv" % store_ind,  index=False)




