

import pandas as pd
from pandas import Series,DataFrame

# numpy, matplotlib, seaborn
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns
#sns.set_style('whitegrid')
%matplotlib inline

train      = pd.read_csv('train.csv')
test       = pd.read_csv('test.csv')
severity   = pd.read_csv("severity_type.csv")
resource   = pd.read_csv("resource_type.csv")
event      = pd.read_csv("event_type.csv")
log        = pd.read_csv("log_feature.csv")

colnames = ["id", "location"]
colnames = colnames + ["event_type " + str(x) for x in range(1,55)]
colnames = colnames + ["feature " + str(x) for x in range(1,387)]
colnames = colnames + ["severity_type " + str(x) for x in range(1,6)]
colnames = colnames + ["resource_type " + str(x) for x in range(1,11)]
colnames = colnames + ["ne", "nf", "nr", "sumvolume", "avgvolume", "minvolume", "maxvolume"]
nbf = len(colnames) - 1
#colnames = colnames + ["rizsa " + str(x) for x in range(nbf*(nbf+1)/2)]

y = train["fault_severity"]
train = train.drop("fault_severity", axis = 1)
T = train.append(test)
N = pd.DataFrame(0, index = np.arange(len(T)), columns = colnames[2:])
T = pd.concat([T.reset_index(), N] , axis = 1)
T = T.drop(["index"], axis = 1)
T.location = T.location.apply(lambda x: int(x[9:]))

iui = np.triu_indices(463)

for i in range(len(T)):
    #print i
    id = T.id[i]
    T.loc[i, severity.severity_type[severity.id == id]] = 1
    #for j in severity.severity_type[severity.id == id]:
    #    T.loc[i, "severity_type "] = int(j[14:])
    
    for j in event.event_type[event.id == id]:
        T.loc[i, j] = 1
        T.loc[i, "ne"] += 1 

    for j in resource.resource_type[resource.id == id]:
        #jnum = int(j[14:])
        #if jnum not in [1, 3, 5, 9, 10]:
        T.loc[i, j] = 1
        #else:
        #    T.loc[i, "resource_type 1"] = 1
        T.loc[i, "nr"] += 1 
    
    L = log.loc[log.id == id,:]
    for j in L.index.values:
        #if L.log_feature[j] not in features:
        #    T.loc[i, "feature 1"] = T.loc[i, "feature 1"] + L.volume[j]
        #else:
        T.loc[i, L.log_feature[j]] = L.volume[j]   
        
    T.loc[i, "nf"] = len(L)
    T.loc[i, "sumvolume"] = L.volume.sum()
    T.loc[i, "avgvolume"] = L.volume.mean()
    T.loc[i, "minvolume"] = L.volume.min()
    T.loc[i, "maxvolume"] = L.volume.max()
    
    #sz = 0
    #for j in range(1, nbf+1):
    #    print j
    #    for k in range(j, nbf+1):
    #        T.iloc[i, j*(927-j)/2 + k] = T.iloc[i, j] * T.iloc[i, k] 
            #sz = sz + 1 
        
    #A = np.mat(T.iloc[i,1:(nbf+1)])
    #B = A.T * A
    #T.iloc[i, (nbf+1):] = B[iui]

T = T.drop("id", axis = 1)

T["feature 202"].sum() == log[log.log_feature == "feature 202"].volume.sum()

from sklearn.preprocessing import scale
from sklearn.cross_validation import train_test_split
from sklearn.metrics import log_loss

#scT = pd.DataFrame(scale(T[["feature " + str(x) for x in range(1,387)]]))
#T = pd.concat([T.ix[:,0:55], scT, T.ix[:,441:]], axis = 1, ignore_index = True)
#T.columns = colnames[2:]

Train, Test, y_train, y_test = train_test_split(T[0:(len(y))], y, train_size=0.7, random_state=88)

Train = pd.DataFrame(Train, columns=T.columns)
Test = pd.DataFrame(Test, columns=T.columns)



from sklearn.ensemble import RandomForestClassifier 

# Create the random forest object which will include all the parameters
# for the fit
forest = RandomForestClassifier(n_estimators = 1000, max_depth = 8)

forest = forest.fit(Train, y_train)

# Take the same decision trees and run it on the test data
output = forest.predict_proba(Test)
print log_loss(y_test, output)

import xgboost as xgb

xgb_params = {"objective" : "multi:softprob",
              "eval_metric" : "mlogloss",
              "num_class" : 3,
              "nthread" : 8, 
              "eta": 0.025, 
              "max_depth": 6}
num_rounds = 1000

dtrain = xgb.DMatrix(Train, label = y_train)
gbdt = xgb.train(xgb_params, dtrain, num_rounds)

output = gbdt.predict(xgb.DMatrix(Test))
print log_loss(y_test, output)

dtrain = xgb.DMatrix(T[0:(len(y))], label = y, feature_names = colnames[1:])
gbdt = xgb.train(xgb_params, dtrain, num_rounds)

output = gbdt.predict(xgb.DMatrix(T[len(y):]))

submission = pd.DataFrame({ 'id': test.id,
                            'predict_0': [row[0] for row in output],
                            'predict_1': [row[1] for row in output],
                            'predict_2': [row[2] for row in output]})
submission.to_csv("Psubmissions/submission2016.csv", index=False)
