# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 14:05:18 2018

@author: ADMIN
"""

import numpy as np
import pandas as pd
import csv
import urllib, json
url = "https://---------"
response = urllib.request.urlopen(url)
dataset = json.loads(response.read())
diction=(dataset.get('data'))
dat=pd.DataFrame(eval(diction))
desc=(dataset.get('description'))
des=pd.DataFrame(desc)


n=des.to_dict()
dicnew=dict(n['description'])
dicnew2=dict(n['tag'])
d={}
for i in range(0,17):
    d[str(dicnew2[i])]=str(dicnew[i])
dat.to_csv('data.csv', encoding='utf-8', index=False)
dat=pd.read_csv('data.csv')


"""
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
dat = sc.fit_transform(dat)
dat=sc.transform(dat)
"""

newd=dat.loc[:,'EI_27901':'TISA_11201']
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
newd = sc.fit_transform(newd)

newd=pd.DataFrame((newd))
import seaborn as sb
sb.pairplot(dat,hue='time',palette='hls')

from sklearn.cluster import DBSCAN
DB = DBSCAN(eps=0.1, min_samples=25)
DB.fit(dat)

from collections import Counter

print(Counter(DB.labels_))
print(dat[DB.labels_==-1])
    
from sklearn import svm
outliers_fraction = 0.01 

nu_estimate = 0.95 * outliers_fraction + 0.05
auto_detection = svm.OneClassSVM(kernel='rbf', gamma=0.00025, nu=nu_estimate)
auto_detection.fit(newd)
evaluation = auto_detection.predict(newd)
var=newd[evaluation==-1]
print(newd[evaluation==-1])

