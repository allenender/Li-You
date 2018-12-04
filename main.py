# Code by Li You
# 2018/11/15
import csv
import re
from sklearn import preprocessing
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn import svm
from sklearn.linear_model import LinearRegression
from sklearn import neighbors
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor

# store features of string type
penalty_type = ['none', 'l2', 'l1', 'elasticnet']
type_list = [penalty_type]
type_default = ['l2']

penalty = 0
l1_ratio = 1
alpha = 2
max_iter = 3
random_state = 4
n_jobs = 5
n_samples = 6
n_features = 7
n_classes = 8
n_clusters_per_class = 9
n_informative = 10
flip_y = 11
scale = 12
size = 13
clusters = 14
penalty_change =15

# read train feaures
with open('train.csv', 'r') as f:
    reader = csv.reader(f)
    training_arr = list(reader)
training_arr.pop(0)
f.close()

# change penalty to int
for i in range(0, len(training_arr)):
    for j in range(0, len(training_arr[i])):
        if training_arr[i][j]=='none':
            training_arr[i][j]='0'
            continue
        if training_arr[i][j]=='l2':
            training_arr[i][j]='3'
            continue
        if training_arr[i][j]=='l1':
            training_arr[i][j]='1'
            continue
        if training_arr[i][j]=='elasticnet':
            training_arr[i][j]='2'
            continue

# read test features
with open('test.csv', 'r') as f:
    reader = csv.reader(f)
    testing_arr = list(reader)
testing_arr.pop(0)
f.close()

# change penalty to number
for i in range(0, len(testing_arr)):
    for j in range(0, len(testing_arr[i])):
        if testing_arr[i][j]=='none':
            testing_arr[i][j]='0'
            continue
        if testing_arr[i][j]=='l2':
            testing_arr[i][j]='3'
            continue
        if testing_arr[i][j]=='l1':
            testing_arr[i][j]='1'
            continue
        if testing_arr[i][j]=='elasticnet':
            testing_arr[i][j]='1'
            continue

arr_y = []
arr_x = []
for i in range(0, len(training_arr)):
    arr_y.append(float(training_arr[i][14]))
    training_arr[i].pop(0)
    training_arr[i].pop()

for i in range(0, len(testing_arr)):
    testing_arr[i].pop(0)

for i in range(0, len(training_arr)):
    for j in range(0, len(training_arr[0])):
        training_arr[i][j] = float(training_arr[i][j])
        if training_arr[i][j]<0:
           training_arr[i][j] = 8

for i in range(0, len(testing_arr)):
    for j in range(0, len(testing_arr[0])):
        testing_arr[i][j] = float(testing_arr[i][j])
        if testing_arr[i][j]<0:
           testing_arr[i][j] = 8

#for normzalization (method 1)(not used at last)
def get_max_value(martix):
    res_list=[]
    for j in range(len(martix[0])):
        one_list=[]
        for i in range(len(martix)):
            one_list.append(martix[i][j])
        res_list.append(max(one_list))
    return res_list


def get_min_value(martix):
    res_list=[]
    for j in range(len(martix[0])):
        one_list=[]
        for i in range(len(martix)):
            one_list.append(martix[i][j])
        res_list.append(min(one_list))
    return res_list


#for normzalization (method 2)(not used at last)
def normalization(x, max, min):
    x = (x - min) / (max - min);
    return x
def featureselect(matrix, features):
    newmatrix = []
    for line in range(0, len(matrix)):
        newmatrix.append([])
        for i in features:
            newmatrix[line].append(matrix[line][i])
    return newmatrix

#do dome feature engineering
def featureprocess(matrix):
    for line in range(0, len(matrix)):
        tem = matrix[line][flip_y]
        matrix[line][flip_y] = float(1) / tem
    for line in range(0, len(matrix)):
        matrix[line].append(matrix[line][n_samples] * matrix[line][n_features] * matrix[line][n_classes] * \
                            matrix[line][n_clusters_per_class] * matrix[line][n_informative] / float(100000))
        matrix[line].append(matrix[line][n_classes] * matrix[line][n_clusters_per_class] / float(10))
        matrix[line].append(matrix[line][penalty] * matrix[line][l1_ratio])
    for line in range(0, len(matrix)):
        tem = matrix[line][n_jobs]
        matrix[line][n_jobs] = float(1) / tem
    #for line in range(0, len(matrix)):
    #    tem1 = matrix[line][penalty]
    #    tem2 = matrix[line][l1_ratio]
    #    if tem1 == 3:
    #        matrix[line][penalty] = 2-matrix[line][l1_ratio]
    #return


featureprocess(training_arr)
featureprocess(testing_arr)

featurevalues = [penalty,max_iter,n_jobs,flip_y,size]
#featurevalues = [penalty,l1_ratio,alpha,max_iter,random_state,n_jobs,n_samples,n_features,n_classes,n_clusters_per_class,n_informative,flip_y,scale]
trainingvalues = featureselect(training_arr, featurevalues)
testingvalues = featureselect(testing_arr, featurevalues)

a=trainingvalues+testingvalues


# use sklearn to do the prediction
clf = ExtraTreesRegressor(n_estimators=5000, criterion='mse')  # score:about 22
#clf = RandomForestRegressor()
#clf = LinearRegression()
#clf = neighbors.KNeighborsRegressor()
clf = clf.fit(trainingvalues, arr_y)
res_arr = list(clf.predict(testingvalues))  # res_arr is an array of int

#res_arr = list(clf.predict(trainingvalues))
#mse = mean_squared_error(arr_y, res_arr)
#print (mse)

# print the result
print(res_arr)
# print(training_label)
predictionFile = open('prediction.csv', 'w')
predictionFile.write('Id,time\n')
for i in range(0, len(res_arr)):
    predictionFile.write(str(i) + ',' + str(res_arr[i]) + '\n')
predictionFile.close()
