import json
import pandas as pd
import numpy as np
import math
from sklearn import linear_model, metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_score, KFold, cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

traindata = 'training.json'
testdata = 'testing.json'
fulldata = 'users.json'
breastdata = 'BreastCancer.csv'

def getData():
    global training_data
    global testing_data
    global full_data
    with open(traindata) as json_data_1:
        training_data = json.load(json_data_1, encoding = 'ISO-8859-1')
    with open(testdata) as json_data_2:
        testing_data = json.load(json_data_2, encoding = 'ISO-8859-1')
    with open(fulldata) as json_data_3:
        full_data = json.load(json_data_3, encoding = 'ISO-8859-1')

def formatData():
    global train_karma
    global train_submitted
    global train_created
    global test_submitted
    global test_karma
    global test_created
    global TRAIN_X
    global TEST_X

    train_karma = []
    train_created = []
    train_submitted = []
    test_karma = []
    test_created = []
    test_submitted = []
    TRAIN_X = []
    TEST_X = []

    for i in training_data:
        if not 'karma' in i or not 'created' in i or not 'submitted' in i:
            i["karma"] = 0;
            i["created"] = 1509813038
            i["submitted"] = 0
        train_karma.append(i["karma"])
        train_created.append(i["created"])
        train_submitted.append(i["submitted"])

    for j in testing_data:
        if not 'karma' in j or not 'created' in j or not 'submitted' in i:
            j["karma"] = 0;
            j["created"] = 1509813038
            j["submitted"] = 0
        test_karma.append(j["karma"])
        test_created.append(j["created"])
        test_submitted.append(i["submitted"])

    TRAIN_X = np.array([train_created, train_submitted])
    TRAIN_X = TRAIN_X.T
    TEST_X = np.array([test_created, test_submitted])
    TEST_X = TEST_X.T

def trainModel():
    global model

    x, y = TRAIN_X, train_karma

    model = linear_model.LinearRegression()
    model.fit(x, y)

def calcMAE():
    train_karma_pred = model.predict(TRAIN_X)
    test_karma_pred = model.predict(TEST_X)

    train_MAE = mean_absolute_error(train_karma, train_karma_pred)
    test_MAE = mean_absolute_error(test_karma, test_karma_pred)

    print('Train MAE: ', train_MAE)
    print('Test MAE: ', test_MAE)

def calcRMSE():
    train_karma_pred = model.predict(TRAIN_X)
    test_karma_pred = model.predict(TEST_X)

    train_MSE = mean_squared_error(train_karma, train_karma_pred)
    test_MSE = mean_squared_error(test_karma, test_karma_pred)
    train_MSE = math.sqrt(train_MSE)
    test_MSE = math.sqrt(test_MSE)

    print('Train RMSE: ', train_MSE)
    print('Test RMSE: ', test_MSE)

def kfold():
    X = []
    y = []
    x_temp_1 = []
    x_temp_2 = []
    total_score = 0
    total_train_MAE = 0
    total_test_MAE = 0
    total_train_RMSE = 0
    total_test_RMSE = 0

    for i in training_data:
        if not 'karma' in i or not 'created' in i or not 'submitted' in i:
            i["karma"] = 0;
            i["created"] = 1509813038
            i["submitted"] = 0
        y.append(i["karma"])
        x_temp_1.append(i["created"])
        x_temp_2.append(i["submitted"])
    X = np.array([x_temp_1, x_temp_2])
    X = X.T
    y = np.array([y])
    y = y.T

    i = 0

    folds = KFold(n_splits = 10)
    for train_indices, test_indices in folds.split(X, y):
        i = i+1
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]

        pl = PolynomialFeatures(degree=10, include_bias=False)
        lm = LinearRegression()

        pipeline = Pipeline([("pl", pl), ("lm", lm)])
        pipeline.fit(X_train, y_train)

        y_train_pred = pipeline.predict(X_train)
        y_test_pred = pipeline.predict(X_test)

        train_RMSE = mean_squared_error(y_train, y_train_pred)
        test_RMSE = mean_squared_error(y_test, y_test_pred)
        train_RMSE = math.sqrt(train_RMSE)
        test_RMSE = math.sqrt(test_RMSE)

        train_MAE = mean_absolute_error(y_train, y_train_pred)
        test_MAE = mean_absolute_error(y_test, y_test_pred)

        total_train_MAE = total_train_MAE + train_MAE
        total_test_MAE = total_test_MAE + test_MAE
        total_train_RMSE = total_train_RMSE + train_RMSE
        total_test_RMSE = total_test_RMSE + test_RMSE
        total_score = total_score + pipeline.score(X_test, y_test)

        #print("Score: " + str(pipeline.score(X_test, y_test)))
        #print '10-fold Train MAE: ' , train_MAE , '(' , i , ')'
        #print '10-fold Test MAE: ' , test_MAE , '(' , i , ')'
        #print '10-fold Train RMSE: ' , train_RMSE , '(' , i , ')'
        #print '10-fold Test RMSE: ' , test_RMSE , '(' , i , ')'

    print('10-fold avg Score:', total_score / 10)
    print('10-fold avg Train MAE:', total_train_MAE / 10)
    print('10-fold avg Test MAE:', total_test_MAE / 10)
    print('10-fold avg Train RMSE:', total_train_RMSE / 10)
    print('10-fold avg Test RMSE:', total_test_RMSE / 10)

def importBreastData():
    global df
    df = pd.read_csv('BreastCancer.csv')
    print(df.head(1))
    print('number of rows:', df.shape[0]) #gives number of row count
    print('number of columns:', df.shape[1]) #gives number of col count
    names = ['ID','Diagnosis','Mean radius','Mean texture','Mean perimeter','Mean area','Mean smoothness','Mean compactness','Mean concavity','Mean concave points','Mean symmetry','Mean fractal dimension','radius SE','texture SE','perimeter SE','area SE','smoothness SE','compactness SE','concavity SE','concave points SE','symmetry SE','fractal dimension SE','Worst radius','Worst texture','Worst perimeter','Worst area','Worst smoothness','Worst compactness','Worst concavity','Worst concave points','Worst symmetry','Worst fractal dimension']
    print('our file contains values about:', names)

    text = "dataen består af id diagnosis og 10 parametre som beskriver canceren, disse 10 parametre har alle en mean værdi, en standard error og en worst"
    print('description of data:', text)

def predictCancer():
    logreg = linear_model.LogisticRegression()

    variables = ['Mean radius','Mean texture','Mean perimeter','Mean area','Mean smoothness','Mean compactness','Mean concavity','Mean concave points','Mean symmetry','Mean fractal dimension','radius SE','texture SE','perimeter SE','area SE','smoothness SE','compactness SE','concavity SE','concave points SE','symmetry SE','fractal dimension SE','Worst radius','Worst texture','Worst perimeter','Worst area','Worst smoothness','Worst compactness','Worst concavity','Worst concave points','Worst symmetry','Worst fractal dimension']
    descriptors = [variables]

    X = df[variables].values.reshape(-1,len(variables))
    Y = df['Diagnosis']

    folds = KFold(n_splits=10)

    accuracies = []

    print()
    print("accuracies:")

    for train_indices, test_indices in folds.split(X, Y):
        X_train, X_test = X[train_indices], X[test_indices]
        Y_train, Y_test = Y[train_indices], Y[test_indices]

        logreg.fit(X_train, Y_train)

        pred = cross_val_predict(logreg,X_test,Y_test, cv=10)

        accuracy = metrics.accuracy_score(Y_test, pred, normalize=True)

        accuracies.append(accuracy)

        print(accuracy)


    npAccuracy = np.array(accuracies).astype(np.float)

    print("Average accuracy:", str(np.mean(npAccuracy)))

def run():
    getData()
    formatData()
    trainModel()
    calcMAE()
    calcRMSE()
    print('Score: ', model.score(TEST_X, test_karma))
    kfold()
    importBreastData()
    predictCancer()
    print('done')

run()
