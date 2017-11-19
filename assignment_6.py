import json
import numpy as np
import math
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error, mean_squared_error

traindata = 'training.json'
testdata = 'testing.json'

def getData():
    global training_data
    global testing_data
    with open(traindata) as json_data_1:
        training_data = json.load(json_data_1, encoding = 'ISO-8859-1')
    with open(testdata) as json_data_2:
        testing_data = json.load(json_data_2, encoding = 'ISO-8859-1')

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
        if not i.has_key('karma') or not i.has_key('created') or not i.has_key('submitted'):
            i["karma"] = 0;
            i["created"] = 1509813038
            i["submitted"] = 0
        train_karma.append(i["karma"])
        train_created.append(i["created"])
        train_submitted.append(i["submitted"])
        
    for j in testing_data:
        if not j.has_key('karma') or not j.has_key('created') or not i.has_key('submitted'):
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
    
    print('Train MSE: ', train_MSE)
    print('Test MSE: ', test_MSE)
    

def run():
    getData()
    formatData()
    trainModel()
    calcMAE()
    calcRMSE()
    print 'done'

run()
