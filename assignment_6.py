import json
import numpy as np

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
    global TRAIN_CREATED
    global TEST_CREATED
    
    train_karma = []
    train_created = []
    train_submitted = []
    test_karma = []
    test_created = []
    test_submitted = []
    TRAIN_CREATED = []
    TEST_CREATED = []
    
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

    TRAIN_CREATED = np.array([train_created])
    TRAIN_CREATED = TRAIN_CREATED.T
    TEST_CREATED = np.array([test_created])
    TEST_CREATED = TEST_CREATED.T   

def run():
    getData()
    formatData()
    print 'done'

run()
