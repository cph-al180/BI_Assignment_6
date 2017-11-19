import json

traindata = 'training.json'
testdata = 'testing.json'

def getData():
    global training_data
    global testing_data
    with open(traindata) as json_data_1:
        training_data = json.load(json_data_1, encoding = 'ISO-8859-1')
    with open(testdata) as json_data_2:
        testing_data = json.load(json_data_2, encoding = 'ISO-8859-1')

def run():
    getData()
    print 'done'

run()
