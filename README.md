# Business Intelligence Assignment 6 - Multivariate linear regression and logistic regression  

## Part 1   - HackerNews MAE & RMSE

Training:  
```python
def trainModel():
    global model
    x, y = TRAIN_X, train_karma
    model = linear_model.LinearRegression()
    model.fit(x, y)  
```
  
MAE:  
```python
def calcMAE():
    train_karma_pred = model.predict(TRAIN_X)
    test_karma_pred = model.predict(TEST_X) 
    train_MAE = mean_absolute_error(train_karma, train_karma_pred)
    test_MAE = mean_absolute_error(test_karma, test_karma_pred)
```  
  
RMSE:  
```python
def calcMAE():
    train_karma_pred = model.predict(TRAIN_X)
    test_karma_pred = model.predict(TEST_X)
    train_MSE = mean_squared_error(train_karma, train_karma_pred)
    test_MSE = mean_squared_error(test_karma, test_karma_pred)
    train_MSE = math.sqrt(train_MSE)
    test_MSE = math.sqrt(test_MSE)
```  
Without # of posts (old):  
`a`  
`a`  
`a`  
`a`  

With # of posts (new):  
`a`  
`a`  
`a`  
`a`  

## Part 2 - K-Fold Cross Validation

## Part 3 - Logistic Model
