# Business Intelligence Assignment 6 - Multivariate linear regression and logistic regression  

## Part 1   - HackerNews MAE & RMSE

Theoretically speaking, if the new metric has any correlation with the amount of karma a given user has, then the new model should have better results. If the results are the same or worse, then the new metric adds nothing to the model.  
As we can see from the results, the MAE is quite a bit lower. The RMSE is still fairly high, but it makes sense that the result with testing data is higher than training, which counts for both MAE and RMSE (using new data). Overall the new results are a big improvement.  
The difference in score between the 2 models, is however pretty interesting. The score of the new model is worse than the old one? If we again look at the training results from the old and new model, we can see that the new model is vastly better. However, if we only compare the testing results, then the new model isn't that much better, and possbily even worse. This highlights the issue with the using the 80/20 split, as the testing data from the new model clearly deviates far from the model, which results in a worse score.

Without # of posts (old):  
`MAE (Training): 4535.2278195244253`  
`MAE (Testing): 4363.9936837520208`  
`RMSE (Training): 10230.170612147558`  
`RMSE (Testing): 7858.119559984959`  
`Score: 0.121689402714`  

With # of posts (new):  
`MAE (Training): 1685.15838569`  
`MAE (Testing): 3621.08926426`  
`RMSE (Training): 4420.63815212`  
`RMSE (Testing): 10015.2422784`   
`Score: -0.0741407688896`

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

## Part 2 - K-Fold Cross Validation

## Part 3 - Logistic Model
