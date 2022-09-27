# Practice Lab: Advice for Applying Machine Learning
In this lab, you will explore techniques to evaluate and improve your machine learning models.

# Outline
- [ 1 - Packages ](#1)
- [ 2 - Evaluating a Learning Algorithm (Polynomial Regression)](#2)
  - [ 2.1 Splitting your data set](#2.1)
  - [ 2.2 Error calculation for model evaluation, linear regression](#2.2)
    - [ Exercise 1](#ex01)
  - [ 2.3 Compare performance on training and test data](#2.3)
- [ 3 - Bias and Variance<img align="Right" src="./images/C2_W3_BiasVarianceDegree.png"  style=" width:500px; padding: 10px 20px ; "> ](#3)
  - [ 3.1 Plot Train, Cross-Validation, Test](#3.1)
  - [ 3.2 Finding the optimal degree](#3.2)
  - [ 3.3 Tuning Regularization.](#3.3)
  - [ 3.4 Getting more data: Increasing Training Set Size (m)](#3.4)
- [ 4 - Evaluating a Learning Algorithm (Neural Network)](#4)
  - [ 4.1 Data Set](#4.1)
  - [ 4.2 Evaluating categorical model by calculating classification error](#4.2)
    - [ Exercise 2](#ex02)
- [ 5 - Model Complexity](#5)
  - [ Exercise 3](#ex03)
  - [ 5.1 Simple model](#5.1)
    - [ Exercise 4](#ex04)
- [ 6 - Regularization](#6)
  - [ Exercise 5](#ex05)
- [ 7 - Iterate to find optimal regularization value](#7)
  - [ 7.1 Test](#7.1)


<a name="1"></a>
## 1 - Packages 

First, let's run the cell below to import all the packages that you will need during this assignment.
- [numpy](https://numpy.org/) is the fundamental package for scientific computing Python.
- [matplotlib](http://matplotlib.org) is a popular library to plot graphs in Python.
- [scikitlearn](https://scikit-learn.org/stable/) is a basic library for data mining
- [tensorflow](https://www.tensorflow.org/) a popular platform for machine learning.


```python
import numpy as np
%matplotlib widget
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import relu,linear
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

from public_tests_a1 import * 

tf.keras.backend.set_floatx('float64')
from assigment_utils import *

tf.autograph.set_verbosity(0)
```

<a name="2"></a>
## 2 - Evaluating a Learning Algorithm (Polynomial Regression)

<img align="Right" src="./images/C2_W3_TrainingVsNew.png"  style=" width:350px; padding: 10px 20px ; "> Let's say you have created a machine learning model and you find it *fits* your training data very well. You're done? Not quite. The goal of creating the model was to be able to predict values for <span style="color:blue">*new* </span> examples. 

How can you test your model's performance on new data before deploying it?   
The answer has two parts:
* Split your original data set into "Training" and "Test" sets. 
    * Use the training data to fit the parameters of the model
    * Use the test data to evaluate the model on *new* data
* Develop an error function to evaluate your model.

<a name="2.1"></a>
### 2.1 Splitting your data set
Lectures advised reserving 20-40% of your data set for testing. Let's use an `sklearn` function [train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) to perform the split. Double-check the shapes after running the following cell.


```python
# Generate some data
X,y,x_ideal,y_ideal = gen_data(18, 2, 0.7)
print("X.shape", X.shape, "y.shape", y.shape)

#split the data using sklearn routine 
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33, random_state=1)
print("X_train.shape", X_train.shape, "y_train.shape", y_train.shape)
print("X_test.shape", X_test.shape, "y_test.shape", y_test.shape)
```

    X.shape (18,) y.shape (18,)
    X_train.shape (12,) y_train.shape (12,)
    X_test.shape (6,) y_test.shape (6,)


#### 2.1.1 Plot Train, Test sets
You can see below the data points that will be part of training (in red) are intermixed with those that the model is not trained on (test). This particular data set is a quadratic function with noise added. The "ideal" curve is shown for reference.


```python
fig, ax = plt.subplots(1,1,figsize=(4,4))
ax.plot(x_ideal, y_ideal, "--", color = "orangered", label="y_ideal", lw=1)
ax.set_title("Training, Test",fontsize = 14)
ax.set_xlabel("x")
ax.set_ylabel("y")

ax.scatter(X_train, y_train, color = "red",           label="train")
ax.scatter(X_test, y_test,   color = dlc["dlblue"],   label="test")
ax.legend(loc='upper left')
plt.show()
```


    Canvas(toolbar=Toolbar(toolitems=[('Home', 'Reset original view', 'home', 'home'), ('Back', 'Back to previous …


<a name="2.2"></a>
### 2.2 Error calculation for model evaluation, linear regression
When *evaluating* a linear regression model, you average the squared error difference of the predicted values and the target values.

$$ J_\text{test}(\mathbf{w},b) = 
            \frac{1}{2m_\text{test}}\sum_{i=0}^{m_\text{test}-1} ( f_{\mathbf{w},b}(\mathbf{x}^{(i)}_\text{test}) - y^{(i)}_\text{test} )^2 
            \tag{1}
$$

<a name="ex01"></a>
### Exercise 1

Below, create a function to evaluate the error on a data set for a linear regression model.


```python
# UNQ_C1
# GRADED CELL: eval_mse
def eval_mse(y, yhat):
    """ 
    Calculate the mean squared error on a data set.
    Args:
      y    : (ndarray  Shape (m,) or (m,1))  target value of each example
      yhat : (ndarray  Shape (m,) or (m,1))  predicted value of each example
    Returns:
      err: (scalar)             
    """
    m = len(y)
    err = 0.0
    for i in range(m):
    ### START CODE HERE ### 
        err+=(y[i]-yhat[i])**2
    ### END CODE HERE ### 
    err/= 2*m
    return(err)
```


```python
y_hat = np.array([2.4, 4.2])
y_tmp = np.array([2.3, 4.1])
eval_mse(y_hat, y_tmp)

# BEGIN UNIT TEST
test_eval_mse(eval_mse)   
# END UNIT TEST
```

    [92m All tests passed.


<details>
  <summary><font size="3" color="darkgreen"><b>Click for hints</b></font></summary>

    
```python
def eval_mse(y, yhat):
    """ 
    Calculate the mean squared error on a data set.
    Args:
      y    : (ndarray  Shape (m,) or (m,1))  target value of each example
      yhat : (ndarray  Shape (m,) or (m,1))  predicted value of each example
    Returns:
      err: (scalar)             
    """
    m = len(y)
    err = 0.0
    for i in range(m):
        err_i  = ( (yhat[i] - y[i])**2 ) 
        err   += err_i                                                                
    err = err / (2*m)                    
    return(err)
``` 

<a name="2.3"></a>
### 2.3 Compare performance on training and test data
Let's build a high degree polynomial model to minimize training error. This will use the linear_regression functions from `sklearn`. The code is in the imported utility file if you would like to see the details. The steps below are:
* create and fit the model. ('fit' is another name for training or running gradient descent).
* compute the error on the training data.
* compute the error on the test data.


```python
# create a model in sklearn, train on training data
degree = 10
lmodel = lin_model(degree)
lmodel.fit(X_train, y_train)

# predict on training data, find training error
yhat = lmodel.predict(X_train)
err_train = lmodel.mse(y_train, yhat)

# predict on test data, find error
yhat = lmodel.predict(X_test)
err_test = lmodel.mse(y_test, yhat)
```

The computed error on the training set is substantially less than that of the test set. 


```python
print(f"training err {err_train:0.2f}, test err {err_test:0.2f}")
```

    training err 58.01, test err 171215.01


The following plot shows why this is. The model fits the training data very well. To do so, it has created a complex function. The test data was not part of the training and the model does a poor job of predicting on this data.  
This model would be described as 1) is overfitting, 2) has high variance 3) 'generalizes' poorly.


```python
# plot predictions over data range 
x = np.linspace(0,int(X.max()),100)  # predict values for plot
y_pred = lmodel.predict(x).reshape(-1,1)

plt_train_test(X_train, y_train, X_test, y_test, x, y_pred, x_ideal, y_ideal, degree)
```


    Canvas(toolbar=Toolbar(toolitems=[('Home', 'Reset original view', 'home', 'home'), ('Back', 'Back to previous …


The test set error shows this model will not work well on new data. If you use the test error to guide improvements in the model, then the model will perform well on the test data... but the test data was meant to represent *new* data.
You need yet another set of data to test new data performance.

The proposal made during lecture is to separate data into three groups. The distribution of training, cross-validation and test sets shown in the below table is a typical distribution, but can be varied depending on the amount of data available.

| data             | % of total | Description |
|------------------|:----------:|:---------|
| training         | 60         | Data used to tune model parameters $w$ and $b$ in training or fitting |
| cross-validation | 20         | Data used to tune other model parameters like degree of polynomial, regularization or the architecture of a neural network.|
| test             | 20         | Data used to test the model after tuning to gauge performance on new data |


Let's generate three data sets below. We'll once again use `train_test_split` from `sklearn` but will call it twice to get three splits:


```python
# Generate  data
X,y, x_ideal,y_ideal = gen_data(40, 5, 0.7)
print("X.shape", X.shape, "y.shape", y.shape)

#split the data using sklearn routine 
X_train, X_, y_train, y_ = train_test_split(X,y,test_size=0.40, random_state=1)
X_cv, X_test, y_cv, y_test = train_test_split(X_,y_,test_size=0.50, random_state=1)
print("X_train.shape", X_train.shape, "y_train.shape", y_train.shape)
print("X_cv.shape", X_cv.shape, "y_cv.shape", y_cv.shape)
print("X_test.shape", X_test.shape, "y_test.shape", y_test.shape)
```

    X.shape (40,) y.shape (40,)
    X_train.shape (24,) y_train.shape (24,)
    X_cv.shape (8,) y_cv.shape (8,)
    X_test.shape (8,) y_test.shape (8,)


<a name="3"></a>
## 3 - Bias and Variance<img align="Right" src="./images/C2_W3_BiasVarianceDegree.png"  style=" width:500px; padding: 10px 20px ; "> 
 Above, it was clear the degree of the polynomial model was too high. How can you choose a good value? It turns out, as shown in the diagram, the training and cross-validation performance can provide guidance. By trying a range of degree values, the training and cross-validation performance can be evaluated. As the degree becomes too large, the cross-validation performance will start to degrade relative to the training performance. Let's try this on our example.

<a name="3.1"></a>
### 3.1 Plot Train, Cross-Validation, Test
You can see below the datapoints that will be part of training (in red) are intermixed with those that the model is not trained on (test and cv).


```python
fig, ax = plt.subplots(1,1,figsize=(4,4))
ax.plot(x_ideal, y_ideal, "--", color = "orangered", label="y_ideal", lw=1)
ax.set_title("Training, CV, Test",fontsize = 14)
ax.set_xlabel("x")
ax.set_ylabel("y")

ax.scatter(X_train, y_train, color = "red",           label="train")
ax.scatter(X_cv, y_cv,       color = dlc["dlorange"], label="cv")
ax.scatter(X_test, y_test,   color = dlc["dlblue"],   label="test")
ax.legend(loc='upper left')
plt.show()
```


    Canvas(toolbar=Toolbar(toolitems=[('Home', 'Reset original view', 'home', 'home'), ('Back', 'Back to previous …


<a name="3.2"></a>
### 3.2 Finding the optimal degree
In previous labs, you found that you could create a model capable of fitting complex curves by utilizing a polynomial (See Course1, Week2 Feature Engineering and Polynomial Regression Lab).  Further, you demonstrated that by increasing the *degree* of the polynomial, you could *create* overfitting. (See Course 1, Week3, Over-Fitting Lab). Let's use that knowledge here to test our ability to tell the difference between over-fitting and under-fitting.

Let's train the model repeatedly, increasing the degree of the polynomial each iteration. Here, we're going to use the [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html#sklearn.linear_model.LinearRegression) linear regression model for speed and simplicity.


```python
max_degree = 9
err_train = np.zeros(max_degree)    
err_cv = np.zeros(max_degree)      
x = np.linspace(0,int(X.max()),100)  
y_pred = np.zeros((100,max_degree))  #columns are lines to plot

for degree in range(max_degree):
    lmodel = lin_model(degree+1)
    lmodel.fit(X_train, y_train)
    yhat = lmodel.predict(X_train)
    err_train[degree] = lmodel.mse(y_train, yhat)
    yhat = lmodel.predict(X_cv)
    err_cv[degree] = lmodel.mse(y_cv, yhat)
    y_pred[:,degree] = lmodel.predict(x)
    
optimal_degree = np.argmin(err_cv)+1
```

<font size="4">Let's plot the result:</font>


```python
plt.close("all")
plt_optimal_degree(X_train, y_train, X_cv, y_cv, x, y_pred, x_ideal, y_ideal, 
                   err_train, err_cv, optimal_degree, max_degree)
```


    Canvas(toolbar=Toolbar(toolitems=[('Home', 'Reset original view', 'home', 'home'), ('Back', 'Back to previous …


The plot above demonstrates that separating data into two groups, data the model is trained on and data the model has not been trained on, can be used to determine if the model is underfitting or overfitting. In our example, we created a variety of models varying from underfitting to overfitting by increasing the degree of the polynomial used. 
- On the left plot, the solid lines represent the predictions from these models. A polynomial model with degree 1 produces a straight line that intersects very few data points, while the maximum degree hews very closely to every data point. 
- on the right:
    - the error on the trained data (blue) decreases as the model complexity increases as expected
    - the error of the cross-validation data decreases initially as the model starts to conform to the data, but then increases as the model starts to over-fit on the training data (fails to *generalize*).     
    
It's worth noting that the curves in these examples as not as smooth as one might draw for a lecture. It's clear the specific data points assigned to each group can change your results significantly. The general trend is what is important.

<a name="3.3"></a>
### 3.3 Tuning Regularization.
In previous labs, you have utilized *regularization* to reduce overfitting. Similar to degree, one can use the same methodology to tune the regularization parameter lambda ($\lambda$).

Let's demonstrate this by starting with a high degree polynomial and varying the regularization parameter.


```python
lambda_range = np.array([0.0, 1e-6, 1e-5, 1e-4,1e-3,1e-2, 1e-1,1,10,100])
num_steps = len(lambda_range)
degree = 10
err_train = np.zeros(num_steps)    
err_cv = np.zeros(num_steps)       
x = np.linspace(0,int(X.max()),100) 
y_pred = np.zeros((100,num_steps))  #columns are lines to plot

for i in range(num_steps):
    lambda_= lambda_range[i]
    lmodel = lin_model(degree, regularization=True, lambda_=lambda_)
    lmodel.fit(X_train, y_train)
    yhat = lmodel.predict(X_train)
    err_train[i] = lmodel.mse(y_train, yhat)
    yhat = lmodel.predict(X_cv)
    err_cv[i] = lmodel.mse(y_cv, yhat)
    y_pred[:,i] = lmodel.predict(x)
    
optimal_reg_idx = np.argmin(err_cv) 
```


```python
plt.close("all")
plt_tune_regularization(X_train, y_train, X_cv, y_cv, x, y_pred, err_train, err_cv, optimal_reg_idx, lambda_range)
```


    Canvas(toolbar=Toolbar(toolitems=[('Home', 'Reset original view', 'home', 'home'), ('Back', 'Back to previous …


Above, the plots show that as regularization increases, the model moves from a high variance (overfitting) model to a high bias (underfitting) model. The vertical line in the right plot shows the optimal value of lambda. In this example, the polynomial degree was set to 10. 

<a name="3.4"></a>
### 3.4 Getting more data: Increasing Training Set Size (m)
When a model is overfitting (high variance), collecting additional data can improve performance. Let's try that here.


```python
X_train, y_train, X_cv, y_cv, x, y_pred, err_train, err_cv, m_range,degree = tune_m()
plt_tune_m(X_train, y_train, X_cv, y_cv, x, y_pred, err_train, err_cv, m_range, degree)
```


    Canvas(toolbar=Toolbar(toolitems=[('Home', 'Reset original view', 'home', 'home'), ('Back', 'Back to previous …


The above plots show that when a model has high variance and is overfitting, adding more examples improves performance. Note the curves on the left plot. The final curve with the highest value of $m$ is a smooth curve that is in the center of the data. On the right, as the number of examples increases, the performance of the training set and cross-validation set converge to similar values. Note that the curves are not as smooth as one might see in a lecture. That is to be expected. The trend remains clear: more data improves generalization. 

> Note that adding more examples when the model has high bias (underfitting) does not improve performance.


<a name="4"></a>
## 4 - Evaluating a Learning Algorithm (Neural Network)
Above, you tuned aspects of a polynomial regression model. Here, you will work with a neural network model. Let's start by creating a classification data set. 

<a name="4.1"></a>
### 4.1 Data Set
Run the cell below to generate a data set and split it into training, cross-validation (CV) and test sets. In this example, we're increasing the percentage of cross-validation data points for emphasis.  


```python
# Generate and split data set
X, y, centers, classes, std = gen_blobs()

# split the data. Large CV population for demonstration
X_train, X_, y_train, y_ = train_test_split(X,y,test_size=0.50, random_state=1)
X_cv, X_test, y_cv, y_test = train_test_split(X_,y_,test_size=0.20, random_state=1)
print("X_train.shape:", X_train.shape, "X_cv.shape:", X_cv.shape, "X_test.shape:", X_test.shape)
```

    X_train.shape: (400, 2) X_cv.shape: (320, 2) X_test.shape: (80, 2)



```python
plt_train_eq_dist(X_train, y_train,classes, X_cv, y_cv, centers, std)
```


    Canvas(toolbar=Toolbar(toolitems=[('Home', 'Reset original view', 'home', 'home'), ('Back', 'Back to previous …


Above, you can see the data on the left. There are six clusters identified by color. Both training points (dots) and cross-validataion points (triangles) are shown. The interesting points are those that fall in ambiguous locations where either cluster might consider them members. What would you expect a neural network model to do? What would be an example of overfitting? underfitting?  
On the right is an example of an 'ideal' model, or a model one might create knowing the source of the data. The lines represent 'equal distance' boundaries where the distance between center points is equal. It's worth noting that this model would "misclassify" roughly 8% of the total data set.

<a name="4.2"></a>
### 4.2 Evaluating categorical model by calculating classification error
The evaluation function for categorical models used here is simply the fraction of incorrect predictions:  
$$ J_{cv} =\frac{1}{m}\sum_{i=0}^{m-1} 
\begin{cases}
    1, & \text{if $\hat{y}^{(i)} \neq y^{(i)}$}\\
    0, & \text{otherwise}
\end{cases}
$$

<a name="ex02"></a>
### Exercise 2

Below, complete the routine to calculate classification error. Note, in this lab, target values are the index of the category and are not [one-hot encoded](https://en.wikipedia.org/wiki/One-hot).


```python
# UNQ_C2
# GRADED CELL: eval_cat_err
def eval_cat_err(y, yhat):
    """ 
    Calculate the categorization error
    Args:
      y    : (ndarray  Shape (m,) or (m,1))  target value of each example
      yhat : (ndarray  Shape (m,) or (m,1))  predicted value of each example
    Returns:|
      cerr: (scalar)             
    """
    m = len(y)
    incorrect = 0
    for i in range(m):
    ### START CODE HERE ### 
        if(yhat[i]!=y[i]):
            incorrect+=1
    ### END CODE HERE ### 
    cerr=incorrect/m
    return(cerr)
```


```python
y_hat = np.array([1, 2, 0])
y_tmp = np.array([1, 2, 3])
print(f"categorization error {np.squeeze(eval_cat_err(y_hat, y_tmp)):0.3f}, expected:0.333" )
y_hat = np.array([[1], [2], [0], [3]])
y_tmp = np.array([[1], [2], [1], [3]])
print(f"categorization error {np.squeeze(eval_cat_err(y_hat, y_tmp)):0.3f}, expected:0.250" )

# BEGIN UNIT TEST  
test_eval_cat_err(eval_cat_err)
# END UNIT TEST
# BEGIN UNIT TEST  
test_eval_cat_err(eval_cat_err)
# END UNIT TEST
```

    categorization error 0.333, expected:0.333
    categorization error 0.250, expected:0.250
    [92m All tests passed.
    [92m All tests passed.


<details>
  <summary><font size="3" color="darkgreen"><b>Click for hints</b></font></summary>
    
```python
def eval_cat_err(y, yhat):
    """ 
    Calculate the categorization error
    Args:
      y    : (ndarray  Shape (m,) or (m,1))  target value of each example
      yhat : (ndarray  Shape (m,) or (m,1))  predicted value of each example
    Returns:|
      cerr: (scalar)             
    """
    m = len(y)
    incorrect = 0
    for i in range(m):
        if yhat[i] != y[i]:    # @REPLACE
            incorrect += 1     # @REPLACE
    cerr = incorrect/m         # @REPLACE
    return(cerr)                                    
``` 

<a name="5"></a>
## 5 - Model Complexity
Below, you will build two models. A complex model and a simple model. You will evaluate the models to determine if they are likely to overfit or underfit.

###  5.1 Complex model

<a name="ex03"></a>
### Exercise 3
Below, compose a three-layer model:
* Dense layer with 120 units, relu activation
* Dense layer with 40 units, relu activation
* Dense layer with 6 units and a linear activation (not softmax)  
Compile using
* loss with `SparseCategoricalCrossentropy`, remember to use  `from_logits=True`
* Adam optimizer with learning rate of 0.01.


```python
# UNQ_C3
# GRADED CELL: model
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

tf.random.set_seed(1234)
model = Sequential(
    [
        ### START CODE HERE ### 
        tf.keras.layers.Dense(120, activation="relu"),
        tf.keras.layers.Dense(40, activation="relu"),
        tf.keras.layers.Dense(6, activation="linear")
        ### END CODE HERE ### 
    ], name="Complex"
)
model.compile(
    ### START CODE HERE ### 
    loss=SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(lr=0.01),
    ### END CODE HERE ### 
)
```


```python
# BEGIN UNIT TEST
model.fit(
    X_train, y_train,
    epochs=1000
)
# END UNIT TEST
```

    Epoch 1/1000
    13/13 [==============================] - 0s 1ms/step - loss: 1.1106
    Epoch 2/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4281
    Epoch 3/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3345
    Epoch 4/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2896
    Epoch 5/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2867
    Epoch 6/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2918
    Epoch 7/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2497
    Epoch 8/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2298
    Epoch 9/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2307
    Epoch 10/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2071
    Epoch 11/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.2115
    Epoch 12/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2070
    Epoch 13/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2366
    Epoch 14/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2261
    Epoch 15/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.2224
    Epoch 16/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2055
    Epoch 17/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2044
    Epoch 18/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2006
    Epoch 19/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2168
    Epoch 20/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.2047
    Epoch 21/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2237
    Epoch 22/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2497
    Epoch 23/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2113
    Epoch 24/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.2025
    Epoch 25/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2107
    Epoch 26/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2000
    Epoch 27/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1935
    Epoch 28/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1963
    Epoch 29/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.2188
    Epoch 30/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2424
    Epoch 31/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1969
    Epoch 32/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1950
    Epoch 33/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1904
    Epoch 34/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.2173
    Epoch 35/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2074
    Epoch 36/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1768
    Epoch 37/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1794
    Epoch 38/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1733
    Epoch 39/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1955
    Epoch 40/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1870
    Epoch 41/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2128
    Epoch 42/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1987
    Epoch 43/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1895
    Epoch 44/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2073
    Epoch 45/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2148
    Epoch 46/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1774
    Epoch 47/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1886
    Epoch 48/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1763
    Epoch 49/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1769
    Epoch 50/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1763
    Epoch 51/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2020
    Epoch 52/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1889
    Epoch 53/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2035
    Epoch 54/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1761
    Epoch 55/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1838
    Epoch 56/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1774
    Epoch 57/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1953
    Epoch 58/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1882
    Epoch 59/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1860
    Epoch 60/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1919
    Epoch 61/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1848
    Epoch 62/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1630
    Epoch 63/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1616
    Epoch 64/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2008
    Epoch 65/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1936
    Epoch 66/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1824
    Epoch 67/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2092
    Epoch 68/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2287
    Epoch 69/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1877
    Epoch 70/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1716
    Epoch 71/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1917
    Epoch 72/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1703
    Epoch 73/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1750
    Epoch 74/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1836
    Epoch 75/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1696
    Epoch 76/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1542
    Epoch 77/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1715
    Epoch 78/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1545
    Epoch 79/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1593
    Epoch 80/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1844
    Epoch 81/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1881
    Epoch 82/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1696
    Epoch 83/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1614
    Epoch 84/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1762
    Epoch 85/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1779
    Epoch 86/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1658
    Epoch 87/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1614
    Epoch 88/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1639
    Epoch 89/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1629
    Epoch 90/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1475
    Epoch 91/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1452
    Epoch 92/1000
    13/13 [==============================] - 0s 4ms/step - loss: 0.1473
    Epoch 93/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1490
    Epoch 94/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1650
    Epoch 95/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1706
    Epoch 96/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1704
    Epoch 97/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1764
    Epoch 98/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1855
    Epoch 99/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1685
    Epoch 100/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1569
    Epoch 101/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1645
    Epoch 102/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1737
    Epoch 103/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1935
    Epoch 104/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1600
    Epoch 105/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1483
    Epoch 106/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1555
    Epoch 107/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1678
    Epoch 108/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1435
    Epoch 109/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1419
    Epoch 110/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1494
    Epoch 111/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1538
    Epoch 112/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1682
    Epoch 113/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1687
    Epoch 114/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1436
    Epoch 115/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1366
    Epoch 116/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1485
    Epoch 117/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1400
    Epoch 118/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1357
    Epoch 119/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1444
    Epoch 120/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1403
    Epoch 121/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1465
    Epoch 122/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1549
    Epoch 123/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1402
    Epoch 124/1000
    13/13 [==============================] - 0s 2ms/step - loss: 0.1337
    Epoch 125/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1422
    Epoch 126/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1560
    Epoch 127/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1319
    Epoch 128/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1389
    Epoch 129/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1404
    Epoch 130/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1299
    Epoch 131/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1247
    Epoch 132/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1244
    Epoch 133/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1260
    Epoch 134/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1158
    Epoch 135/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1343
    Epoch 136/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1306
    Epoch 137/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1294
    Epoch 138/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1297
    Epoch 139/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1342
    Epoch 140/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1255
    Epoch 141/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1232
    Epoch 142/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1199
    Epoch 143/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1192
    Epoch 144/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1192
    Epoch 145/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1342
    Epoch 146/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1477
    Epoch 147/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1780
    Epoch 148/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1673
    Epoch 149/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1402
    Epoch 150/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1292
    Epoch 151/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1296
    Epoch 152/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1221
    Epoch 153/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1300
    Epoch 154/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1316
    Epoch 155/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1274
    Epoch 156/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1192
    Epoch 157/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1266
    Epoch 158/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1185
    Epoch 159/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1197
    Epoch 160/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1148
    Epoch 161/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1137
    Epoch 162/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1427
    Epoch 163/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1420
    Epoch 164/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1327
    Epoch 165/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1276
    Epoch 166/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1099
    Epoch 167/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1205
    Epoch 168/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1307
    Epoch 169/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1476
    Epoch 170/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1673
    Epoch 171/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1349
    Epoch 172/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1183
    Epoch 173/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1225
    Epoch 174/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1276
    Epoch 175/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1029
    Epoch 176/1000
    13/13 [==============================] - 0s 2ms/step - loss: 0.1134
    Epoch 177/1000
    13/13 [==============================] - 0s 2ms/step - loss: 0.1081
    Epoch 178/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1245
    Epoch 179/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1346
    Epoch 180/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1233
    Epoch 181/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1113
    Epoch 182/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1040
    Epoch 183/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1155
    Epoch 184/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1049
    Epoch 185/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1111
    Epoch 186/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1079
    Epoch 187/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1021
    Epoch 188/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1048
    Epoch 189/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0971
    Epoch 190/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0985
    Epoch 191/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1026
    Epoch 192/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1111
    Epoch 193/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0991
    Epoch 194/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0890
    Epoch 195/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0880
    Epoch 196/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1006
    Epoch 197/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0974
    Epoch 198/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1141
    Epoch 199/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1423
    Epoch 200/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1381
    Epoch 201/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1105
    Epoch 202/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1005
    Epoch 203/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0846
    Epoch 204/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1125
    Epoch 205/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1129
    Epoch 206/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1219
    Epoch 207/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1161
    Epoch 208/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1137
    Epoch 209/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1178
    Epoch 210/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1017
    Epoch 211/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1051
    Epoch 212/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1014
    Epoch 213/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1096
    Epoch 214/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1087
    Epoch 215/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1047
    Epoch 216/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1044
    Epoch 217/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1044
    Epoch 218/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1006
    Epoch 219/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1093
    Epoch 220/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1041
    Epoch 221/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0956
    Epoch 222/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1109
    Epoch 223/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1041
    Epoch 224/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1000
    Epoch 225/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0968
    Epoch 226/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0951
    Epoch 227/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1092
    Epoch 228/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1041
    Epoch 229/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1032
    Epoch 230/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1153
    Epoch 231/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1237
    Epoch 232/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0978
    Epoch 233/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1074
    Epoch 234/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1059
    Epoch 235/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1122
    Epoch 236/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0974
    Epoch 237/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0879
    Epoch 238/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0913
    Epoch 239/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0831
    Epoch 240/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0752
    Epoch 241/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0733
    Epoch 242/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0886
    Epoch 243/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0837
    Epoch 244/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0866
    Epoch 245/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0933
    Epoch 246/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0976
    Epoch 247/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1150
    Epoch 248/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0904
    Epoch 249/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1073
    Epoch 250/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1296
    Epoch 251/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1022
    Epoch 252/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0987
    Epoch 253/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0846
    Epoch 254/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0813
    Epoch 255/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0924
    Epoch 256/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0799
    Epoch 257/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0947
    Epoch 258/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0956
    Epoch 259/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0788
    Epoch 260/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1018
    Epoch 261/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0942
    Epoch 262/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0780
    Epoch 263/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0821
    Epoch 264/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0795
    Epoch 265/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0924
    Epoch 266/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0948
    Epoch 267/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0767
    Epoch 268/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0720
    Epoch 269/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0742
    Epoch 270/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0747
    Epoch 271/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0726
    Epoch 272/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0984
    Epoch 273/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1074
    Epoch 274/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0836
    Epoch 275/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0783
    Epoch 276/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0799
    Epoch 277/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1225
    Epoch 278/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1017
    Epoch 279/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0990
    Epoch 280/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1014
    Epoch 281/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0808
    Epoch 282/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0798
    Epoch 283/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0847
    Epoch 284/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0755
    Epoch 285/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0631
    Epoch 286/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0651
    Epoch 287/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0602
    Epoch 288/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0733
    Epoch 289/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0659
    Epoch 290/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0682
    Epoch 291/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0745
    Epoch 292/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0848
    Epoch 293/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0701
    Epoch 294/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0828
    Epoch 295/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0741
    Epoch 296/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0890
    Epoch 297/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0800
    Epoch 298/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0803
    Epoch 299/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0765
    Epoch 300/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0733
    Epoch 301/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0544
    Epoch 302/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0718
    Epoch 303/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0877
    Epoch 304/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0687
    Epoch 305/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0671
    Epoch 306/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0575
    Epoch 307/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0773
    Epoch 308/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0779
    Epoch 309/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0696
    Epoch 310/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0883
    Epoch 311/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0880
    Epoch 312/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0707
    Epoch 313/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0603
    Epoch 314/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0772
    Epoch 315/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0660
    Epoch 316/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0586
    Epoch 317/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0618
    Epoch 318/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0588
    Epoch 319/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0674
    Epoch 320/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0598
    Epoch 321/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0670
    Epoch 322/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0970
    Epoch 323/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1366
    Epoch 324/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1148
    Epoch 325/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0837
    Epoch 326/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0749
    Epoch 327/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0746
    Epoch 328/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0698
    Epoch 329/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0691
    Epoch 330/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0541
    Epoch 331/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0558
    Epoch 332/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0653
    Epoch 333/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0593
    Epoch 334/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0606
    Epoch 335/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0696
    Epoch 336/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0713
    Epoch 337/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0628
    Epoch 338/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0752
    Epoch 339/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0723
    Epoch 340/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0647
    Epoch 341/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0688
    Epoch 342/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0793
    Epoch 343/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0595
    Epoch 344/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0528
    Epoch 345/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0552
    Epoch 346/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0534
    Epoch 347/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0471
    Epoch 348/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0491
    Epoch 349/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0524
    Epoch 350/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0696
    Epoch 351/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0690
    Epoch 352/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0864
    Epoch 353/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0999
    Epoch 354/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1094
    Epoch 355/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1189
    Epoch 356/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1059
    Epoch 357/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0655
    Epoch 358/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0652
    Epoch 359/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0544
    Epoch 360/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0545
    Epoch 361/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0549
    Epoch 362/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0581
    Epoch 363/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0506
    Epoch 364/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0579
    Epoch 365/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0583
    Epoch 366/1000
    13/13 [==============================] - 0s 4ms/step - loss: 0.0607
    Epoch 367/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0428
    Epoch 368/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0495
    Epoch 369/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0721
    Epoch 370/1000
    13/13 [==============================] - 0s 4ms/step - loss: 0.0817
    Epoch 371/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0588
    Epoch 372/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0516
    Epoch 373/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0526
    Epoch 374/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0463
    Epoch 375/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0447
    Epoch 376/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0441
    Epoch 377/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0422
    Epoch 378/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0391
    Epoch 379/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0343
    Epoch 380/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0461
    Epoch 381/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0442
    Epoch 382/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0496
    Epoch 383/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0509
    Epoch 384/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0479
    Epoch 385/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0520
    Epoch 386/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0391
    Epoch 387/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0394
    Epoch 388/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0510
    Epoch 389/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0525
    Epoch 390/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0666
    Epoch 391/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0490
    Epoch 392/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0551
    Epoch 393/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0689
    Epoch 394/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0663
    Epoch 395/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0844
    Epoch 396/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0704
    Epoch 397/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0700
    Epoch 398/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0591
    Epoch 399/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0586
    Epoch 400/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0628
    Epoch 401/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1717
    Epoch 402/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1648
    Epoch 403/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1616
    Epoch 404/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1326
    Epoch 405/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1367
    Epoch 406/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1098
    Epoch 407/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1122
    Epoch 408/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1798
    Epoch 409/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1268
    Epoch 410/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1123
    Epoch 411/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0720
    Epoch 412/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0774
    Epoch 413/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0661
    Epoch 414/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0720
    Epoch 415/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0580
    Epoch 416/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0572
    Epoch 417/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0586
    Epoch 418/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0546
    Epoch 419/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0573
    Epoch 420/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0721
    Epoch 421/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0658
    Epoch 422/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0686
    Epoch 423/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0491
    Epoch 424/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0647
    Epoch 425/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0465
    Epoch 426/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0435
    Epoch 427/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0362
    Epoch 428/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0411
    Epoch 429/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0374
    Epoch 430/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0412
    Epoch 431/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0391
    Epoch 432/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0412
    Epoch 433/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0479
    Epoch 434/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0436
    Epoch 435/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0482
    Epoch 436/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0420
    Epoch 437/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0347
    Epoch 438/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0390
    Epoch 439/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0328
    Epoch 440/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0371
    Epoch 441/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0334
    Epoch 442/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0348
    Epoch 443/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0370
    Epoch 444/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0408
    Epoch 445/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0329
    Epoch 446/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0318
    Epoch 447/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0391
    Epoch 448/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0408
    Epoch 449/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0346
    Epoch 450/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0340
    Epoch 451/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0332
    Epoch 452/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0325
    Epoch 453/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0406
    Epoch 454/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0394
    Epoch 455/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0584
    Epoch 456/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0440
    Epoch 457/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0412
    Epoch 458/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0468
    Epoch 459/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0373
    Epoch 460/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0329
    Epoch 461/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0390
    Epoch 462/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0284
    Epoch 463/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0310
    Epoch 464/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0348
    Epoch 465/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0302
    Epoch 466/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0348
    Epoch 467/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0350
    Epoch 468/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0347
    Epoch 469/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0305
    Epoch 470/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0369
    Epoch 471/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0436
    Epoch 472/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0543
    Epoch 473/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0477
    Epoch 474/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0630
    Epoch 475/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1523
    Epoch 476/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3248
    Epoch 477/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1600
    Epoch 478/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1623
    Epoch 479/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1206
    Epoch 480/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0955
    Epoch 481/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1595
    Epoch 482/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1626
    Epoch 483/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1170
    Epoch 484/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1481
    Epoch 485/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0686
    Epoch 486/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0590
    Epoch 487/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0651
    Epoch 488/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0575
    Epoch 489/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0593
    Epoch 490/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0539
    Epoch 491/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0451
    Epoch 492/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0436
    Epoch 493/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0484
    Epoch 494/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0639
    Epoch 495/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0497
    Epoch 496/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0787
    Epoch 497/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0805
    Epoch 498/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0639
    Epoch 499/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0504
    Epoch 500/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0478
    Epoch 501/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0466
    Epoch 502/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0419
    Epoch 503/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0365
    Epoch 504/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0352
    Epoch 505/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0368
    Epoch 506/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0337
    Epoch 507/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0375
    Epoch 508/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0317
    Epoch 509/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0318
    Epoch 510/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0364
    Epoch 511/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0337
    Epoch 512/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0290
    Epoch 513/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0317
    Epoch 514/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0320
    Epoch 515/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0271
    Epoch 516/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0343
    Epoch 517/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0308
    Epoch 518/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0388
    Epoch 519/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0444
    Epoch 520/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0381
    Epoch 521/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0356
    Epoch 522/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0324
    Epoch 523/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0292
    Epoch 524/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0308
    Epoch 525/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0308
    Epoch 526/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0365
    Epoch 527/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0351
    Epoch 528/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0305
    Epoch 529/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0320
    Epoch 530/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0351
    Epoch 531/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0290
    Epoch 532/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0329
    Epoch 533/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0387
    Epoch 534/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0431
    Epoch 535/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0414
    Epoch 536/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0318
    Epoch 537/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0285
    Epoch 538/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0278
    Epoch 539/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0274
    Epoch 540/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0338
    Epoch 541/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0262
    Epoch 542/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0283
    Epoch 543/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0265
    Epoch 544/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0267
    Epoch 545/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0278
    Epoch 546/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0256
    Epoch 547/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0302
    Epoch 548/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0323
    Epoch 549/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0262
    Epoch 550/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0288
    Epoch 551/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0283
    Epoch 552/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0315
    Epoch 553/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0411
    Epoch 554/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0376
    Epoch 555/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0346
    Epoch 556/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0296
    Epoch 557/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0307
    Epoch 558/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0270
    Epoch 559/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0268
    Epoch 560/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0303
    Epoch 561/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0251
    Epoch 562/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0267
    Epoch 563/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0249
    Epoch 564/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0265
    Epoch 565/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0297
    Epoch 566/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0338
    Epoch 567/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0432
    Epoch 568/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0483
    Epoch 569/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1205
    Epoch 570/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1063
    Epoch 571/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1035
    Epoch 572/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1415
    Epoch 573/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1534
    Epoch 574/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1474
    Epoch 575/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0772
    Epoch 576/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0691
    Epoch 577/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0770
    Epoch 578/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0637
    Epoch 579/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0528
    Epoch 580/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0371
    Epoch 581/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0356
    Epoch 582/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0431
    Epoch 583/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0300
    Epoch 584/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0309
    Epoch 585/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0307
    Epoch 586/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0321
    Epoch 587/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0266
    Epoch 588/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0274
    Epoch 589/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0276
    Epoch 590/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0267
    Epoch 591/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0305
    Epoch 592/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0278
    Epoch 593/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0343
    Epoch 594/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0259
    Epoch 595/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0259
    Epoch 596/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0258
    Epoch 597/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0262
    Epoch 598/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0254
    Epoch 599/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0251
    Epoch 600/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0241
    Epoch 601/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0269
    Epoch 602/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0287
    Epoch 603/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0257
    Epoch 604/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0254
    Epoch 605/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0232
    Epoch 606/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0281
    Epoch 607/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0247
    Epoch 608/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0254
    Epoch 609/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0237
    Epoch 610/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0253
    Epoch 611/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0256
    Epoch 612/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0235
    Epoch 613/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0290
    Epoch 614/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0236
    Epoch 615/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0249
    Epoch 616/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0253
    Epoch 617/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0231
    Epoch 618/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0241
    Epoch 619/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0253
    Epoch 620/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0290
    Epoch 621/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0456
    Epoch 622/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0647
    Epoch 623/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1078
    Epoch 624/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1180
    Epoch 625/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0837
    Epoch 626/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0510
    Epoch 627/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0333
    Epoch 628/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0327
    Epoch 629/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0389
    Epoch 630/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0347
    Epoch 631/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0342
    Epoch 632/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0272
    Epoch 633/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0240
    Epoch 634/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0235
    Epoch 635/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0243
    Epoch 636/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0225
    Epoch 637/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0222
    Epoch 638/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0223
    Epoch 639/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0215
    Epoch 640/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0247
    Epoch 641/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0248
    Epoch 642/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0257
    Epoch 643/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0213
    Epoch 644/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0277
    Epoch 645/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0266
    Epoch 646/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0320
    Epoch 647/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0269
    Epoch 648/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0357
    Epoch 649/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0321
    Epoch 650/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0255
    Epoch 651/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0287
    Epoch 652/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0251
    Epoch 653/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0242
    Epoch 654/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0239
    Epoch 655/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0218
    Epoch 656/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0227
    Epoch 657/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0247
    Epoch 658/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0265
    Epoch 659/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0257
    Epoch 660/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0233
    Epoch 661/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0246
    Epoch 662/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0313
    Epoch 663/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0238
    Epoch 664/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0277
    Epoch 665/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0205
    Epoch 666/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0238
    Epoch 667/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0249
    Epoch 668/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0441
    Epoch 669/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0441
    Epoch 670/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0305
    Epoch 671/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0323
    Epoch 672/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0356
    Epoch 673/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0670
    Epoch 674/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1732
    Epoch 675/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0889
    Epoch 676/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1098
    Epoch 677/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0468
    Epoch 678/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0532
    Epoch 679/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0577
    Epoch 680/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0880
    Epoch 681/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1123
    Epoch 682/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1581
    Epoch 683/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1343
    Epoch 684/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1065
    Epoch 685/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1236
    Epoch 686/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1184
    Epoch 687/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1218
    Epoch 688/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1673
    Epoch 689/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1437
    Epoch 690/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0897
    Epoch 691/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0665
    Epoch 692/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0579
    Epoch 693/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0563
    Epoch 694/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0425
    Epoch 695/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0441
    Epoch 696/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0411
    Epoch 697/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0429
    Epoch 698/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0347
    Epoch 699/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0367
    Epoch 700/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0311
    Epoch 701/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0333
    Epoch 702/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0308
    Epoch 703/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0287
    Epoch 704/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0297
    Epoch 705/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0282
    Epoch 706/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0263
    Epoch 707/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0286
    Epoch 708/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0275
    Epoch 709/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0274
    Epoch 710/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0252
    Epoch 711/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0277
    Epoch 712/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0261
    Epoch 713/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0311
    Epoch 714/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0265
    Epoch 715/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0281
    Epoch 716/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0275
    Epoch 717/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0264
    Epoch 718/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0240
    Epoch 719/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0234
    Epoch 720/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0284
    Epoch 721/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0311
    Epoch 722/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0244
    Epoch 723/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0249
    Epoch 724/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0269
    Epoch 725/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0224
    Epoch 726/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0238
    Epoch 727/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0234
    Epoch 728/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0223
    Epoch 729/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0220
    Epoch 730/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0268
    Epoch 731/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0363
    Epoch 732/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0300
    Epoch 733/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0208
    Epoch 734/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0254
    Epoch 735/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0264
    Epoch 736/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0230
    Epoch 737/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0224
    Epoch 738/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0270
    Epoch 739/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0257
    Epoch 740/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0228
    Epoch 741/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0249
    Epoch 742/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0241
    Epoch 743/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0210
    Epoch 744/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0216
    Epoch 745/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0208
    Epoch 746/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0227
    Epoch 747/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0193
    Epoch 748/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0241
    Epoch 749/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0217
    Epoch 750/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0248
    Epoch 751/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0203
    Epoch 752/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0194
    Epoch 753/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0252
    Epoch 754/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0203
    Epoch 755/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0206
    Epoch 756/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0192
    Epoch 757/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0213
    Epoch 758/1000
    13/13 [==============================] - 0s 2ms/step - loss: 0.0206
    Epoch 759/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0247
    Epoch 760/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0227
    Epoch 761/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0204
    Epoch 762/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0219
    Epoch 763/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0266
    Epoch 764/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0699
    Epoch 765/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0436
    Epoch 766/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0451
    Epoch 767/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1029
    Epoch 768/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1082
    Epoch 769/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0924
    Epoch 770/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0936
    Epoch 771/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0690
    Epoch 772/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0589
    Epoch 773/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0519
    Epoch 774/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0714
    Epoch 775/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1015
    Epoch 776/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0932
    Epoch 777/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1891
    Epoch 778/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1356
    Epoch 779/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1081
    Epoch 780/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0973
    Epoch 781/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0768
    Epoch 782/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0761
    Epoch 783/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1075
    Epoch 784/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0789
    Epoch 785/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0467
    Epoch 786/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0394
    Epoch 787/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0360
    Epoch 788/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0324
    Epoch 789/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0329
    Epoch 790/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0291
    Epoch 791/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0283
    Epoch 792/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0291
    Epoch 793/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0261
    Epoch 794/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0294
    Epoch 795/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0250
    Epoch 796/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0292
    Epoch 797/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0286
    Epoch 798/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0271
    Epoch 799/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0307
    Epoch 800/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0298
    Epoch 801/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0371
    Epoch 802/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0259
    Epoch 803/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0274
    Epoch 804/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0266
    Epoch 805/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0260
    Epoch 806/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0254
    Epoch 807/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0258
    Epoch 808/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0252
    Epoch 809/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0280
    Epoch 810/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0249
    Epoch 811/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0255
    Epoch 812/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0259
    Epoch 813/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0310
    Epoch 814/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0258
    Epoch 815/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0246
    Epoch 816/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0263
    Epoch 817/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0328
    Epoch 818/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0247
    Epoch 819/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0250
    Epoch 820/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0258
    Epoch 821/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0252
    Epoch 822/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0256
    Epoch 823/1000
    13/13 [==============================] - 0s 2ms/step - loss: 0.0299
    Epoch 824/1000
    13/13 [==============================] - 0s 2ms/step - loss: 0.0312
    Epoch 825/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0243
    Epoch 826/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0263
    Epoch 827/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0247
    Epoch 828/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0233
    Epoch 829/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0246
    Epoch 830/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0262
    Epoch 831/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0259
    Epoch 832/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0238
    Epoch 833/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0221
    Epoch 834/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0240
    Epoch 835/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0248
    Epoch 836/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0253
    Epoch 837/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0340
    Epoch 838/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0229
    Epoch 839/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0294
    Epoch 840/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0286
    Epoch 841/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0268
    Epoch 842/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0283
    Epoch 843/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0271
    Epoch 844/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0247
    Epoch 845/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0235
    Epoch 846/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0300
    Epoch 847/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0246
    Epoch 848/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0244
    Epoch 849/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0219
    Epoch 850/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0258
    Epoch 851/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0244
    Epoch 852/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0257
    Epoch 853/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0220
    Epoch 854/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0221
    Epoch 855/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0256
    Epoch 856/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0211
    Epoch 857/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0227
    Epoch 858/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0252
    Epoch 859/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0224
    Epoch 860/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0214
    Epoch 861/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0204
    Epoch 862/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0228
    Epoch 863/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0206
    Epoch 864/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0198
    Epoch 865/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0200
    Epoch 866/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0273
    Epoch 867/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0271
    Epoch 868/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0217
    Epoch 869/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0231
    Epoch 870/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0325
    Epoch 871/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0354
    Epoch 872/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0321
    Epoch 873/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0216
    Epoch 874/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0201
    Epoch 875/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0218
    Epoch 876/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0217
    Epoch 877/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0275
    Epoch 878/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0305
    Epoch 879/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0440
    Epoch 880/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0466
    Epoch 881/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0729
    Epoch 882/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0460
    Epoch 883/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0439
    Epoch 884/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0811
    Epoch 885/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0291
    Epoch 886/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0309
    Epoch 887/1000
    13/13 [==============================] - 0s 4ms/step - loss: 0.0289
    Epoch 888/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0294
    Epoch 889/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0283
    Epoch 890/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0240
    Epoch 891/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0232
    Epoch 892/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0225
    Epoch 893/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0196
    Epoch 894/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0218
    Epoch 895/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0189
    Epoch 896/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0221
    Epoch 897/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0204
    Epoch 898/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0200
    Epoch 899/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0208
    Epoch 900/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0205
    Epoch 901/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0199
    Epoch 902/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0298
    Epoch 903/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0185
    Epoch 904/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0290
    Epoch 905/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0272
    Epoch 906/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0237
    Epoch 907/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0190
    Epoch 908/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0210
    Epoch 909/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0189
    Epoch 910/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0199
    Epoch 911/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0688
    Epoch 912/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1337
    Epoch 913/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1883
    Epoch 914/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.2096
    Epoch 915/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1323
    Epoch 916/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0795
    Epoch 917/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1167
    Epoch 918/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0621
    Epoch 919/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0929
    Epoch 920/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0352
    Epoch 921/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0303
    Epoch 922/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0287
    Epoch 923/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0457
    Epoch 924/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0712
    Epoch 925/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0553
    Epoch 926/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0385
    Epoch 927/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0311
    Epoch 928/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0394
    Epoch 929/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0261
    Epoch 930/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0346
    Epoch 931/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0332
    Epoch 932/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0322
    Epoch 933/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0311
    Epoch 934/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0493
    Epoch 935/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0289
    Epoch 936/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0325
    Epoch 937/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0255
    Epoch 938/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0210
    Epoch 939/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0235
    Epoch 940/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0259
    Epoch 941/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0371
    Epoch 942/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0300
    Epoch 943/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0265
    Epoch 944/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0327
    Epoch 945/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0367
    Epoch 946/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0307
    Epoch 947/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0376
    Epoch 948/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0375
    Epoch 949/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0350
    Epoch 950/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0284
    Epoch 951/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0293
    Epoch 952/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0374
    Epoch 953/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0353
    Epoch 954/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0395
    Epoch 955/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0405
    Epoch 956/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0432
    Epoch 957/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0234
    Epoch 958/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0266
    Epoch 959/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0213
    Epoch 960/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0200
    Epoch 961/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0203
    Epoch 962/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0190
    Epoch 963/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0239
    Epoch 964/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0240
    Epoch 965/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0261
    Epoch 966/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0197
    Epoch 967/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0206
    Epoch 968/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0188
    Epoch 969/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0200
    Epoch 970/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0169
    Epoch 971/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0161
    Epoch 972/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0176
    Epoch 973/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0218
    Epoch 974/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0161
    Epoch 975/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0203
    Epoch 976/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0384
    Epoch 977/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0292
    Epoch 978/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0234
    Epoch 979/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0522
    Epoch 980/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0851
    Epoch 981/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0541
    Epoch 982/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0380
    Epoch 983/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0328
    Epoch 984/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0276
    Epoch 985/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0227
    Epoch 986/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0235
    Epoch 987/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0287
    Epoch 988/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0170
    Epoch 989/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0166
    Epoch 990/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0175
    Epoch 991/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0149
    Epoch 992/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0152
    Epoch 993/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0153
    Epoch 994/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0142
    Epoch 995/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0199
    Epoch 996/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0231
    Epoch 997/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0199
    Epoch 998/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0188
    Epoch 999/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0155
    Epoch 1000/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0172





    <keras.callbacks.History at 0x7f2914699250>




```python
# BEGIN UNIT TEST
model.summary()

model_test(model, classes, X_train.shape[1]) 
# END UNIT TEST
```

    Model: "Complex"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     dense (Dense)               (None, 120)               360       
                                                                     
     dense_1 (Dense)             (None, 40)                4840      
                                                                     
     dense_2 (Dense)             (None, 6)                 246       
                                                                     
    =================================================================
    Total params: 5,446
    Trainable params: 5,446
    Non-trainable params: 0
    _________________________________________________________________
    [92mAll tests passed!


<details>
  <summary><font size="3" color="darkgreen"><b>Click for hints</b></font></summary>
    
Summary should match this (layer instance names may increment )
```
Model: "Complex"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
L1 (Dense)                   (None, 120)               360       
_________________________________________________________________
L2 (Dense)                   (None, 40)                4840      
_________________________________________________________________
L3 (Dense)                   (None, 6)                 246       
=================================================================
Total params: 5,446
Trainable params: 5,446
Non-trainable params: 0
_________________________________________________________________
```
  <details>
  <summary><font size="3" color="darkgreen"><b>Click for more hints</b></font></summary>
  
```python
tf.random.set_seed(1234)
model = Sequential(
    [
        Dense(120, activation = 'relu', name = "L1"),      
        Dense(40, activation = 'relu', name = "L2"),         
        Dense(classes, activation = 'linear', name = "L3")  
    ], name="Complex"
)
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),          
    optimizer=tf.keras.optimizers.Adam(0.01),   
)

model.fit(
    X_train,y_train,
    epochs=1000
)                                  
``` 


```python
#make a model for plotting routines to call
model_predict = lambda Xl: np.argmax(tf.nn.softmax(model.predict(Xl)).numpy(),axis=1)
plt_nn(model_predict,X_train,y_train, classes, X_cv, y_cv, suptitle="Complex Model")
```


    Canvas(toolbar=Toolbar(toolitems=[('Home', 'Reset original view', 'home', 'home'), ('Back', 'Back to previous …


This model has worked very hard to capture outliers of each category. As a result, it has miscategorized some of the cross-validation data. Let's calculate the classification error.


```python
training_cerr_complex = eval_cat_err(y_train, model_predict(X_train))
cv_cerr_complex = eval_cat_err(y_cv, model_predict(X_cv))
print(f"categorization error, training, complex model: {training_cerr_complex:0.3f}")
print(f"categorization error, cv,       complex model: {cv_cerr_complex:0.3f}")
```

    categorization error, training, complex model: 0.003
    categorization error, cv,       complex model: 0.122


<a name="5.1"></a>
### 5.1 Simple model
Now, let's try a simple model

<a name="ex04"></a>
### Exercise 4

Below, compose a two-layer model:
* Dense layer with 6 units, relu activation
* Dense layer with 6 units and a linear activation. 
Compile using
* loss with `SparseCategoricalCrossentropy`, remember to use  `from_logits=True`
* Adam optimizer with learning rate of 0.01.


```python
# UNQ_C4
# GRADED CELL: model_s

tf.random.set_seed(1234)
model_s = Sequential(
    [
        ### START CODE HERE ### 
        tf.keras.layers.Dense(6, activation="relu"),
        tf.keras.layers.Dense(6, activation="linear")
        ### END CODE HERE ### 
    ], name = "Simple"
)
model_s.compile(
    ### START CODE HERE ### 
    loss=SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(lr=0.01),
    ### START CODE HERE ### 
)
```


```python
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# BEGIN UNIT TEST
model_s.fit(
    X_train,y_train,
    epochs=1000
)
# END UNIT TEST
```

    Epoch 1/1000
    13/13 [==============================] - 0s 926us/step - loss: 1.7306
    Epoch 2/1000
    13/13 [==============================] - 0s 869us/step - loss: 1.4468
    Epoch 3/1000
    13/13 [==============================] - 0s 830us/step - loss: 1.2902
    Epoch 4/1000
    13/13 [==============================] - 0s 909us/step - loss: 1.1367
    Epoch 5/1000
    13/13 [==============================] - 0s 968us/step - loss: 0.9710
    Epoch 6/1000
    13/13 [==============================] - 0s 939us/step - loss: 0.7947
    Epoch 7/1000
    13/13 [==============================] - 0s 913us/step - loss: 0.6499
    Epoch 8/1000
    13/13 [==============================] - 0s 880us/step - loss: 0.5378
    Epoch 9/1000
    13/13 [==============================] - 0s 862us/step - loss: 0.4652
    Epoch 10/1000
    13/13 [==============================] - 0s 911us/step - loss: 0.4184
    Epoch 11/1000
    13/13 [==============================] - 0s 916us/step - loss: 0.3860
    Epoch 12/1000
    13/13 [==============================] - 0s 924us/step - loss: 0.3641
    Epoch 13/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3487
    Epoch 14/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3316
    Epoch 15/1000
    13/13 [==============================] - 0s 813us/step - loss: 0.3201
    Epoch 16/1000
    13/13 [==============================] - 0s 817us/step - loss: 0.3110
    Epoch 17/1000
    13/13 [==============================] - 0s 797us/step - loss: 0.3026
    Epoch 18/1000
    13/13 [==============================] - 0s 813us/step - loss: 0.2953
    Epoch 19/1000
    13/13 [==============================] - 0s 800us/step - loss: 0.2880
    Epoch 20/1000
    13/13 [==============================] - 0s 994us/step - loss: 0.2824
    Epoch 21/1000
    13/13 [==============================] - 0s 899us/step - loss: 0.2768
    Epoch 22/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2716
    Epoch 23/1000
    13/13 [==============================] - 0s 816us/step - loss: 0.2690
    Epoch 24/1000
    13/13 [==============================] - 0s 803us/step - loss: 0.2618
    Epoch 25/1000
    13/13 [==============================] - 0s 802us/step - loss: 0.2606
    Epoch 26/1000
    13/13 [==============================] - 0s 861us/step - loss: 0.2560
    Epoch 27/1000
    13/13 [==============================] - 0s 931us/step - loss: 0.2516
    Epoch 28/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2500
    Epoch 29/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2497
    Epoch 30/1000
    13/13 [==============================] - 0s 851us/step - loss: 0.2424
    Epoch 31/1000
    13/13 [==============================] - 0s 805us/step - loss: 0.2406
    Epoch 32/1000
    13/13 [==============================] - 0s 807us/step - loss: 0.2386
    Epoch 33/1000
    13/13 [==============================] - 0s 807us/step - loss: 0.2371
    Epoch 34/1000
    13/13 [==============================] - 0s 802us/step - loss: 0.2355
    Epoch 35/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2328
    Epoch 36/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2311
    Epoch 37/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2289
    Epoch 38/1000
    13/13 [==============================] - 0s 921us/step - loss: 0.2271
    Epoch 39/1000
    13/13 [==============================] - 0s 938us/step - loss: 0.2278
    Epoch 40/1000
    13/13 [==============================] - 0s 914us/step - loss: 0.2269
    Epoch 41/1000
    13/13 [==============================] - 0s 893us/step - loss: 0.2244
    Epoch 42/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2250
    Epoch 43/1000
    13/13 [==============================] - 0s 903us/step - loss: 0.2228
    Epoch 44/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2227
    Epoch 45/1000
    13/13 [==============================] - 0s 805us/step - loss: 0.2230
    Epoch 46/1000
    13/13 [==============================] - 0s 795us/step - loss: 0.2198
    Epoch 47/1000
    13/13 [==============================] - 0s 811us/step - loss: 0.2188
    Epoch 48/1000
    13/13 [==============================] - 0s 808us/step - loss: 0.2156
    Epoch 49/1000
    13/13 [==============================] - 0s 797us/step - loss: 0.2156
    Epoch 50/1000
    13/13 [==============================] - 0s 888us/step - loss: 0.2165
    Epoch 51/1000
    13/13 [==============================] - 0s 872us/step - loss: 0.2155
    Epoch 52/1000
    13/13 [==============================] - 0s 892us/step - loss: 0.2130
    Epoch 53/1000
    13/13 [==============================] - 0s 822us/step - loss: 0.2121
    Epoch 54/1000
    13/13 [==============================] - 0s 806us/step - loss: 0.2122
    Epoch 55/1000
    13/13 [==============================] - 0s 779us/step - loss: 0.2105
    Epoch 56/1000
    13/13 [==============================] - 0s 796us/step - loss: 0.2116
    Epoch 57/1000
    13/13 [==============================] - 0s 802us/step - loss: 0.2121
    Epoch 58/1000
    13/13 [==============================] - 0s 827us/step - loss: 0.2084
    Epoch 59/1000
    13/13 [==============================] - 0s 825us/step - loss: 0.2122
    Epoch 60/1000
    13/13 [==============================] - 0s 840us/step - loss: 0.2101
    Epoch 61/1000
    13/13 [==============================] - 0s 856us/step - loss: 0.2095
    Epoch 62/1000
    13/13 [==============================] - 0s 829us/step - loss: 0.2092
    Epoch 63/1000
    13/13 [==============================] - 0s 821us/step - loss: 0.2116
    Epoch 64/1000
    13/13 [==============================] - 0s 824us/step - loss: 0.2085
    Epoch 65/1000
    13/13 [==============================] - 0s 818us/step - loss: 0.2120
    Epoch 66/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2087
    Epoch 67/1000
    13/13 [==============================] - 0s 993us/step - loss: 0.2107
    Epoch 68/1000
    13/13 [==============================] - 0s 955us/step - loss: 0.2090
    Epoch 69/1000
    13/13 [==============================] - 0s 863us/step - loss: 0.2084
    Epoch 70/1000
    13/13 [==============================] - 0s 800us/step - loss: 0.2053
    Epoch 71/1000
    13/13 [==============================] - 0s 806us/step - loss: 0.2060
    Epoch 72/1000
    13/13 [==============================] - 0s 801us/step - loss: 0.2061
    Epoch 73/1000
    13/13 [==============================] - 0s 805us/step - loss: 0.2075
    Epoch 74/1000
    13/13 [==============================] - 0s 899us/step - loss: 0.2067
    Epoch 75/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2039
    Epoch 76/1000
    13/13 [==============================] - 0s 973us/step - loss: 0.2036
    Epoch 77/1000
    13/13 [==============================] - 0s 833us/step - loss: 0.2062
    Epoch 78/1000
    13/13 [==============================] - 0s 883us/step - loss: 0.2017
    Epoch 79/1000
    13/13 [==============================] - 0s 834us/step - loss: 0.2044
    Epoch 80/1000
    13/13 [==============================] - 0s 837us/step - loss: 0.2055
    Epoch 81/1000
    13/13 [==============================] - 0s 860us/step - loss: 0.1999
    Epoch 82/1000
    13/13 [==============================] - 0s 987us/step - loss: 0.2028
    Epoch 83/1000
    13/13 [==============================] - 0s 989us/step - loss: 0.2019
    Epoch 84/1000
    13/13 [==============================] - 0s 888us/step - loss: 0.2042
    Epoch 85/1000
    13/13 [==============================] - 0s 836us/step - loss: 0.2016
    Epoch 86/1000
    13/13 [==============================] - 0s 838us/step - loss: 0.2068
    Epoch 87/1000
    13/13 [==============================] - 0s 819us/step - loss: 0.2005
    Epoch 88/1000
    13/13 [==============================] - 0s 820us/step - loss: 0.2011
    Epoch 89/1000
    13/13 [==============================] - 0s 977us/step - loss: 0.2000
    Epoch 90/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1998
    Epoch 91/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1992
    Epoch 92/1000
    13/13 [==============================] - 0s 818us/step - loss: 0.2001
    Epoch 93/1000
    13/13 [==============================] - 0s 792us/step - loss: 0.1997
    Epoch 94/1000
    13/13 [==============================] - 0s 798us/step - loss: 0.2008
    Epoch 95/1000
    13/13 [==============================] - 0s 803us/step - loss: 0.2015
    Epoch 96/1000
    13/13 [==============================] - 0s 817us/step - loss: 0.2011
    Epoch 97/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2006
    Epoch 98/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2031
    Epoch 99/1000
    13/13 [==============================] - 0s 840us/step - loss: 0.1991
    Epoch 100/1000
    13/13 [==============================] - 0s 827us/step - loss: 0.2006
    Epoch 101/1000
    13/13 [==============================] - 0s 799us/step - loss: 0.2010
    Epoch 102/1000
    13/13 [==============================] - 0s 792us/step - loss: 0.2018
    Epoch 103/1000
    13/13 [==============================] - 0s 803us/step - loss: 0.2026
    Epoch 104/1000
    13/13 [==============================] - 0s 921us/step - loss: 0.1988
    Epoch 105/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1974
    Epoch 106/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1966
    Epoch 107/1000
    13/13 [==============================] - 0s 831us/step - loss: 0.1963
    Epoch 108/1000
    13/13 [==============================] - 0s 811us/step - loss: 0.1969
    Epoch 109/1000
    13/13 [==============================] - 0s 809us/step - loss: 0.1987
    Epoch 110/1000
    13/13 [==============================] - 0s 835us/step - loss: 0.1978
    Epoch 111/1000
    13/13 [==============================] - 0s 920us/step - loss: 0.1962
    Epoch 112/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1979
    Epoch 113/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1944
    Epoch 114/1000
    13/13 [==============================] - 0s 832us/step - loss: 0.1987
    Epoch 115/1000
    13/13 [==============================] - 0s 991us/step - loss: 0.1934
    Epoch 116/1000
    13/13 [==============================] - 0s 842us/step - loss: 0.2009
    Epoch 117/1000
    13/13 [==============================] - 0s 846us/step - loss: 0.1943
    Epoch 118/1000
    13/13 [==============================] - 0s 837us/step - loss: 0.1969
    Epoch 119/1000
    13/13 [==============================] - 0s 952us/step - loss: 0.1951
    Epoch 120/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1964
    Epoch 121/1000
    13/13 [==============================] - 0s 943us/step - loss: 0.1957
    Epoch 122/1000
    13/13 [==============================] - 0s 825us/step - loss: 0.1970
    Epoch 123/1000
    13/13 [==============================] - 0s 811us/step - loss: 0.1960
    Epoch 124/1000
    13/13 [==============================] - 0s 809us/step - loss: 0.1973
    Epoch 125/1000
    13/13 [==============================] - 0s 813us/step - loss: 0.1961
    Epoch 126/1000
    13/13 [==============================] - 0s 854us/step - loss: 0.1957
    Epoch 127/1000
    13/13 [==============================] - 0s 923us/step - loss: 0.1949
    Epoch 128/1000
    13/13 [==============================] - 0s 939us/step - loss: 0.1946
    Epoch 129/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1944
    Epoch 130/1000
    13/13 [==============================] - 0s 855us/step - loss: 0.1969
    Epoch 131/1000
    13/13 [==============================] - 0s 821us/step - loss: 0.1926
    Epoch 132/1000
    13/13 [==============================] - 0s 850us/step - loss: 0.1925
    Epoch 133/1000
    13/13 [==============================] - 0s 846us/step - loss: 0.1933
    Epoch 134/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1942
    Epoch 135/1000
    13/13 [==============================] - 0s 965us/step - loss: 0.1976
    Epoch 136/1000
    13/13 [==============================] - 0s 920us/step - loss: 0.1939
    Epoch 137/1000
    13/13 [==============================] - 0s 840us/step - loss: 0.1931
    Epoch 138/1000
    13/13 [==============================] - 0s 834us/step - loss: 0.1947
    Epoch 139/1000
    13/13 [==============================] - 0s 809us/step - loss: 0.1941
    Epoch 140/1000
    13/13 [==============================] - 0s 806us/step - loss: 0.1917
    Epoch 141/1000
    13/13 [==============================] - 0s 807us/step - loss: 0.1922
    Epoch 142/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1917
    Epoch 143/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1944
    Epoch 144/1000
    13/13 [==============================] - 0s 876us/step - loss: 0.1948
    Epoch 145/1000
    13/13 [==============================] - 0s 828us/step - loss: 0.1921
    Epoch 146/1000
    13/13 [==============================] - 0s 803us/step - loss: 0.1920
    Epoch 147/1000
    13/13 [==============================] - 0s 809us/step - loss: 0.1925
    Epoch 148/1000
    13/13 [==============================] - 0s 795us/step - loss: 0.1899
    Epoch 149/1000
    13/13 [==============================] - 0s 889us/step - loss: 0.1913
    Epoch 150/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1914
    Epoch 151/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1944
    Epoch 152/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1920
    Epoch 153/1000
    13/13 [==============================] - 0s 836us/step - loss: 0.1949
    Epoch 154/1000
    13/13 [==============================] - 0s 819us/step - loss: 0.1904
    Epoch 155/1000
    13/13 [==============================] - 0s 888us/step - loss: 0.1917
    Epoch 156/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1898
    Epoch 157/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1913
    Epoch 158/1000
    13/13 [==============================] - 0s 959us/step - loss: 0.1905
    Epoch 159/1000
    13/13 [==============================] - 0s 887us/step - loss: 0.1898
    Epoch 160/1000
    13/13 [==============================] - 0s 886us/step - loss: 0.1910
    Epoch 161/1000
    13/13 [==============================] - 0s 884us/step - loss: 0.1913
    Epoch 162/1000
    13/13 [==============================] - 0s 877us/step - loss: 0.1930
    Epoch 163/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1913
    Epoch 164/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1907
    Epoch 165/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1910
    Epoch 166/1000
    13/13 [==============================] - 0s 838us/step - loss: 0.1891
    Epoch 167/1000
    13/13 [==============================] - 0s 799us/step - loss: 0.1940
    Epoch 168/1000
    13/13 [==============================] - 0s 808us/step - loss: 0.1914
    Epoch 169/1000
    13/13 [==============================] - 0s 799us/step - loss: 0.1914
    Epoch 170/1000
    13/13 [==============================] - 0s 815us/step - loss: 0.1893
    Epoch 171/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1894
    Epoch 172/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1879
    Epoch 173/1000
    13/13 [==============================] - 0s 960us/step - loss: 0.1924
    Epoch 174/1000
    13/13 [==============================] - 0s 890us/step - loss: 0.1887
    Epoch 175/1000
    13/13 [==============================] - 0s 865us/step - loss: 0.1876
    Epoch 176/1000
    13/13 [==============================] - 0s 907us/step - loss: 0.1861
    Epoch 177/1000
    13/13 [==============================] - 0s 887us/step - loss: 0.1922
    Epoch 178/1000
    13/13 [==============================] - 0s 921us/step - loss: 0.1977
    Epoch 179/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1881
    Epoch 180/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1894
    Epoch 181/1000
    13/13 [==============================] - 0s 819us/step - loss: 0.1906
    Epoch 182/1000
    13/13 [==============================] - 0s 823us/step - loss: 0.1894
    Epoch 183/1000
    13/13 [==============================] - 0s 814us/step - loss: 0.1872
    Epoch 184/1000
    13/13 [==============================] - 0s 807us/step - loss: 0.1893
    Epoch 185/1000
    13/13 [==============================] - 0s 850us/step - loss: 0.1885
    Epoch 186/1000
    13/13 [==============================] - 0s 966us/step - loss: 0.1867
    Epoch 187/1000
    13/13 [==============================] - 0s 993us/step - loss: 0.1866
    Epoch 188/1000
    13/13 [==============================] - 0s 910us/step - loss: 0.1884
    Epoch 189/1000
    13/13 [==============================] - 0s 858us/step - loss: 0.1907
    Epoch 190/1000
    13/13 [==============================] - 0s 806us/step - loss: 0.1890
    Epoch 191/1000
    13/13 [==============================] - 0s 797us/step - loss: 0.1880
    Epoch 192/1000
    13/13 [==============================] - 0s 798us/step - loss: 0.1863
    Epoch 193/1000
    13/13 [==============================] - 0s 881us/step - loss: 0.1904
    Epoch 194/1000
    13/13 [==============================] - 0s 935us/step - loss: 0.1857
    Epoch 195/1000
    13/13 [==============================] - 0s 920us/step - loss: 0.1859
    Epoch 196/1000
    13/13 [==============================] - 0s 895us/step - loss: 0.1856
    Epoch 197/1000
    13/13 [==============================] - 0s 842us/step - loss: 0.1879
    Epoch 198/1000
    13/13 [==============================] - 0s 828us/step - loss: 0.1884
    Epoch 199/1000
    13/13 [==============================] - 0s 891us/step - loss: 0.1894
    Epoch 200/1000
    13/13 [==============================] - 0s 835us/step - loss: 0.1860
    Epoch 201/1000
    13/13 [==============================] - 0s 907us/step - loss: 0.1869
    Epoch 202/1000
    13/13 [==============================] - 0s 921us/step - loss: 0.1837
    Epoch 203/1000
    13/13 [==============================] - 0s 982us/step - loss: 0.1861
    Epoch 204/1000
    13/13 [==============================] - 0s 823us/step - loss: 0.1869
    Epoch 205/1000
    13/13 [==============================] - 0s 851us/step - loss: 0.1846
    Epoch 206/1000
    13/13 [==============================] - 0s 838us/step - loss: 0.1881
    Epoch 207/1000
    13/13 [==============================] - 0s 825us/step - loss: 0.1841
    Epoch 208/1000
    13/13 [==============================] - 0s 905us/step - loss: 0.1902
    Epoch 209/1000
    13/13 [==============================] - 0s 977us/step - loss: 0.1850
    Epoch 210/1000
    13/13 [==============================] - 0s 927us/step - loss: 0.1883
    Epoch 211/1000
    13/13 [==============================] - 0s 842us/step - loss: 0.1863
    Epoch 212/1000
    13/13 [==============================] - 0s 869us/step - loss: 0.1856
    Epoch 213/1000
    13/13 [==============================] - 0s 841us/step - loss: 0.1860
    Epoch 214/1000
    13/13 [==============================] - 0s 833us/step - loss: 0.1890
    Epoch 215/1000
    13/13 [==============================] - 0s 878us/step - loss: 0.1855
    Epoch 216/1000
    13/13 [==============================] - 0s 887us/step - loss: 0.1891
    Epoch 217/1000
    13/13 [==============================] - 0s 910us/step - loss: 0.1834
    Epoch 218/1000
    13/13 [==============================] - 0s 918us/step - loss: 0.1887
    Epoch 219/1000
    13/13 [==============================] - 0s 855us/step - loss: 0.1857
    Epoch 220/1000
    13/13 [==============================] - 0s 855us/step - loss: 0.1844
    Epoch 221/1000
    13/13 [==============================] - 0s 849us/step - loss: 0.1846
    Epoch 222/1000
    13/13 [==============================] - 0s 845us/step - loss: 0.1843
    Epoch 223/1000
    13/13 [==============================] - 0s 859us/step - loss: 0.1878
    Epoch 224/1000
    13/13 [==============================] - 0s 892us/step - loss: 0.1884
    Epoch 225/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1851
    Epoch 226/1000
    13/13 [==============================] - 0s 847us/step - loss: 0.1844
    Epoch 227/1000
    13/13 [==============================] - 0s 835us/step - loss: 0.1824
    Epoch 228/1000
    13/13 [==============================] - 0s 880us/step - loss: 0.1849
    Epoch 229/1000
    13/13 [==============================] - 0s 857us/step - loss: 0.1879
    Epoch 230/1000
    13/13 [==============================] - 0s 834us/step - loss: 0.1860
    Epoch 231/1000
    13/13 [==============================] - 0s 979us/step - loss: 0.1834
    Epoch 232/1000
    13/13 [==============================] - 0s 937us/step - loss: 0.1882
    Epoch 233/1000
    13/13 [==============================] - 0s 967us/step - loss: 0.1851
    Epoch 234/1000
    13/13 [==============================] - 0s 840us/step - loss: 0.1874
    Epoch 235/1000
    13/13 [==============================] - 0s 835us/step - loss: 0.1822
    Epoch 236/1000
    13/13 [==============================] - 0s 820us/step - loss: 0.1841
    Epoch 237/1000
    13/13 [==============================] - 0s 835us/step - loss: 0.1876
    Epoch 238/1000
    13/13 [==============================] - 0s 834us/step - loss: 0.1923
    Epoch 239/1000
    13/13 [==============================] - 0s 987us/step - loss: 0.1867
    Epoch 240/1000
    13/13 [==============================] - 0s 960us/step - loss: 0.1832
    Epoch 241/1000
    13/13 [==============================] - 0s 864us/step - loss: 0.1863
    Epoch 242/1000
    13/13 [==============================] - 0s 858us/step - loss: 0.1978
    Epoch 243/1000
    13/13 [==============================] - 0s 857us/step - loss: 0.1946
    Epoch 244/1000
    13/13 [==============================] - 0s 854us/step - loss: 0.1871
    Epoch 245/1000
    13/13 [==============================] - 0s 845us/step - loss: 0.1826
    Epoch 246/1000
    13/13 [==============================] - 0s 903us/step - loss: 0.1850
    Epoch 247/1000
    13/13 [==============================] - 0s 861us/step - loss: 0.1836
    Epoch 248/1000
    13/13 [==============================] - 0s 915us/step - loss: 0.1820
    Epoch 249/1000
    13/13 [==============================] - 0s 875us/step - loss: 0.1857
    Epoch 250/1000
    13/13 [==============================] - 0s 808us/step - loss: 0.1829
    Epoch 251/1000
    13/13 [==============================] - 0s 818us/step - loss: 0.1838
    Epoch 252/1000
    13/13 [==============================] - 0s 843us/step - loss: 0.1828
    Epoch 253/1000
    13/13 [==============================] - 0s 826us/step - loss: 0.1842
    Epoch 254/1000
    13/13 [==============================] - 0s 868us/step - loss: 0.1832
    Epoch 255/1000
    13/13 [==============================] - 0s 893us/step - loss: 0.1830
    Epoch 256/1000
    13/13 [==============================] - 0s 937us/step - loss: 0.1830
    Epoch 257/1000
    13/13 [==============================] - 0s 852us/step - loss: 0.1833
    Epoch 258/1000
    13/13 [==============================] - 0s 839us/step - loss: 0.1826
    Epoch 259/1000
    13/13 [==============================] - 0s 843us/step - loss: 0.1796
    Epoch 260/1000
    13/13 [==============================] - 0s 846us/step - loss: 0.1876
    Epoch 261/1000
    13/13 [==============================] - 0s 840us/step - loss: 0.1819
    Epoch 262/1000
    13/13 [==============================] - 0s 923us/step - loss: 0.1826
    Epoch 263/1000
    13/13 [==============================] - 0s 926us/step - loss: 0.1827
    Epoch 264/1000
    13/13 [==============================] - 0s 896us/step - loss: 0.1820
    Epoch 265/1000
    13/13 [==============================] - 0s 844us/step - loss: 0.1831
    Epoch 266/1000
    13/13 [==============================] - 0s 844us/step - loss: 0.1805
    Epoch 267/1000
    13/13 [==============================] - 0s 848us/step - loss: 0.1835
    Epoch 268/1000
    13/13 [==============================] - 0s 839us/step - loss: 0.1812
    Epoch 269/1000
    13/13 [==============================] - 0s 870us/step - loss: 0.1817
    Epoch 270/1000
    13/13 [==============================] - 0s 947us/step - loss: 0.1836
    Epoch 271/1000
    13/13 [==============================] - 0s 936us/step - loss: 0.1801
    Epoch 272/1000
    13/13 [==============================] - 0s 992us/step - loss: 0.1868
    Epoch 273/1000
    13/13 [==============================] - 0s 877us/step - loss: 0.1869
    Epoch 274/1000
    13/13 [==============================] - 0s 919us/step - loss: 0.1815
    Epoch 275/1000
    13/13 [==============================] - 0s 944us/step - loss: 0.1847
    Epoch 276/1000
    13/13 [==============================] - 0s 874us/step - loss: 0.1787
    Epoch 277/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1841
    Epoch 278/1000
    13/13 [==============================] - 0s 936us/step - loss: 0.1804
    Epoch 279/1000
    13/13 [==============================] - 0s 992us/step - loss: 0.1861
    Epoch 280/1000
    13/13 [==============================] - 0s 806us/step - loss: 0.1816
    Epoch 281/1000
    13/13 [==============================] - 0s 818us/step - loss: 0.1797
    Epoch 282/1000
    13/13 [==============================] - 0s 788us/step - loss: 0.1807
    Epoch 283/1000
    13/13 [==============================] - 0s 789us/step - loss: 0.1815
    Epoch 284/1000
    13/13 [==============================] - 0s 921us/step - loss: 0.1822
    Epoch 285/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1813
    Epoch 286/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1815
    Epoch 287/1000
    13/13 [==============================] - 0s 813us/step - loss: 0.1829
    Epoch 288/1000
    13/13 [==============================] - 0s 779us/step - loss: 0.1849
    Epoch 289/1000
    13/13 [==============================] - 0s 786us/step - loss: 0.1805
    Epoch 290/1000
    13/13 [==============================] - 0s 805us/step - loss: 0.1807
    Epoch 291/1000
    13/13 [==============================] - 0s 794us/step - loss: 0.1801
    Epoch 292/1000
    13/13 [==============================] - 0s 937us/step - loss: 0.1793
    Epoch 293/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1815
    Epoch 294/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1784
    Epoch 295/1000
    13/13 [==============================] - 0s 841us/step - loss: 0.1867
    Epoch 296/1000
    13/13 [==============================] - 0s 834us/step - loss: 0.1805
    Epoch 297/1000
    13/13 [==============================] - 0s 822us/step - loss: 0.1855
    Epoch 298/1000
    13/13 [==============================] - 0s 820us/step - loss: 0.1816
    Epoch 299/1000
    13/13 [==============================] - 0s 924us/step - loss: 0.1798
    Epoch 300/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1817
    Epoch 301/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1823
    Epoch 302/1000
    13/13 [==============================] - 0s 827us/step - loss: 0.1878
    Epoch 303/1000
    13/13 [==============================] - 0s 810us/step - loss: 0.1788
    Epoch 304/1000
    13/13 [==============================] - 0s 801us/step - loss: 0.1850
    Epoch 305/1000
    13/13 [==============================] - 0s 801us/step - loss: 0.1827
    Epoch 306/1000
    13/13 [==============================] - 0s 801us/step - loss: 0.1818
    Epoch 307/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1811
    Epoch 308/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1827
    Epoch 309/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1814
    Epoch 310/1000
    13/13 [==============================] - 0s 806us/step - loss: 0.1854
    Epoch 311/1000
    13/13 [==============================] - 0s 803us/step - loss: 0.1785
    Epoch 312/1000
    13/13 [==============================] - 0s 794us/step - loss: 0.1831
    Epoch 313/1000
    13/13 [==============================] - 0s 808us/step - loss: 0.1775
    Epoch 314/1000
    13/13 [==============================] - 0s 892us/step - loss: 0.1820
    Epoch 315/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1801
    Epoch 316/1000
    13/13 [==============================] - 0s 972us/step - loss: 0.1792
    Epoch 317/1000
    13/13 [==============================] - 0s 885us/step - loss: 0.1847
    Epoch 318/1000
    13/13 [==============================] - 0s 818us/step - loss: 0.1841
    Epoch 319/1000
    13/13 [==============================] - 0s 817us/step - loss: 0.1811
    Epoch 320/1000
    13/13 [==============================] - 0s 811us/step - loss: 0.1841
    Epoch 321/1000
    13/13 [==============================] - 0s 835us/step - loss: 0.1785
    Epoch 322/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1815
    Epoch 323/1000
    13/13 [==============================] - 0s 958us/step - loss: 0.1792
    Epoch 324/1000
    13/13 [==============================] - 0s 903us/step - loss: 0.1829
    Epoch 325/1000
    13/13 [==============================] - 0s 942us/step - loss: 0.1800
    Epoch 326/1000
    13/13 [==============================] - 0s 877us/step - loss: 0.1783
    Epoch 327/1000
    13/13 [==============================] - 0s 904us/step - loss: 0.1797
    Epoch 328/1000
    13/13 [==============================] - 0s 841us/step - loss: 0.1846
    Epoch 329/1000
    13/13 [==============================] - 0s 928us/step - loss: 0.1790
    Epoch 330/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1815
    Epoch 331/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1801
    Epoch 332/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1803
    Epoch 333/1000
    13/13 [==============================] - 0s 924us/step - loss: 0.1824
    Epoch 334/1000
    13/13 [==============================] - 0s 861us/step - loss: 0.1849
    Epoch 335/1000
    13/13 [==============================] - 0s 864us/step - loss: 0.1835
    Epoch 336/1000
    13/13 [==============================] - 0s 936us/step - loss: 0.1797
    Epoch 337/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1805
    Epoch 338/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1796
    Epoch 339/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1807
    Epoch 340/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1794
    Epoch 341/1000
    13/13 [==============================] - 0s 943us/step - loss: 0.1808
    Epoch 342/1000
    13/13 [==============================] - 0s 997us/step - loss: 0.1790
    Epoch 343/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1797
    Epoch 344/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1804
    Epoch 345/1000
    13/13 [==============================] - 0s 830us/step - loss: 0.1838
    Epoch 346/1000
    13/13 [==============================] - 0s 821us/step - loss: 0.1832
    Epoch 347/1000
    13/13 [==============================] - 0s 804us/step - loss: 0.1819
    Epoch 348/1000
    13/13 [==============================] - 0s 848us/step - loss: 0.1800
    Epoch 349/1000
    13/13 [==============================] - 0s 857us/step - loss: 0.1789
    Epoch 350/1000
    13/13 [==============================] - 0s 958us/step - loss: 0.1787
    Epoch 351/1000
    13/13 [==============================] - 0s 951us/step - loss: 0.1784
    Epoch 352/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1846
    Epoch 353/1000
    13/13 [==============================] - 0s 857us/step - loss: 0.1826
    Epoch 354/1000
    13/13 [==============================] - 0s 848us/step - loss: 0.1802
    Epoch 355/1000
    13/13 [==============================] - 0s 861us/step - loss: 0.1792
    Epoch 356/1000
    13/13 [==============================] - 0s 873us/step - loss: 0.1786
    Epoch 357/1000
    13/13 [==============================] - 0s 993us/step - loss: 0.1802
    Epoch 358/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1781
    Epoch 359/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1800
    Epoch 360/1000
    13/13 [==============================] - 0s 899us/step - loss: 0.1821
    Epoch 361/1000
    13/13 [==============================] - 0s 857us/step - loss: 0.1789
    Epoch 362/1000
    13/13 [==============================] - 0s 859us/step - loss: 0.1798
    Epoch 363/1000
    13/13 [==============================] - 0s 842us/step - loss: 0.1815
    Epoch 364/1000
    13/13 [==============================] - 0s 924us/step - loss: 0.1799
    Epoch 365/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1811
    Epoch 366/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1785
    Epoch 367/1000
    13/13 [==============================] - 0s 898us/step - loss: 0.1776
    Epoch 368/1000
    13/13 [==============================] - 0s 850us/step - loss: 0.1784
    Epoch 369/1000
    13/13 [==============================] - 0s 824us/step - loss: 0.1819
    Epoch 370/1000
    13/13 [==============================] - 0s 802us/step - loss: 0.1771
    Epoch 371/1000
    13/13 [==============================] - 0s 819us/step - loss: 0.1799
    Epoch 372/1000
    13/13 [==============================] - 0s 858us/step - loss: 0.1780
    Epoch 373/1000
    13/13 [==============================] - 0s 839us/step - loss: 0.1773
    Epoch 374/1000
    13/13 [==============================] - 0s 869us/step - loss: 0.1769
    Epoch 375/1000
    13/13 [==============================] - 0s 878us/step - loss: 0.1770
    Epoch 376/1000
    13/13 [==============================] - 0s 937us/step - loss: 0.1766
    Epoch 377/1000
    13/13 [==============================] - 0s 994us/step - loss: 0.1768
    Epoch 378/1000
    13/13 [==============================] - 0s 921us/step - loss: 0.1794
    Epoch 379/1000
    13/13 [==============================] - 0s 999us/step - loss: 0.1799
    Epoch 380/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1768
    Epoch 381/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1805
    Epoch 382/1000
    13/13 [==============================] - 0s 887us/step - loss: 0.1782
    Epoch 383/1000
    13/13 [==============================] - 0s 869us/step - loss: 0.1843
    Epoch 384/1000
    13/13 [==============================] - 0s 861us/step - loss: 0.1763
    Epoch 385/1000
    13/13 [==============================] - 0s 852us/step - loss: 0.1790
    Epoch 386/1000
    13/13 [==============================] - 0s 1000us/step - loss: 0.1781
    Epoch 387/1000
    13/13 [==============================] - 0s 957us/step - loss: 0.1771
    Epoch 388/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1809
    Epoch 389/1000
    13/13 [==============================] - 0s 848us/step - loss: 0.1807
    Epoch 390/1000
    13/13 [==============================] - 0s 804us/step - loss: 0.1792
    Epoch 391/1000
    13/13 [==============================] - 0s 804us/step - loss: 0.1767
    Epoch 392/1000
    13/13 [==============================] - 0s 825us/step - loss: 0.1767
    Epoch 393/1000
    13/13 [==============================] - 0s 827us/step - loss: 0.1763
    Epoch 394/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1768
    Epoch 395/1000
    13/13 [==============================] - 0s 977us/step - loss: 0.1789
    Epoch 396/1000
    13/13 [==============================] - 0s 989us/step - loss: 0.1801
    Epoch 397/1000
    13/13 [==============================] - 0s 941us/step - loss: 0.1805
    Epoch 398/1000
    13/13 [==============================] - 0s 911us/step - loss: 0.1783
    Epoch 399/1000
    13/13 [==============================] - 0s 897us/step - loss: 0.1775
    Epoch 400/1000
    13/13 [==============================] - 0s 902us/step - loss: 0.1796
    Epoch 401/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1776
    Epoch 402/1000
    13/13 [==============================] - 0s 992us/step - loss: 0.1771
    Epoch 403/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1765
    Epoch 404/1000
    13/13 [==============================] - 0s 923us/step - loss: 0.1775
    Epoch 405/1000
    13/13 [==============================] - 0s 917us/step - loss: 0.1753
    Epoch 406/1000
    13/13 [==============================] - 0s 925us/step - loss: 0.1759
    Epoch 407/1000
    13/13 [==============================] - 0s 944us/step - loss: 0.1776
    Epoch 408/1000
    13/13 [==============================] - 0s 976us/step - loss: 0.1779
    Epoch 409/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1759
    Epoch 410/1000
    13/13 [==============================] - 0s 971us/step - loss: 0.1798
    Epoch 411/1000
    13/13 [==============================] - 0s 854us/step - loss: 0.1807
    Epoch 412/1000
    13/13 [==============================] - 0s 847us/step - loss: 0.1778
    Epoch 413/1000
    13/13 [==============================] - 0s 920us/step - loss: 0.1771
    Epoch 414/1000
    13/13 [==============================] - 0s 958us/step - loss: 0.1760
    Epoch 415/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1760
    Epoch 416/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1782
    Epoch 417/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1756
    Epoch 418/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1762
    Epoch 419/1000
    13/13 [==============================] - 0s 955us/step - loss: 0.1756
    Epoch 420/1000
    13/13 [==============================] - 0s 979us/step - loss: 0.1773
    Epoch 421/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1761
    Epoch 422/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1753
    Epoch 423/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1777
    Epoch 424/1000
    13/13 [==============================] - 0s 937us/step - loss: 0.1754
    Epoch 425/1000
    13/13 [==============================] - 0s 912us/step - loss: 0.1779
    Epoch 426/1000
    13/13 [==============================] - 0s 999us/step - loss: 0.1781
    Epoch 427/1000
    13/13 [==============================] - 0s 973us/step - loss: 0.1739
    Epoch 428/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1757
    Epoch 429/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1755
    Epoch 430/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1775
    Epoch 431/1000
    13/13 [==============================] - 0s 873us/step - loss: 0.1775
    Epoch 432/1000
    13/13 [==============================] - 0s 891us/step - loss: 0.1773
    Epoch 433/1000
    13/13 [==============================] - 0s 908us/step - loss: 0.1777
    Epoch 434/1000
    13/13 [==============================] - 0s 871us/step - loss: 0.1781
    Epoch 435/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1761
    Epoch 436/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1775
    Epoch 437/1000
    13/13 [==============================] - 0s 984us/step - loss: 0.1788
    Epoch 438/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1762
    Epoch 439/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1752
    Epoch 440/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1742
    Epoch 441/1000
    13/13 [==============================] - 0s 954us/step - loss: 0.1765
    Epoch 442/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1776
    Epoch 443/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1755
    Epoch 444/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1773
    Epoch 445/1000
    13/13 [==============================] - 0s 871us/step - loss: 0.1763
    Epoch 446/1000
    13/13 [==============================] - 0s 893us/step - loss: 0.1764
    Epoch 447/1000
    13/13 [==============================] - 0s 924us/step - loss: 0.1792
    Epoch 448/1000
    13/13 [==============================] - 0s 961us/step - loss: 0.1746
    Epoch 449/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1752
    Epoch 450/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1773
    Epoch 451/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1772
    Epoch 452/1000
    13/13 [==============================] - 0s 969us/step - loss: 0.1764
    Epoch 453/1000
    13/13 [==============================] - 0s 931us/step - loss: 0.1754
    Epoch 454/1000
    13/13 [==============================] - 0s 929us/step - loss: 0.1748
    Epoch 455/1000
    13/13 [==============================] - 0s 944us/step - loss: 0.1752
    Epoch 456/1000
    13/13 [==============================] - 0s 948us/step - loss: 0.1753
    Epoch 457/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1785
    Epoch 458/1000
    13/13 [==============================] - 0s 957us/step - loss: 0.1744
    Epoch 459/1000
    13/13 [==============================] - 0s 895us/step - loss: 0.1758
    Epoch 460/1000
    13/13 [==============================] - 0s 895us/step - loss: 0.1759
    Epoch 461/1000
    13/13 [==============================] - 0s 918us/step - loss: 0.1750
    Epoch 462/1000
    13/13 [==============================] - 0s 883us/step - loss: 0.1745
    Epoch 463/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1792
    Epoch 464/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1752
    Epoch 465/1000
    13/13 [==============================] - 0s 853us/step - loss: 0.1756
    Epoch 466/1000
    13/13 [==============================] - 0s 883us/step - loss: 0.1752
    Epoch 467/1000
    13/13 [==============================] - 0s 892us/step - loss: 0.1774
    Epoch 468/1000
    13/13 [==============================] - 0s 900us/step - loss: 0.1748
    Epoch 469/1000
    13/13 [==============================] - 0s 887us/step - loss: 0.1767
    Epoch 470/1000
    13/13 [==============================] - 0s 931us/step - loss: 0.1813
    Epoch 471/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1793
    Epoch 472/1000
    13/13 [==============================] - 0s 961us/step - loss: 0.1748
    Epoch 473/1000
    13/13 [==============================] - 0s 862us/step - loss: 0.1762
    Epoch 474/1000
    13/13 [==============================] - 0s 868us/step - loss: 0.1822
    Epoch 475/1000
    13/13 [==============================] - 0s 825us/step - loss: 0.1788
    Epoch 476/1000
    13/13 [==============================] - 0s 870us/step - loss: 0.1760
    Epoch 477/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1758
    Epoch 478/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1763
    Epoch 479/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1751
    Epoch 480/1000
    13/13 [==============================] - 0s 886us/step - loss: 0.1749
    Epoch 481/1000
    13/13 [==============================] - 0s 879us/step - loss: 0.1742
    Epoch 482/1000
    13/13 [==============================] - 0s 882us/step - loss: 0.1745
    Epoch 483/1000
    13/13 [==============================] - 0s 898us/step - loss: 0.1763
    Epoch 484/1000
    13/13 [==============================] - 0s 956us/step - loss: 0.1767
    Epoch 485/1000
    13/13 [==============================] - 0s 937us/step - loss: 0.1780
    Epoch 486/1000
    13/13 [==============================] - 0s 948us/step - loss: 0.1739
    Epoch 487/1000
    13/13 [==============================] - 0s 869us/step - loss: 0.1781
    Epoch 488/1000
    13/13 [==============================] - 0s 861us/step - loss: 0.1755
    Epoch 489/1000
    13/13 [==============================] - 0s 834us/step - loss: 0.1766
    Epoch 490/1000
    13/13 [==============================] - 0s 927us/step - loss: 0.1783
    Epoch 491/1000
    13/13 [==============================] - 0s 931us/step - loss: 0.1769
    Epoch 492/1000
    13/13 [==============================] - 0s 960us/step - loss: 0.1752
    Epoch 493/1000
    13/13 [==============================] - 0s 903us/step - loss: 0.1772
    Epoch 494/1000
    13/13 [==============================] - 0s 864us/step - loss: 0.1739
    Epoch 495/1000
    13/13 [==============================] - 0s 847us/step - loss: 0.1750
    Epoch 496/1000
    13/13 [==============================] - 0s 869us/step - loss: 0.1798
    Epoch 497/1000
    13/13 [==============================] - 0s 834us/step - loss: 0.1744
    Epoch 498/1000
    13/13 [==============================] - 0s 813us/step - loss: 0.1750
    Epoch 499/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1750
    Epoch 500/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1735
    Epoch 501/1000
    13/13 [==============================] - 0s 890us/step - loss: 0.1783
    Epoch 502/1000
    13/13 [==============================] - 0s 847us/step - loss: 0.1749
    Epoch 503/1000
    13/13 [==============================] - 0s 833us/step - loss: 0.1749
    Epoch 504/1000
    13/13 [==============================] - 0s 801us/step - loss: 0.1741
    Epoch 505/1000
    13/13 [==============================] - 0s 856us/step - loss: 0.1767
    Epoch 506/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1752
    Epoch 507/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1764
    Epoch 508/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1719
    Epoch 509/1000
    13/13 [==============================] - 0s 837us/step - loss: 0.1791
    Epoch 510/1000
    13/13 [==============================] - 0s 845us/step - loss: 0.1746
    Epoch 511/1000
    13/13 [==============================] - 0s 825us/step - loss: 0.1786
    Epoch 512/1000
    13/13 [==============================] - 0s 873us/step - loss: 0.1737
    Epoch 513/1000
    13/13 [==============================] - 0s 876us/step - loss: 0.1781
    Epoch 514/1000
    13/13 [==============================] - 0s 914us/step - loss: 0.1766
    Epoch 515/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1730
    Epoch 516/1000
    13/13 [==============================] - 0s 875us/step - loss: 0.1738
    Epoch 517/1000
    13/13 [==============================] - 0s 816us/step - loss: 0.1729
    Epoch 518/1000
    13/13 [==============================] - 0s 845us/step - loss: 0.1747
    Epoch 519/1000
    13/13 [==============================] - 0s 825us/step - loss: 0.1759
    Epoch 520/1000
    13/13 [==============================] - 0s 929us/step - loss: 0.1748
    Epoch 521/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1762
    Epoch 522/1000
    13/13 [==============================] - 0s 932us/step - loss: 0.1750
    Epoch 523/1000
    13/13 [==============================] - 0s 922us/step - loss: 0.1751
    Epoch 524/1000
    13/13 [==============================] - 0s 879us/step - loss: 0.1747
    Epoch 525/1000
    13/13 [==============================] - 0s 916us/step - loss: 0.1739
    Epoch 526/1000
    13/13 [==============================] - 0s 891us/step - loss: 0.1731
    Epoch 527/1000
    13/13 [==============================] - 0s 936us/step - loss: 0.1783
    Epoch 528/1000
    13/13 [==============================] - 0s 969us/step - loss: 0.1810
    Epoch 529/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1770
    Epoch 530/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1740
    Epoch 531/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1743
    Epoch 532/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1759
    Epoch 533/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1786
    Epoch 534/1000
    13/13 [==============================] - 0s 923us/step - loss: 0.1766
    Epoch 535/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1755
    Epoch 536/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1749
    Epoch 537/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1713
    Epoch 538/1000
    13/13 [==============================] - 0s 830us/step - loss: 0.1774
    Epoch 539/1000
    13/13 [==============================] - 0s 831us/step - loss: 0.1741
    Epoch 540/1000
    13/13 [==============================] - 0s 869us/step - loss: 0.1774
    Epoch 541/1000
    13/13 [==============================] - 0s 990us/step - loss: 0.1734
    Epoch 542/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1754
    Epoch 543/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1735
    Epoch 544/1000
    13/13 [==============================] - 0s 833us/step - loss: 0.1758
    Epoch 545/1000
    13/13 [==============================] - 0s 853us/step - loss: 0.1723
    Epoch 546/1000
    13/13 [==============================] - 0s 937us/step - loss: 0.1786
    Epoch 547/1000
    13/13 [==============================] - 0s 845us/step - loss: 0.1743
    Epoch 548/1000
    13/13 [==============================] - 0s 918us/step - loss: 0.1750
    Epoch 549/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1747
    Epoch 550/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1768
    Epoch 551/1000
    13/13 [==============================] - 0s 840us/step - loss: 0.1732
    Epoch 552/1000
    13/13 [==============================] - 0s 820us/step - loss: 0.1736
    Epoch 553/1000
    13/13 [==============================] - 0s 810us/step - loss: 0.1725
    Epoch 554/1000
    13/13 [==============================] - 0s 827us/step - loss: 0.1748
    Epoch 555/1000
    13/13 [==============================] - 0s 803us/step - loss: 0.1733
    Epoch 556/1000
    13/13 [==============================] - 0s 900us/step - loss: 0.1727
    Epoch 557/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1754
    Epoch 558/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1781
    Epoch 559/1000
    13/13 [==============================] - 0s 848us/step - loss: 0.1805
    Epoch 560/1000
    13/13 [==============================] - 0s 884us/step - loss: 0.1764
    Epoch 561/1000
    13/13 [==============================] - 0s 849us/step - loss: 0.1784
    Epoch 562/1000
    13/13 [==============================] - 0s 902us/step - loss: 0.1715
    Epoch 563/1000
    13/13 [==============================] - 0s 958us/step - loss: 0.1730
    Epoch 564/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1733
    Epoch 565/1000
    13/13 [==============================] - 0s 978us/step - loss: 0.1718
    Epoch 566/1000
    13/13 [==============================] - 0s 902us/step - loss: 0.1750
    Epoch 567/1000
    13/13 [==============================] - 0s 885us/step - loss: 0.1751
    Epoch 568/1000
    13/13 [==============================] - 0s 832us/step - loss: 0.1728
    Epoch 569/1000
    13/13 [==============================] - 0s 846us/step - loss: 0.1730
    Epoch 570/1000
    13/13 [==============================] - 0s 941us/step - loss: 0.1761
    Epoch 571/1000
    13/13 [==============================] - 0s 938us/step - loss: 0.1798
    Epoch 572/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1762
    Epoch 573/1000
    13/13 [==============================] - 0s 868us/step - loss: 0.1727
    Epoch 574/1000
    13/13 [==============================] - 0s 859us/step - loss: 0.1722
    Epoch 575/1000
    13/13 [==============================] - 0s 841us/step - loss: 0.1717
    Epoch 576/1000
    13/13 [==============================] - 0s 825us/step - loss: 0.1730
    Epoch 577/1000
    13/13 [==============================] - 0s 812us/step - loss: 0.1751
    Epoch 578/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1741
    Epoch 579/1000
    13/13 [==============================] - 0s 941us/step - loss: 0.1732
    Epoch 580/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1725
    Epoch 581/1000
    13/13 [==============================] - 0s 854us/step - loss: 0.1731
    Epoch 582/1000
    13/13 [==============================] - 0s 829us/step - loss: 0.1709
    Epoch 583/1000
    13/13 [==============================] - 0s 809us/step - loss: 0.1727
    Epoch 584/1000
    13/13 [==============================] - 0s 819us/step - loss: 0.1742
    Epoch 585/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1721
    Epoch 586/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1730
    Epoch 587/1000
    13/13 [==============================] - 0s 929us/step - loss: 0.1728
    Epoch 588/1000
    13/13 [==============================] - 0s 899us/step - loss: 0.1718
    Epoch 589/1000
    13/13 [==============================] - 0s 859us/step - loss: 0.1710
    Epoch 590/1000
    13/13 [==============================] - 0s 812us/step - loss: 0.1787
    Epoch 591/1000
    13/13 [==============================] - 0s 808us/step - loss: 0.1789
    Epoch 592/1000
    13/13 [==============================] - 0s 901us/step - loss: 0.1745
    Epoch 593/1000
    13/13 [==============================] - 0s 999us/step - loss: 0.1775
    Epoch 594/1000
    13/13 [==============================] - 0s 953us/step - loss: 0.1727
    Epoch 595/1000
    13/13 [==============================] - 0s 938us/step - loss: 0.1738
    Epoch 596/1000
    13/13 [==============================] - 0s 882us/step - loss: 0.1746
    Epoch 597/1000
    13/13 [==============================] - 0s 854us/step - loss: 0.1734
    Epoch 598/1000
    13/13 [==============================] - 0s 843us/step - loss: 0.1738
    Epoch 599/1000
    13/13 [==============================] - 0s 832us/step - loss: 0.1707
    Epoch 600/1000
    13/13 [==============================] - 0s 911us/step - loss: 0.1735
    Epoch 601/1000
    13/13 [==============================] - 0s 909us/step - loss: 0.1731
    Epoch 602/1000
    13/13 [==============================] - 0s 950us/step - loss: 0.1727
    Epoch 603/1000
    13/13 [==============================] - 0s 874us/step - loss: 0.1722
    Epoch 604/1000
    13/13 [==============================] - 0s 838us/step - loss: 0.1720
    Epoch 605/1000
    13/13 [==============================] - 0s 846us/step - loss: 0.1747
    Epoch 606/1000
    13/13 [==============================] - 0s 825us/step - loss: 0.1770
    Epoch 607/1000
    13/13 [==============================] - 0s 833us/step - loss: 0.1741
    Epoch 608/1000
    13/13 [==============================] - 0s 901us/step - loss: 0.1748
    Epoch 609/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1731
    Epoch 610/1000
    13/13 [==============================] - 0s 927us/step - loss: 0.1743
    Epoch 611/1000
    13/13 [==============================] - 0s 837us/step - loss: 0.1725
    Epoch 612/1000
    13/13 [==============================] - 0s 809us/step - loss: 0.1706
    Epoch 613/1000
    13/13 [==============================] - 0s 800us/step - loss: 0.1732
    Epoch 614/1000
    13/13 [==============================] - 0s 816us/step - loss: 0.1746
    Epoch 615/1000
    13/13 [==============================] - 0s 853us/step - loss: 0.1729
    Epoch 616/1000
    13/13 [==============================] - 0s 923us/step - loss: 0.1711
    Epoch 617/1000
    13/13 [==============================] - 0s 947us/step - loss: 0.1722
    Epoch 618/1000
    13/13 [==============================] - 0s 935us/step - loss: 0.1802
    Epoch 619/1000
    13/13 [==============================] - 0s 889us/step - loss: 0.1725
    Epoch 620/1000
    13/13 [==============================] - 0s 850us/step - loss: 0.1773
    Epoch 621/1000
    13/13 [==============================] - 0s 845us/step - loss: 0.1710
    Epoch 622/1000
    13/13 [==============================] - 0s 886us/step - loss: 0.1746
    Epoch 623/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1728
    Epoch 624/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1709
    Epoch 625/1000
    13/13 [==============================] - 0s 943us/step - loss: 0.1776
    Epoch 626/1000
    13/13 [==============================] - 0s 817us/step - loss: 0.1717
    Epoch 627/1000
    13/13 [==============================] - 0s 806us/step - loss: 0.1728
    Epoch 628/1000
    13/13 [==============================] - 0s 809us/step - loss: 0.1711
    Epoch 629/1000
    13/13 [==============================] - 0s 805us/step - loss: 0.1732
    Epoch 630/1000
    13/13 [==============================] - 0s 829us/step - loss: 0.1719
    Epoch 631/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1711
    Epoch 632/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1752
    Epoch 633/1000
    13/13 [==============================] - 0s 853us/step - loss: 0.1731
    Epoch 634/1000
    13/13 [==============================] - 0s 818us/step - loss: 0.1758
    Epoch 635/1000
    13/13 [==============================] - 0s 804us/step - loss: 0.1713
    Epoch 636/1000
    13/13 [==============================] - 0s 800us/step - loss: 0.1744
    Epoch 637/1000
    13/13 [==============================] - 0s 800us/step - loss: 0.1728
    Epoch 638/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1725
    Epoch 639/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1718
    Epoch 640/1000
    13/13 [==============================] - 0s 909us/step - loss: 0.1732
    Epoch 641/1000
    13/13 [==============================] - 0s 839us/step - loss: 0.1736
    Epoch 642/1000
    13/13 [==============================] - 0s 800us/step - loss: 0.1700
    Epoch 643/1000
    13/13 [==============================] - 0s 789us/step - loss: 0.1705
    Epoch 644/1000
    13/13 [==============================] - 0s 790us/step - loss: 0.1725
    Epoch 645/1000
    13/13 [==============================] - 0s 808us/step - loss: 0.1711
    Epoch 646/1000
    13/13 [==============================] - 0s 964us/step - loss: 0.1723
    Epoch 647/1000
    13/13 [==============================] - 0s 905us/step - loss: 0.1719
    Epoch 648/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1718
    Epoch 649/1000
    13/13 [==============================] - 0s 899us/step - loss: 0.1740
    Epoch 650/1000
    13/13 [==============================] - 0s 870us/step - loss: 0.1737
    Epoch 651/1000
    13/13 [==============================] - 0s 825us/step - loss: 0.1705
    Epoch 652/1000
    13/13 [==============================] - 0s 817us/step - loss: 0.1699
    Epoch 653/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1712
    Epoch 654/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1704
    Epoch 655/1000
    13/13 [==============================] - 0s 909us/step - loss: 0.1705
    Epoch 656/1000
    13/13 [==============================] - 0s 796us/step - loss: 0.1701
    Epoch 657/1000
    13/13 [==============================] - 0s 797us/step - loss: 0.1701
    Epoch 658/1000
    13/13 [==============================] - 0s 799us/step - loss: 0.1739
    Epoch 659/1000
    13/13 [==============================] - 0s 802us/step - loss: 0.1712
    Epoch 660/1000
    13/13 [==============================] - 0s 789us/step - loss: 0.1697
    Epoch 661/1000
    13/13 [==============================] - 0s 930us/step - loss: 0.1718
    Epoch 662/1000
    13/13 [==============================] - 0s 944us/step - loss: 0.1720
    Epoch 663/1000
    13/13 [==============================] - 0s 895us/step - loss: 0.1725
    Epoch 664/1000
    13/13 [==============================] - 0s 848us/step - loss: 0.1694
    Epoch 665/1000
    13/13 [==============================] - 0s 925us/step - loss: 0.1700
    Epoch 666/1000
    13/13 [==============================] - 0s 898us/step - loss: 0.1740
    Epoch 667/1000
    13/13 [==============================] - 0s 899us/step - loss: 0.1693
    Epoch 668/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1722
    Epoch 669/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1732
    Epoch 670/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1704
    Epoch 671/1000
    13/13 [==============================] - 0s 817us/step - loss: 0.1696
    Epoch 672/1000
    13/13 [==============================] - 0s 815us/step - loss: 0.1733
    Epoch 673/1000
    13/13 [==============================] - 0s 820us/step - loss: 0.1726
    Epoch 674/1000
    13/13 [==============================] - 0s 792us/step - loss: 0.1740
    Epoch 675/1000
    13/13 [==============================] - 0s 897us/step - loss: 0.1699
    Epoch 676/1000
    13/13 [==============================] - 0s 901us/step - loss: 0.1712
    Epoch 677/1000
    13/13 [==============================] - 0s 875us/step - loss: 0.1711
    Epoch 678/1000
    13/13 [==============================] - 0s 856us/step - loss: 0.1718
    Epoch 679/1000
    13/13 [==============================] - 0s 875us/step - loss: 0.1795
    Epoch 680/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1709
    Epoch 681/1000
    13/13 [==============================] - 0s 965us/step - loss: 0.1703
    Epoch 682/1000
    13/13 [==============================] - 0s 881us/step - loss: 0.1717
    Epoch 683/1000
    13/13 [==============================] - 0s 981us/step - loss: 0.1758
    Epoch 684/1000
    13/13 [==============================] - 0s 930us/step - loss: 0.1699
    Epoch 685/1000
    13/13 [==============================] - 0s 913us/step - loss: 0.1753
    Epoch 686/1000
    13/13 [==============================] - 0s 839us/step - loss: 0.1728
    Epoch 687/1000
    13/13 [==============================] - 0s 856us/step - loss: 0.1733
    Epoch 688/1000
    13/13 [==============================] - 0s 837us/step - loss: 0.1706
    Epoch 689/1000
    13/13 [==============================] - 0s 839us/step - loss: 0.1705
    Epoch 690/1000
    13/13 [==============================] - 0s 852us/step - loss: 0.1698
    Epoch 691/1000
    13/13 [==============================] - 0s 914us/step - loss: 0.1721
    Epoch 692/1000
    13/13 [==============================] - 0s 953us/step - loss: 0.1712
    Epoch 693/1000
    13/13 [==============================] - 0s 953us/step - loss: 0.1716
    Epoch 694/1000
    13/13 [==============================] - 0s 884us/step - loss: 0.1692
    Epoch 695/1000
    13/13 [==============================] - 0s 821us/step - loss: 0.1718
    Epoch 696/1000
    13/13 [==============================] - 0s 818us/step - loss: 0.1704
    Epoch 697/1000
    13/13 [==============================] - 0s 817us/step - loss: 0.1711
    Epoch 698/1000
    13/13 [==============================] - 0s 849us/step - loss: 0.1708
    Epoch 699/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1702
    Epoch 700/1000
    13/13 [==============================] - 0s 921us/step - loss: 0.1737
    Epoch 701/1000
    13/13 [==============================] - 0s 918us/step - loss: 0.1720
    Epoch 702/1000
    13/13 [==============================] - 0s 843us/step - loss: 0.1701
    Epoch 703/1000
    13/13 [==============================] - 0s 825us/step - loss: 0.1710
    Epoch 704/1000
    13/13 [==============================] - 0s 830us/step - loss: 0.1690
    Epoch 705/1000
    13/13 [==============================] - 0s 851us/step - loss: 0.1719
    Epoch 706/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1718
    Epoch 707/1000
    13/13 [==============================] - 0s 927us/step - loss: 0.1680
    Epoch 708/1000
    13/13 [==============================] - 0s 870us/step - loss: 0.1756
    Epoch 709/1000
    13/13 [==============================] - 0s 818us/step - loss: 0.1754
    Epoch 710/1000
    13/13 [==============================] - 0s 811us/step - loss: 0.1721
    Epoch 711/1000
    13/13 [==============================] - 0s 823us/step - loss: 0.1751
    Epoch 712/1000
    13/13 [==============================] - 0s 792us/step - loss: 0.1714
    Epoch 713/1000
    13/13 [==============================] - 0s 871us/step - loss: 0.1716
    Epoch 714/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1703
    Epoch 715/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1704
    Epoch 716/1000
    13/13 [==============================] - 0s 839us/step - loss: 0.1749
    Epoch 717/1000
    13/13 [==============================] - 0s 836us/step - loss: 0.1676
    Epoch 718/1000
    13/13 [==============================] - 0s 818us/step - loss: 0.1713
    Epoch 719/1000
    13/13 [==============================] - 0s 828us/step - loss: 0.1690
    Epoch 720/1000
    13/13 [==============================] - 0s 895us/step - loss: 0.1700
    Epoch 721/1000
    13/13 [==============================] - 0s 902us/step - loss: 0.1713
    Epoch 722/1000
    13/13 [==============================] - 0s 967us/step - loss: 0.1712
    Epoch 723/1000
    13/13 [==============================] - 0s 861us/step - loss: 0.1697
    Epoch 724/1000
    13/13 [==============================] - 0s 829us/step - loss: 0.1718
    Epoch 725/1000
    13/13 [==============================] - 0s 822us/step - loss: 0.1741
    Epoch 726/1000
    13/13 [==============================] - 0s 844us/step - loss: 0.1719
    Epoch 727/1000
    13/13 [==============================] - 0s 834us/step - loss: 0.1716
    Epoch 728/1000
    13/13 [==============================] - 0s 858us/step - loss: 0.1713
    Epoch 729/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1694
    Epoch 730/1000
    13/13 [==============================] - 0s 917us/step - loss: 0.1764
    Epoch 731/1000
    13/13 [==============================] - 0s 849us/step - loss: 0.1758
    Epoch 732/1000
    13/13 [==============================] - 0s 836us/step - loss: 0.1735
    Epoch 733/1000
    13/13 [==============================] - 0s 823us/step - loss: 0.1700
    Epoch 734/1000
    13/13 [==============================] - 0s 937us/step - loss: 0.1698
    Epoch 735/1000
    13/13 [==============================] - 0s 914us/step - loss: 0.1699
    Epoch 736/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1716
    Epoch 737/1000
    13/13 [==============================] - 0s 912us/step - loss: 0.1701
    Epoch 738/1000
    13/13 [==============================] - 0s 863us/step - loss: 0.1720
    Epoch 739/1000
    13/13 [==============================] - 0s 838us/step - loss: 0.1737
    Epoch 740/1000
    13/13 [==============================] - 0s 816us/step - loss: 0.1730
    Epoch 741/1000
    13/13 [==============================] - 0s 819us/step - loss: 0.1700
    Epoch 742/1000
    13/13 [==============================] - 0s 814us/step - loss: 0.1684
    Epoch 743/1000
    13/13 [==============================] - 0s 925us/step - loss: 0.1713
    Epoch 744/1000
    13/13 [==============================] - 0s 938us/step - loss: 0.1695
    Epoch 745/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1715
    Epoch 746/1000
    13/13 [==============================] - 0s 831us/step - loss: 0.1690
    Epoch 747/1000
    13/13 [==============================] - 0s 813us/step - loss: 0.1706
    Epoch 748/1000
    13/13 [==============================] - 0s 810us/step - loss: 0.1687
    Epoch 749/1000
    13/13 [==============================] - 0s 797us/step - loss: 0.1694
    Epoch 750/1000
    13/13 [==============================] - 0s 798us/step - loss: 0.1700
    Epoch 751/1000
    13/13 [==============================] - 0s 892us/step - loss: 0.1697
    Epoch 752/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1696
    Epoch 753/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1707
    Epoch 754/1000
    13/13 [==============================] - 0s 836us/step - loss: 0.1719
    Epoch 755/1000
    13/13 [==============================] - 0s 836us/step - loss: 0.1716
    Epoch 756/1000
    13/13 [==============================] - 0s 840us/step - loss: 0.1766
    Epoch 757/1000
    13/13 [==============================] - 0s 834us/step - loss: 0.1752
    Epoch 758/1000
    13/13 [==============================] - 0s 900us/step - loss: 0.1689
    Epoch 759/1000
    13/13 [==============================] - 0s 887us/step - loss: 0.1709
    Epoch 760/1000
    13/13 [==============================] - 0s 891us/step - loss: 0.1696
    Epoch 761/1000
    13/13 [==============================] - 0s 846us/step - loss: 0.1684
    Epoch 762/1000
    13/13 [==============================] - 0s 795us/step - loss: 0.1731
    Epoch 763/1000
    13/13 [==============================] - 0s 804us/step - loss: 0.1725
    Epoch 764/1000
    13/13 [==============================] - 0s 822us/step - loss: 0.1754
    Epoch 765/1000
    13/13 [==============================] - 0s 880us/step - loss: 0.1697
    Epoch 766/1000
    13/13 [==============================] - 0s 905us/step - loss: 0.1735
    Epoch 767/1000
    13/13 [==============================] - 0s 930us/step - loss: 0.1705
    Epoch 768/1000
    13/13 [==============================] - 0s 993us/step - loss: 0.1699
    Epoch 769/1000
    13/13 [==============================] - 0s 927us/step - loss: 0.1701
    Epoch 770/1000
    13/13 [==============================] - 0s 848us/step - loss: 0.1693
    Epoch 771/1000
    13/13 [==============================] - 0s 897us/step - loss: 0.1708
    Epoch 772/1000
    13/13 [==============================] - 0s 843us/step - loss: 0.1693
    Epoch 773/1000
    13/13 [==============================] - 0s 850us/step - loss: 0.1697
    Epoch 774/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1712
    Epoch 775/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1704
    Epoch 776/1000
    13/13 [==============================] - 0s 941us/step - loss: 0.1681
    Epoch 777/1000
    13/13 [==============================] - 0s 852us/step - loss: 0.1704
    Epoch 778/1000
    13/13 [==============================] - 0s 842us/step - loss: 0.1721
    Epoch 779/1000
    13/13 [==============================] - 0s 824us/step - loss: 0.1706
    Epoch 780/1000
    13/13 [==============================] - 0s 806us/step - loss: 0.1747
    Epoch 781/1000
    13/13 [==============================] - 0s 885us/step - loss: 0.1722
    Epoch 782/1000
    13/13 [==============================] - 0s 994us/step - loss: 0.1714
    Epoch 783/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1697
    Epoch 784/1000
    13/13 [==============================] - 0s 805us/step - loss: 0.1691
    Epoch 785/1000
    13/13 [==============================] - 0s 792us/step - loss: 0.1710
    Epoch 786/1000
    13/13 [==============================] - 0s 789us/step - loss: 0.1770
    Epoch 787/1000
    13/13 [==============================] - 0s 792us/step - loss: 0.1710
    Epoch 788/1000
    13/13 [==============================] - 0s 794us/step - loss: 0.1672
    Epoch 789/1000
    13/13 [==============================] - 0s 914us/step - loss: 0.1706
    Epoch 790/1000
    13/13 [==============================] - 0s 947us/step - loss: 0.1718
    Epoch 791/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1678
    Epoch 792/1000
    13/13 [==============================] - 0s 820us/step - loss: 0.1691
    Epoch 793/1000
    13/13 [==============================] - 0s 811us/step - loss: 0.1715
    Epoch 794/1000
    13/13 [==============================] - 0s 796us/step - loss: 0.1784
    Epoch 795/1000
    13/13 [==============================] - 0s 798us/step - loss: 0.1659
    Epoch 796/1000
    13/13 [==============================] - 0s 786us/step - loss: 0.1756
    Epoch 797/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1708
    Epoch 798/1000
    13/13 [==============================] - 0s 980us/step - loss: 0.1706
    Epoch 799/1000
    13/13 [==============================] - 0s 959us/step - loss: 0.1695
    Epoch 800/1000
    13/13 [==============================] - 0s 907us/step - loss: 0.1668
    Epoch 801/1000
    13/13 [==============================] - 0s 834us/step - loss: 0.1703
    Epoch 802/1000
    13/13 [==============================] - 0s 878us/step - loss: 0.1683
    Epoch 803/1000
    13/13 [==============================] - 0s 895us/step - loss: 0.1704
    Epoch 804/1000
    13/13 [==============================] - 0s 904us/step - loss: 0.1701
    Epoch 805/1000
    13/13 [==============================] - 0s 935us/step - loss: 0.1691
    Epoch 806/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1712
    Epoch 807/1000
    13/13 [==============================] - 0s 865us/step - loss: 0.1679
    Epoch 808/1000
    13/13 [==============================] - 0s 828us/step - loss: 0.1688
    Epoch 809/1000
    13/13 [==============================] - 0s 816us/step - loss: 0.1704
    Epoch 810/1000
    13/13 [==============================] - 0s 815us/step - loss: 0.1699
    Epoch 811/1000
    13/13 [==============================] - 0s 816us/step - loss: 0.1693
    Epoch 812/1000
    13/13 [==============================] - 0s 893us/step - loss: 0.1678
    Epoch 813/1000
    13/13 [==============================] - 0s 943us/step - loss: 0.1694
    Epoch 814/1000
    13/13 [==============================] - 0s 962us/step - loss: 0.1676
    Epoch 815/1000
    13/13 [==============================] - 0s 881us/step - loss: 0.1698
    Epoch 816/1000
    13/13 [==============================] - 0s 822us/step - loss: 0.1717
    Epoch 817/1000
    13/13 [==============================] - 0s 813us/step - loss: 0.1712
    Epoch 818/1000
    13/13 [==============================] - 0s 808us/step - loss: 0.1681
    Epoch 819/1000
    13/13 [==============================] - 0s 791us/step - loss: 0.1723
    Epoch 820/1000
    13/13 [==============================] - 0s 904us/step - loss: 0.1733
    Epoch 821/1000
    13/13 [==============================] - 0s 899us/step - loss: 0.1692
    Epoch 822/1000
    13/13 [==============================] - 0s 869us/step - loss: 0.1745
    Epoch 823/1000
    13/13 [==============================] - 0s 804us/step - loss: 0.1762
    Epoch 824/1000
    13/13 [==============================] - 0s 799us/step - loss: 0.1713
    Epoch 825/1000
    13/13 [==============================] - 0s 797us/step - loss: 0.1697
    Epoch 826/1000
    13/13 [==============================] - 0s 794us/step - loss: 0.1698
    Epoch 827/1000
    13/13 [==============================] - 0s 803us/step - loss: 0.1720
    Epoch 828/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1696
    Epoch 829/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1707
    Epoch 830/1000
    13/13 [==============================] - 0s 899us/step - loss: 0.1693
    Epoch 831/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1691
    Epoch 832/1000
    13/13 [==============================] - 0s 832us/step - loss: 0.1689
    Epoch 833/1000
    13/13 [==============================] - 0s 815us/step - loss: 0.1716
    Epoch 834/1000
    13/13 [==============================] - 0s 801us/step - loss: 0.1669
    Epoch 835/1000
    13/13 [==============================] - 0s 860us/step - loss: 0.1683
    Epoch 836/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1673
    Epoch 837/1000
    13/13 [==============================] - 0s 941us/step - loss: 0.1684
    Epoch 838/1000
    13/13 [==============================] - 0s 876us/step - loss: 0.1688
    Epoch 839/1000
    13/13 [==============================] - 0s 814us/step - loss: 0.1695
    Epoch 840/1000
    13/13 [==============================] - 0s 807us/step - loss: 0.1689
    Epoch 841/1000
    13/13 [==============================] - 0s 798us/step - loss: 0.1702
    Epoch 842/1000
    13/13 [==============================] - 0s 818us/step - loss: 0.1711
    Epoch 843/1000
    13/13 [==============================] - 0s 947us/step - loss: 0.1689
    Epoch 844/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1682
    Epoch 845/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1694
    Epoch 846/1000
    13/13 [==============================] - 0s 816us/step - loss: 0.1678
    Epoch 847/1000
    13/13 [==============================] - 0s 819us/step - loss: 0.1693
    Epoch 848/1000
    13/13 [==============================] - 0s 856us/step - loss: 0.1707
    Epoch 849/1000
    13/13 [==============================] - 0s 876us/step - loss: 0.1699
    Epoch 850/1000
    13/13 [==============================] - 0s 827us/step - loss: 0.1683
    Epoch 851/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1688
    Epoch 852/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1751
    Epoch 853/1000
    13/13 [==============================] - 0s 847us/step - loss: 0.1707
    Epoch 854/1000
    13/13 [==============================] - 0s 811us/step - loss: 0.1680
    Epoch 855/1000
    13/13 [==============================] - 0s 805us/step - loss: 0.1688
    Epoch 856/1000
    13/13 [==============================] - 0s 806us/step - loss: 0.1690
    Epoch 857/1000
    13/13 [==============================] - 0s 801us/step - loss: 0.1676
    Epoch 858/1000
    13/13 [==============================] - 0s 859us/step - loss: 0.1720
    Epoch 859/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1691
    Epoch 860/1000
    13/13 [==============================] - 0s 862us/step - loss: 0.1692
    Epoch 861/1000
    13/13 [==============================] - 0s 821us/step - loss: 0.1705
    Epoch 862/1000
    13/13 [==============================] - 0s 810us/step - loss: 0.1675
    Epoch 863/1000
    13/13 [==============================] - 0s 817us/step - loss: 0.1715
    Epoch 864/1000
    13/13 [==============================] - 0s 818us/step - loss: 0.1684
    Epoch 865/1000
    13/13 [==============================] - 0s 805us/step - loss: 0.1703
    Epoch 866/1000
    13/13 [==============================] - 0s 975us/step - loss: 0.1702
    Epoch 867/1000
    13/13 [==============================] - 0s 945us/step - loss: 0.1695
    Epoch 868/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1728
    Epoch 869/1000
    13/13 [==============================] - 0s 836us/step - loss: 0.1682
    Epoch 870/1000
    13/13 [==============================] - 0s 814us/step - loss: 0.1681
    Epoch 871/1000
    13/13 [==============================] - 0s 796us/step - loss: 0.1684
    Epoch 872/1000
    13/13 [==============================] - 0s 794us/step - loss: 0.1680
    Epoch 873/1000
    13/13 [==============================] - 0s 793us/step - loss: 0.1720
    Epoch 874/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1705
    Epoch 875/1000
    13/13 [==============================] - 0s 999us/step - loss: 0.1686
    Epoch 876/1000
    13/13 [==============================] - 0s 868us/step - loss: 0.1676
    Epoch 877/1000
    13/13 [==============================] - 0s 820us/step - loss: 0.1750
    Epoch 878/1000
    13/13 [==============================] - 0s 826us/step - loss: 0.1728
    Epoch 879/1000
    13/13 [==============================] - 0s 822us/step - loss: 0.1733
    Epoch 880/1000
    13/13 [==============================] - 0s 805us/step - loss: 0.1690
    Epoch 881/1000
    13/13 [==============================] - 0s 862us/step - loss: 0.1721
    Epoch 882/1000
    13/13 [==============================] - 0s 953us/step - loss: 0.1754
    Epoch 883/1000
    13/13 [==============================] - 0s 991us/step - loss: 0.1727
    Epoch 884/1000
    13/13 [==============================] - 0s 909us/step - loss: 0.1697
    Epoch 885/1000
    13/13 [==============================] - 0s 806us/step - loss: 0.1670
    Epoch 886/1000
    13/13 [==============================] - 0s 800us/step - loss: 0.1675
    Epoch 887/1000
    13/13 [==============================] - 0s 805us/step - loss: 0.1723
    Epoch 888/1000
    13/13 [==============================] - 0s 793us/step - loss: 0.1701
    Epoch 889/1000
    13/13 [==============================] - 0s 833us/step - loss: 0.1677
    Epoch 890/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1712
    Epoch 891/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1684
    Epoch 892/1000
    13/13 [==============================] - 0s 880us/step - loss: 0.1695
    Epoch 893/1000
    13/13 [==============================] - 0s 891us/step - loss: 0.1680
    Epoch 894/1000
    13/13 [==============================] - 0s 889us/step - loss: 0.1694
    Epoch 895/1000
    13/13 [==============================] - 0s 862us/step - loss: 0.1683
    Epoch 896/1000
    13/13 [==============================] - 0s 920us/step - loss: 0.1694
    Epoch 897/1000
    13/13 [==============================] - 0s 930us/step - loss: 0.1714
    Epoch 898/1000
    13/13 [==============================] - 0s 894us/step - loss: 0.1682
    Epoch 899/1000
    13/13 [==============================] - 0s 956us/step - loss: 0.1704
    Epoch 900/1000
    13/13 [==============================] - 0s 895us/step - loss: 0.1664
    Epoch 901/1000
    13/13 [==============================] - 0s 874us/step - loss: 0.1683
    Epoch 902/1000
    13/13 [==============================] - 0s 882us/step - loss: 0.1682
    Epoch 903/1000
    13/13 [==============================] - 0s 895us/step - loss: 0.1669
    Epoch 904/1000
    13/13 [==============================] - 0s 913us/step - loss: 0.1688
    Epoch 905/1000
    13/13 [==============================] - 0s 912us/step - loss: 0.1686
    Epoch 906/1000
    13/13 [==============================] - 0s 914us/step - loss: 0.1739
    Epoch 907/1000
    13/13 [==============================] - 0s 839us/step - loss: 0.1693
    Epoch 908/1000
    13/13 [==============================] - 0s 881us/step - loss: 0.1689
    Epoch 909/1000
    13/13 [==============================] - 0s 907us/step - loss: 0.1673
    Epoch 910/1000
    13/13 [==============================] - 0s 892us/step - loss: 0.1700
    Epoch 911/1000
    13/13 [==============================] - 0s 876us/step - loss: 0.1672
    Epoch 912/1000
    13/13 [==============================] - 0s 926us/step - loss: 0.1672
    Epoch 913/1000
    13/13 [==============================] - 0s 954us/step - loss: 0.1702
    Epoch 914/1000
    13/13 [==============================] - 0s 968us/step - loss: 0.1662
    Epoch 915/1000
    13/13 [==============================] - 0s 997us/step - loss: 0.1716
    Epoch 916/1000
    13/13 [==============================] - 0s 874us/step - loss: 0.1669
    Epoch 917/1000
    13/13 [==============================] - 0s 856us/step - loss: 0.1704
    Epoch 918/1000
    13/13 [==============================] - 0s 883us/step - loss: 0.1659
    Epoch 919/1000
    13/13 [==============================] - 0s 843us/step - loss: 0.1725
    Epoch 920/1000
    13/13 [==============================] - 0s 912us/step - loss: 0.1718
    Epoch 921/1000
    13/13 [==============================] - 0s 924us/step - loss: 0.1670
    Epoch 922/1000
    13/13 [==============================] - 0s 909us/step - loss: 0.1695
    Epoch 923/1000
    13/13 [==============================] - 0s 820us/step - loss: 0.1670
    Epoch 924/1000
    13/13 [==============================] - 0s 850us/step - loss: 0.1672
    Epoch 925/1000
    13/13 [==============================] - 0s 847us/step - loss: 0.1685
    Epoch 926/1000
    13/13 [==============================] - 0s 821us/step - loss: 0.1681
    Epoch 927/1000
    13/13 [==============================] - 0s 904us/step - loss: 0.1698
    Epoch 928/1000
    13/13 [==============================] - 0s 927us/step - loss: 0.1660
    Epoch 929/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1704
    Epoch 930/1000
    13/13 [==============================] - 0s 826us/step - loss: 0.1678
    Epoch 931/1000
    13/13 [==============================] - 0s 815us/step - loss: 0.1703
    Epoch 932/1000
    13/13 [==============================] - 0s 804us/step - loss: 0.1700
    Epoch 933/1000
    13/13 [==============================] - 0s 809us/step - loss: 0.1699
    Epoch 934/1000
    13/13 [==============================] - 0s 877us/step - loss: 0.1691
    Epoch 935/1000
    13/13 [==============================] - 0s 929us/step - loss: 0.1689
    Epoch 936/1000
    13/13 [==============================] - 0s 942us/step - loss: 0.1680
    Epoch 937/1000
    13/13 [==============================] - 0s 916us/step - loss: 0.1701
    Epoch 938/1000
    13/13 [==============================] - 0s 816us/step - loss: 0.1681
    Epoch 939/1000
    13/13 [==============================] - 0s 822us/step - loss: 0.1693
    Epoch 940/1000
    13/13 [==============================] - 0s 891us/step - loss: 0.1703
    Epoch 941/1000
    13/13 [==============================] - 0s 894us/step - loss: 0.1674
    Epoch 942/1000
    13/13 [==============================] - 0s 891us/step - loss: 0.1667
    Epoch 943/1000
    13/13 [==============================] - 0s 941us/step - loss: 0.1682
    Epoch 944/1000
    13/13 [==============================] - 0s 977us/step - loss: 0.1706
    Epoch 945/1000
    13/13 [==============================] - 0s 908us/step - loss: 0.1679
    Epoch 946/1000
    13/13 [==============================] - 0s 880us/step - loss: 0.1647
    Epoch 947/1000
    13/13 [==============================] - 0s 870us/step - loss: 0.1759
    Epoch 948/1000
    13/13 [==============================] - 0s 864us/step - loss: 0.1712
    Epoch 949/1000
    13/13 [==============================] - 0s 846us/step - loss: 0.1679
    Epoch 950/1000
    13/13 [==============================] - 0s 905us/step - loss: 0.1669
    Epoch 951/1000
    13/13 [==============================] - 0s 938us/step - loss: 0.1733
    Epoch 952/1000
    13/13 [==============================] - 0s 1000us/step - loss: 0.1662
    Epoch 953/1000
    13/13 [==============================] - 0s 903us/step - loss: 0.1751
    Epoch 954/1000
    13/13 [==============================] - 0s 902us/step - loss: 0.1705
    Epoch 955/1000
    13/13 [==============================] - 0s 854us/step - loss: 0.1661
    Epoch 956/1000
    13/13 [==============================] - 0s 852us/step - loss: 0.1658
    Epoch 957/1000
    13/13 [==============================] - 0s 878us/step - loss: 0.1676
    Epoch 958/1000
    13/13 [==============================] - 0s 910us/step - loss: 0.1718
    Epoch 959/1000
    13/13 [==============================] - 0s 884us/step - loss: 0.1644
    Epoch 960/1000
    13/13 [==============================] - 0s 879us/step - loss: 0.1697
    Epoch 961/1000
    13/13 [==============================] - 0s 845us/step - loss: 0.1654
    Epoch 962/1000
    13/13 [==============================] - 0s 830us/step - loss: 0.1667
    Epoch 963/1000
    13/13 [==============================] - 0s 829us/step - loss: 0.1757
    Epoch 964/1000
    13/13 [==============================] - 0s 842us/step - loss: 0.1661
    Epoch 965/1000
    13/13 [==============================] - 0s 887us/step - loss: 0.1713
    Epoch 966/1000
    13/13 [==============================] - 0s 925us/step - loss: 0.1671
    Epoch 967/1000
    13/13 [==============================] - 0s 951us/step - loss: 0.1697
    Epoch 968/1000
    13/13 [==============================] - 0s 882us/step - loss: 0.1716
    Epoch 969/1000
    13/13 [==============================] - 0s 893us/step - loss: 0.1688
    Epoch 970/1000
    13/13 [==============================] - 0s 914us/step - loss: 0.1672
    Epoch 971/1000
    13/13 [==============================] - 0s 900us/step - loss: 0.1664
    Epoch 972/1000
    13/13 [==============================] - 0s 928us/step - loss: 0.1684
    Epoch 973/1000
    13/13 [==============================] - 0s 969us/step - loss: 0.1660
    Epoch 974/1000
    13/13 [==============================] - 0s 946us/step - loss: 0.1678
    Epoch 975/1000
    13/13 [==============================] - 0s 898us/step - loss: 0.1675
    Epoch 976/1000
    13/13 [==============================] - 0s 835us/step - loss: 0.1710
    Epoch 977/1000
    13/13 [==============================] - 0s 834us/step - loss: 0.1722
    Epoch 978/1000
    13/13 [==============================] - 0s 822us/step - loss: 0.1648
    Epoch 979/1000
    13/13 [==============================] - 0s 870us/step - loss: 0.1716
    Epoch 980/1000
    13/13 [==============================] - 0s 979us/step - loss: 0.1666
    Epoch 981/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1666
    Epoch 982/1000
    13/13 [==============================] - 0s 862us/step - loss: 0.1696
    Epoch 983/1000
    13/13 [==============================] - 0s 846us/step - loss: 0.1703
    Epoch 984/1000
    13/13 [==============================] - 0s 835us/step - loss: 0.1655
    Epoch 985/1000
    13/13 [==============================] - 0s 838us/step - loss: 0.1658
    Epoch 986/1000
    13/13 [==============================] - 0s 830us/step - loss: 0.1691
    Epoch 987/1000
    13/13 [==============================] - 0s 861us/step - loss: 0.1665
    Epoch 988/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1680
    Epoch 989/1000
    13/13 [==============================] - 0s 895us/step - loss: 0.1682
    Epoch 990/1000
    13/13 [==============================] - 0s 846us/step - loss: 0.1664
    Epoch 991/1000
    13/13 [==============================] - 0s 814us/step - loss: 0.1682
    Epoch 992/1000
    13/13 [==============================] - 0s 822us/step - loss: 0.1685
    Epoch 993/1000
    13/13 [==============================] - 0s 868us/step - loss: 0.1672
    Epoch 994/1000
    13/13 [==============================] - 0s 810us/step - loss: 0.1660
    Epoch 995/1000
    13/13 [==============================] - 0s 852us/step - loss: 0.1705
    Epoch 996/1000
    13/13 [==============================] - 0s 878us/step - loss: 0.1678
    Epoch 997/1000
    13/13 [==============================] - 0s 904us/step - loss: 0.1689
    Epoch 998/1000
    13/13 [==============================] - 0s 852us/step - loss: 0.1701
    Epoch 999/1000
    13/13 [==============================] - 0s 794us/step - loss: 0.1711
    Epoch 1000/1000
    13/13 [==============================] - 0s 804us/step - loss: 0.1628





    <keras.callbacks.History at 0x7f2914347b10>




```python
# BEGIN UNIT TEST
model_s.summary()

model_s_test(model_s, classes, X_train.shape[1])
# END UNIT TEST
```

    Model: "Simple"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     dense_3 (Dense)             (None, 6)                 18        
                                                                     
     dense_4 (Dense)             (None, 6)                 42        
                                                                     
    =================================================================
    Total params: 60
    Trainable params: 60
    Non-trainable params: 0
    _________________________________________________________________
    [92mAll tests passed!


<details>
  <summary><font size="3" color="darkgreen"><b>Click for hints</b></font></summary>
    
Summary should match this (layer instance names may increment )
```
Model: "Simple"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
L1 (Dense)                   (None, 6)                 18        
_________________________________________________________________
L2 (Dense)                   (None, 6)                 42        
=================================================================
Total params: 60
Trainable params: 60
Non-trainable params: 0
_________________________________________________________________
```
  <details>
  <summary><font size="3" color="darkgreen"><b>Click for more hints</b></font></summary>
  
```python
tf.random.set_seed(1234)
model_s = Sequential(
    [
        Dense(6, activation = 'relu', name="L1"),            # @REPLACE
        Dense(classes, activation = 'linear', name="L2")     # @REPLACE
    ], name = "Simple"
)
model_s.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),     # @REPLACE
    optimizer=tf.keras.optimizers.Adam(0.01),     # @REPLACE
)

model_s.fit(
    X_train,y_train,
    epochs=1000
)                                   
``` 


```python
#make a model for plotting routines to call
model_predict_s = lambda Xl: np.argmax(tf.nn.softmax(model_s.predict(Xl)).numpy(),axis=1)
plt_nn(model_predict_s,X_train,y_train, classes, X_cv, y_cv, suptitle="Simple Model")
```


    Canvas(toolbar=Toolbar(toolitems=[('Home', 'Reset original view', 'home', 'home'), ('Back', 'Back to previous …


This simple models does pretty well. Let's calculate the classification error.


```python
training_cerr_simple = eval_cat_err(y_train, model_predict_s(X_train))
cv_cerr_simple = eval_cat_err(y_cv, model_predict_s(X_cv))
print(f"categorization error, training, simple model, {training_cerr_simple:0.3f}, complex model: {training_cerr_complex:0.3f}" )
print(f"categorization error, cv,       simple model, {cv_cerr_simple:0.3f}, complex model: {cv_cerr_complex:0.3f}" )
```

    categorization error, training, simple model, 0.062, complex model: 0.003
    categorization error, cv,       simple model, 0.087, complex model: 0.122


Our simple model has a little higher classification error on training data but does better on cross-validation data than the more complex model.

<a name="6"></a>
## 6 - Regularization
As in the case of polynomial regression, one can apply regularization to moderate the impact of a more complex model. Let's try this below.

<a name="ex05"></a>
### Exercise 5

Reconstruct your complex model, but this time include regularization.
Below, compose a three-layer model:
* Dense layer with 120 units, relu activation, `kernel_regularizer=tf.keras.regularizers.l2(0.1)`
* Dense layer with 40 units, relu activation, `kernel_regularizer=tf.keras.regularizers.l2(0.1)`
* Dense layer with 6 units and a linear activation. 
Compile using
* loss with `SparseCategoricalCrossentropy`, remember to use  `from_logits=True`
* Adam optimizer with learning rate of 0.01.


```python
# UNQ_C5
# GRADED CELL: model_r

tf.random.set_seed(1234)
model_r = Sequential(
    [
        ### START CODE HERE ### 
        tf.keras.layers.Dense(120, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.1)),
        tf.keras.layers.Dense(40, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.1)),
        tf.keras.layers.Dense(6, activation="linear")
        ### START CODE HERE ### 
    ], name= None
)
model_r.compile(
    ### START CODE HERE ### 
    loss=SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(lr=0.01),
    ### START CODE HERE ### 
)
```


```python
# BEGIN UNIT TEST
model_r.fit(
    X_train, y_train,
    epochs=1000
)
# END UNIT TEST
```

    Epoch 1/1000
    13/13 [==============================] - 0s 1ms/step - loss: 4.4464
    Epoch 2/1000
    13/13 [==============================] - 0s 1ms/step - loss: 1.7086
    Epoch 3/1000
    13/13 [==============================] - 0s 1ms/step - loss: 1.3465
    Epoch 4/1000
    13/13 [==============================] - 0s 2ms/step - loss: 1.0870
    Epoch 5/1000
    13/13 [==============================] - 0s 1ms/step - loss: 1.0137
    Epoch 6/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.9718
    Epoch 7/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.9481
    Epoch 8/1000
    13/13 [==============================] - 0s 4ms/step - loss: 0.8934
    Epoch 9/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.8171
    Epoch 10/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.7715
    Epoch 11/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.7611
    Epoch 12/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.7521
    Epoch 13/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.7430
    Epoch 14/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.7474
    Epoch 15/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.7045
    Epoch 16/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.7056
    Epoch 17/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.7182
    Epoch 18/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.7126
    Epoch 19/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.6868
    Epoch 20/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.6733
    Epoch 21/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.6572
    Epoch 22/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.6630
    Epoch 23/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.6508
    Epoch 24/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.6395
    Epoch 25/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.6603
    Epoch 26/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.7651
    Epoch 27/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.6369
    Epoch 28/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.6122
    Epoch 29/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.6002
    Epoch 30/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.6216
    Epoch 31/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.6096
    Epoch 32/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.6260
    Epoch 33/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.6151
    Epoch 34/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.6551
    Epoch 35/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.6538
    Epoch 36/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.6324
    Epoch 37/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.5940
    Epoch 38/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.5739
    Epoch 39/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.5686
    Epoch 40/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.5697
    Epoch 41/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.5845
    Epoch 42/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.5564
    Epoch 43/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.5791
    Epoch 44/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.5855
    Epoch 45/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.5822
    Epoch 46/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.5683
    Epoch 47/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.5278
    Epoch 48/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.5762
    Epoch 49/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.5532
    Epoch 50/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.5313
    Epoch 51/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.5409
    Epoch 52/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.5302
    Epoch 53/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.5362
    Epoch 54/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.5209
    Epoch 55/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.5680
    Epoch 56/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.5131
    Epoch 57/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.5216
    Epoch 58/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.5181
    Epoch 59/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.5470
    Epoch 60/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.5524
    Epoch 61/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.5482
    Epoch 62/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.5393
    Epoch 63/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.5135
    Epoch 64/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.5322
    Epoch 65/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.5148
    Epoch 66/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.5021
    Epoch 67/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.5041
    Epoch 68/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.5086
    Epoch 69/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.5108
    Epoch 70/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.5156
    Epoch 71/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.5115
    Epoch 72/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.5003
    Epoch 73/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4989
    Epoch 74/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.5097
    Epoch 75/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.5001
    Epoch 76/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.5060
    Epoch 77/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4977
    Epoch 78/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.5227
    Epoch 79/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.5380
    Epoch 80/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.5101
    Epoch 81/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.5247
    Epoch 82/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4910
    Epoch 83/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4799
    Epoch 84/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4673
    Epoch 85/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4877
    Epoch 86/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4816
    Epoch 87/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.4969
    Epoch 88/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4812
    Epoch 89/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4776
    Epoch 90/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4696
    Epoch 91/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.4759
    Epoch 92/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4731
    Epoch 93/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4599
    Epoch 94/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4623
    Epoch 95/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4669
    Epoch 96/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.4545
    Epoch 97/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4709
    Epoch 98/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4669
    Epoch 99/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4961
    Epoch 100/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.4954
    Epoch 101/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4874
    Epoch 102/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4759
    Epoch 103/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4739
    Epoch 104/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4682
    Epoch 105/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.5125
    Epoch 106/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4548
    Epoch 107/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4610
    Epoch 108/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4702
    Epoch 109/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.4565
    Epoch 110/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4568
    Epoch 111/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4550
    Epoch 112/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4541
    Epoch 113/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.4450
    Epoch 114/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4411
    Epoch 115/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4398
    Epoch 116/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4482
    Epoch 117/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4724
    Epoch 118/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.4591
    Epoch 119/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4686
    Epoch 120/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4736
    Epoch 121/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.5020
    Epoch 122/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.4630
    Epoch 123/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4543
    Epoch 124/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4465
    Epoch 125/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4328
    Epoch 126/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.4386
    Epoch 127/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4468
    Epoch 128/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4348
    Epoch 129/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4419
    Epoch 130/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4371
    Epoch 131/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.4542
    Epoch 132/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4331
    Epoch 133/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4236
    Epoch 134/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4470
    Epoch 135/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.4431
    Epoch 136/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4460
    Epoch 137/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4281
    Epoch 138/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4470
    Epoch 139/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.4480
    Epoch 140/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4627
    Epoch 141/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4332
    Epoch 142/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4201
    Epoch 143/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4340
    Epoch 144/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.4382
    Epoch 145/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4264
    Epoch 146/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4260
    Epoch 147/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4603
    Epoch 148/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.4396
    Epoch 149/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4239
    Epoch 150/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4208
    Epoch 151/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4169
    Epoch 152/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.4201
    Epoch 153/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4391
    Epoch 154/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4230
    Epoch 155/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4316
    Epoch 156/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4312
    Epoch 157/1000
    13/13 [==============================] - 0s 4ms/step - loss: 0.4280
    Epoch 158/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4210
    Epoch 159/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4066
    Epoch 160/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4302
    Epoch 161/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.4433
    Epoch 162/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4284
    Epoch 163/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4102
    Epoch 164/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4265
    Epoch 165/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.4454
    Epoch 166/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4595
    Epoch 167/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4779
    Epoch 168/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4529
    Epoch 169/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4328
    Epoch 170/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.4336
    Epoch 171/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4206
    Epoch 172/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4214
    Epoch 173/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4343
    Epoch 174/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.4415
    Epoch 175/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4200
    Epoch 176/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4431
    Epoch 177/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4323
    Epoch 178/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4162
    Epoch 179/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4214
    Epoch 180/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4130
    Epoch 181/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4324
    Epoch 182/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4232
    Epoch 183/1000
    13/13 [==============================] - 0s 4ms/step - loss: 0.4093
    Epoch 184/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4030
    Epoch 185/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4055
    Epoch 186/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4087
    Epoch 187/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.4134
    Epoch 188/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4165
    Epoch 189/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3974
    Epoch 190/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3971
    Epoch 191/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.4116
    Epoch 192/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4153
    Epoch 193/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4132
    Epoch 194/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4158
    Epoch 195/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4026
    Epoch 196/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3953
    Epoch 197/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4191
    Epoch 198/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3963
    Epoch 199/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4080
    Epoch 200/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.4032
    Epoch 201/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4268
    Epoch 202/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3954
    Epoch 203/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3980
    Epoch 204/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.4088
    Epoch 205/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4571
    Epoch 206/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4315
    Epoch 207/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4097
    Epoch 208/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4166
    Epoch 209/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.4393
    Epoch 210/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4124
    Epoch 211/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4216
    Epoch 212/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4118
    Epoch 213/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.4038
    Epoch 214/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4036
    Epoch 215/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3945
    Epoch 216/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4068
    Epoch 217/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3940
    Epoch 218/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4194
    Epoch 219/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3976
    Epoch 220/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3994
    Epoch 221/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3873
    Epoch 222/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4067
    Epoch 223/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4034
    Epoch 224/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4393
    Epoch 225/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4334
    Epoch 226/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.4213
    Epoch 227/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4377
    Epoch 228/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3912
    Epoch 229/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4028
    Epoch 230/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.4112
    Epoch 231/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4021
    Epoch 232/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4107
    Epoch 233/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3893
    Epoch 234/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3889
    Epoch 235/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3881
    Epoch 236/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3966
    Epoch 237/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3954
    Epoch 238/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4168
    Epoch 239/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.4049
    Epoch 240/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3863
    Epoch 241/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3890
    Epoch 242/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3908
    Epoch 243/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3888
    Epoch 244/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3984
    Epoch 245/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3993
    Epoch 246/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4078
    Epoch 247/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3814
    Epoch 248/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3897
    Epoch 249/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3995
    Epoch 250/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3910
    Epoch 251/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4142
    Epoch 252/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.4036
    Epoch 253/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3950
    Epoch 254/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4073
    Epoch 255/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4041
    Epoch 256/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3808
    Epoch 257/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4020
    Epoch 258/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3885
    Epoch 259/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3947
    Epoch 260/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3841
    Epoch 261/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.4000
    Epoch 262/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4665
    Epoch 263/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4367
    Epoch 264/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3957
    Epoch 265/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3989
    Epoch 266/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4251
    Epoch 267/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4346
    Epoch 268/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4114
    Epoch 269/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3832
    Epoch 270/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3787
    Epoch 271/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3874
    Epoch 272/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3891
    Epoch 273/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4039
    Epoch 274/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3776
    Epoch 275/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3903
    Epoch 276/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3870
    Epoch 277/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3825
    Epoch 278/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3812
    Epoch 279/1000
    13/13 [==============================] - 0s 4ms/step - loss: 0.4026
    Epoch 280/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3938
    Epoch 281/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3764
    Epoch 282/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3800
    Epoch 283/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3876
    Epoch 284/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3853
    Epoch 285/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4070
    Epoch 286/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3956
    Epoch 287/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3915
    Epoch 288/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3877
    Epoch 289/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3760
    Epoch 290/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3892
    Epoch 291/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3911
    Epoch 292/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3697
    Epoch 293/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3800
    Epoch 294/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4007
    Epoch 295/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4066
    Epoch 296/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3768
    Epoch 297/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3841
    Epoch 298/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3884
    Epoch 299/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3926
    Epoch 300/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4250
    Epoch 301/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3915
    Epoch 302/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3894
    Epoch 303/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3858
    Epoch 304/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3804
    Epoch 305/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3810
    Epoch 306/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3883
    Epoch 307/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3922
    Epoch 308/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3879
    Epoch 309/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3801
    Epoch 310/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3715
    Epoch 311/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3690
    Epoch 312/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3733
    Epoch 313/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3863
    Epoch 314/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3843
    Epoch 315/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3822
    Epoch 316/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3789
    Epoch 317/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3808
    Epoch 318/1000
    13/13 [==============================] - 0s 4ms/step - loss: 0.3742
    Epoch 319/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3791
    Epoch 320/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3836
    Epoch 321/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3935
    Epoch 322/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3927
    Epoch 323/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4023
    Epoch 324/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4109
    Epoch 325/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3989
    Epoch 326/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3860
    Epoch 327/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3807
    Epoch 328/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3919
    Epoch 329/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3763
    Epoch 330/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3669
    Epoch 331/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3715
    Epoch 332/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3724
    Epoch 333/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4101
    Epoch 334/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3930
    Epoch 335/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3933
    Epoch 336/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3975
    Epoch 337/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4038
    Epoch 338/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3737
    Epoch 339/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3719
    Epoch 340/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3868
    Epoch 341/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3792
    Epoch 342/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3749
    Epoch 343/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3693
    Epoch 344/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3644
    Epoch 345/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3633
    Epoch 346/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3662
    Epoch 347/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3888
    Epoch 348/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4182
    Epoch 349/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3776
    Epoch 350/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4027
    Epoch 351/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3697
    Epoch 352/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3903
    Epoch 353/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3757
    Epoch 354/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3691
    Epoch 355/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3733
    Epoch 356/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3651
    Epoch 357/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3814
    Epoch 358/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3961
    Epoch 359/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3892
    Epoch 360/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3938
    Epoch 361/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4104
    Epoch 362/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4556
    Epoch 363/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4061
    Epoch 364/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3714
    Epoch 365/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3674
    Epoch 366/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3638
    Epoch 367/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3693
    Epoch 368/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3912
    Epoch 369/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3991
    Epoch 370/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3732
    Epoch 371/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3608
    Epoch 372/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3611
    Epoch 373/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3791
    Epoch 374/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3565
    Epoch 375/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3797
    Epoch 376/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3772
    Epoch 377/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3616
    Epoch 378/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3748
    Epoch 379/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3832
    Epoch 380/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3814
    Epoch 381/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4119
    Epoch 382/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3712
    Epoch 383/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3780
    Epoch 384/1000
    13/13 [==============================] - 0s 2ms/step - loss: 0.3642
    Epoch 385/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3681
    Epoch 386/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3574
    Epoch 387/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3764
    Epoch 388/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3717
    Epoch 389/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3674
    Epoch 390/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3531
    Epoch 391/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3664
    Epoch 392/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3819
    Epoch 393/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3605
    Epoch 394/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3635
    Epoch 395/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3932
    Epoch 396/1000
    13/13 [==============================] - 0s 4ms/step - loss: 0.3799
    Epoch 397/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3915
    Epoch 398/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3771
    Epoch 399/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3753
    Epoch 400/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3727
    Epoch 401/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3584
    Epoch 402/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3613
    Epoch 403/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3600
    Epoch 404/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3617
    Epoch 405/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3545
    Epoch 406/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3600
    Epoch 407/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3698
    Epoch 408/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3630
    Epoch 409/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3818
    Epoch 410/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3842
    Epoch 411/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3936
    Epoch 412/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3794
    Epoch 413/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3626
    Epoch 414/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3576
    Epoch 415/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3730
    Epoch 416/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3806
    Epoch 417/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3915
    Epoch 418/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3629
    Epoch 419/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3673
    Epoch 420/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3534
    Epoch 421/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3874
    Epoch 422/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3942
    Epoch 423/1000
    13/13 [==============================] - 0s 2ms/step - loss: 0.3729
    Epoch 424/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3723
    Epoch 425/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3682
    Epoch 426/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3655
    Epoch 427/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3641
    Epoch 428/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3707
    Epoch 429/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3673
    Epoch 430/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3631
    Epoch 431/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3523
    Epoch 432/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3592
    Epoch 433/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3893
    Epoch 434/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3961
    Epoch 435/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4097
    Epoch 436/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3961
    Epoch 437/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3837
    Epoch 438/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3836
    Epoch 439/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3501
    Epoch 440/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3474
    Epoch 441/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3626
    Epoch 442/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3807
    Epoch 443/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3725
    Epoch 444/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3662
    Epoch 445/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3735
    Epoch 446/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3537
    Epoch 447/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3685
    Epoch 448/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3609
    Epoch 449/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3533
    Epoch 450/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3551
    Epoch 451/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3492
    Epoch 452/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3630
    Epoch 453/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3763
    Epoch 454/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3718
    Epoch 455/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3727
    Epoch 456/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3628
    Epoch 457/1000
    13/13 [==============================] - 0s 2ms/step - loss: 0.3558
    Epoch 458/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3812
    Epoch 459/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3643
    Epoch 460/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3624
    Epoch 461/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3632
    Epoch 462/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3509
    Epoch 463/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3559
    Epoch 464/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3718
    Epoch 465/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3495
    Epoch 466/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3765
    Epoch 467/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3667
    Epoch 468/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4002
    Epoch 469/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.4147
    Epoch 470/1000
    13/13 [==============================] - 0s 2ms/step - loss: 0.3473
    Epoch 471/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3688
    Epoch 472/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4113
    Epoch 473/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.4088
    Epoch 474/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3998
    Epoch 475/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3723
    Epoch 476/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3604
    Epoch 477/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3805
    Epoch 478/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3670
    Epoch 479/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3594
    Epoch 480/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3609
    Epoch 481/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3550
    Epoch 482/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3755
    Epoch 483/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3802
    Epoch 484/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3782
    Epoch 485/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3808
    Epoch 486/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3564
    Epoch 487/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3470
    Epoch 488/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3539
    Epoch 489/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3401
    Epoch 490/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3561
    Epoch 491/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3693
    Epoch 492/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3690
    Epoch 493/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3510
    Epoch 494/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3548
    Epoch 495/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3525
    Epoch 496/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3736
    Epoch 497/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4008
    Epoch 498/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3497
    Epoch 499/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3444
    Epoch 500/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3610
    Epoch 501/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3546
    Epoch 502/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3586
    Epoch 503/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3814
    Epoch 504/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3645
    Epoch 505/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3684
    Epoch 506/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3834
    Epoch 507/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3581
    Epoch 508/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3402
    Epoch 509/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3503
    Epoch 510/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3488
    Epoch 511/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3514
    Epoch 512/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3611
    Epoch 513/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3482
    Epoch 514/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3461
    Epoch 515/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3535
    Epoch 516/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3595
    Epoch 517/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3676
    Epoch 518/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3638
    Epoch 519/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3670
    Epoch 520/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3616
    Epoch 521/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3475
    Epoch 522/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3659
    Epoch 523/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3748
    Epoch 524/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3416
    Epoch 525/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3484
    Epoch 526/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3559
    Epoch 527/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3420
    Epoch 528/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3476
    Epoch 529/1000
    13/13 [==============================] - 0s 4ms/step - loss: 0.3793
    Epoch 530/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3642
    Epoch 531/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3761
    Epoch 532/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3456
    Epoch 533/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3398
    Epoch 534/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3614
    Epoch 535/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3618
    Epoch 536/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3422
    Epoch 537/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.4039
    Epoch 538/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3591
    Epoch 539/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3597
    Epoch 540/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3934
    Epoch 541/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.4010
    Epoch 542/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3746
    Epoch 543/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3709
    Epoch 544/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3576
    Epoch 545/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3510
    Epoch 546/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3669
    Epoch 547/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3648
    Epoch 548/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3654
    Epoch 549/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3436
    Epoch 550/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3411
    Epoch 551/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3460
    Epoch 552/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3460
    Epoch 553/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3396
    Epoch 554/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3513
    Epoch 555/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3890
    Epoch 556/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3884
    Epoch 557/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3706
    Epoch 558/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3578
    Epoch 559/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3826
    Epoch 560/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3486
    Epoch 561/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3443
    Epoch 562/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3528
    Epoch 563/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3515
    Epoch 564/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3615
    Epoch 565/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3448
    Epoch 566/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3620
    Epoch 567/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3439
    Epoch 568/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3493
    Epoch 569/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3499
    Epoch 570/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3386
    Epoch 571/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3667
    Epoch 572/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3514
    Epoch 573/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3500
    Epoch 574/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3619
    Epoch 575/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3435
    Epoch 576/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3396
    Epoch 577/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3557
    Epoch 578/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4221
    Epoch 579/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3583
    Epoch 580/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3376
    Epoch 581/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3628
    Epoch 582/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3540
    Epoch 583/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3571
    Epoch 584/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3818
    Epoch 585/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3954
    Epoch 586/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3669
    Epoch 587/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3536
    Epoch 588/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3407
    Epoch 589/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3348
    Epoch 590/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3374
    Epoch 591/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3489
    Epoch 592/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3452
    Epoch 593/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3429
    Epoch 594/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3425
    Epoch 595/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4209
    Epoch 596/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3978
    Epoch 597/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3565
    Epoch 598/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3443
    Epoch 599/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3419
    Epoch 600/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3529
    Epoch 601/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3345
    Epoch 602/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3436
    Epoch 603/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3594
    Epoch 604/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3504
    Epoch 605/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3590
    Epoch 606/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3738
    Epoch 607/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3654
    Epoch 608/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3516
    Epoch 609/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3480
    Epoch 610/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3599
    Epoch 611/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3539
    Epoch 612/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3668
    Epoch 613/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3593
    Epoch 614/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3483
    Epoch 615/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3536
    Epoch 616/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3456
    Epoch 617/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3287
    Epoch 618/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3673
    Epoch 619/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4033
    Epoch 620/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3884
    Epoch 621/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3619
    Epoch 622/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3834
    Epoch 623/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3413
    Epoch 624/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3359
    Epoch 625/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3319
    Epoch 626/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3425
    Epoch 627/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3567
    Epoch 628/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3715
    Epoch 629/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3719
    Epoch 630/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3774
    Epoch 631/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3697
    Epoch 632/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3777
    Epoch 633/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3753
    Epoch 634/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3749
    Epoch 635/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3667
    Epoch 636/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3486
    Epoch 637/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3488
    Epoch 638/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3443
    Epoch 639/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3455
    Epoch 640/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3583
    Epoch 641/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3428
    Epoch 642/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3522
    Epoch 643/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3642
    Epoch 644/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3473
    Epoch 645/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3546
    Epoch 646/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3543
    Epoch 647/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3561
    Epoch 648/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3643
    Epoch 649/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3590
    Epoch 650/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3484
    Epoch 651/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3427
    Epoch 652/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3329
    Epoch 653/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3478
    Epoch 654/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3550
    Epoch 655/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3478
    Epoch 656/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3361
    Epoch 657/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3457
    Epoch 658/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3430
    Epoch 659/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3480
    Epoch 660/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3667
    Epoch 661/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3403
    Epoch 662/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3545
    Epoch 663/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3889
    Epoch 664/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3568
    Epoch 665/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3541
    Epoch 666/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3520
    Epoch 667/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3340
    Epoch 668/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3299
    Epoch 669/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3509
    Epoch 670/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3352
    Epoch 671/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3466
    Epoch 672/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3784
    Epoch 673/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4029
    Epoch 674/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4009
    Epoch 675/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3426
    Epoch 676/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3406
    Epoch 677/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3369
    Epoch 678/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3356
    Epoch 679/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3463
    Epoch 680/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3406
    Epoch 681/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3549
    Epoch 682/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3399
    Epoch 683/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3363
    Epoch 684/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3415
    Epoch 685/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3470
    Epoch 686/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3487
    Epoch 687/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3424
    Epoch 688/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3321
    Epoch 689/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3976
    Epoch 690/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3724
    Epoch 691/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3471
    Epoch 692/1000
    13/13 [==============================] - 0s 2ms/step - loss: 0.3554
    Epoch 693/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3445
    Epoch 694/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3483
    Epoch 695/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3390
    Epoch 696/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3378
    Epoch 697/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3355
    Epoch 698/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3517
    Epoch 699/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3456
    Epoch 700/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3493
    Epoch 701/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3460
    Epoch 702/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3256
    Epoch 703/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3269
    Epoch 704/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3510
    Epoch 705/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3470
    Epoch 706/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3533
    Epoch 707/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3518
    Epoch 708/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3458
    Epoch 709/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3581
    Epoch 710/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3513
    Epoch 711/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3361
    Epoch 712/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3854
    Epoch 713/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3573
    Epoch 714/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3398
    Epoch 715/1000
    13/13 [==============================] - 0s 4ms/step - loss: 0.3291
    Epoch 716/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3360
    Epoch 717/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3615
    Epoch 718/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3587
    Epoch 719/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.4233
    Epoch 720/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4165
    Epoch 721/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3999
    Epoch 722/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3667
    Epoch 723/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3688
    Epoch 724/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3474
    Epoch 725/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3534
    Epoch 726/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3492
    Epoch 727/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3512
    Epoch 728/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3524
    Epoch 729/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3441
    Epoch 730/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3547
    Epoch 731/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3466
    Epoch 732/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3483
    Epoch 733/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3376
    Epoch 734/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3519
    Epoch 735/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3520
    Epoch 736/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3650
    Epoch 737/1000
    13/13 [==============================] - 0s 2ms/step - loss: 0.3722
    Epoch 738/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3423
    Epoch 739/1000
    13/13 [==============================] - 0s 2ms/step - loss: 0.3472
    Epoch 740/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3422
    Epoch 741/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3447
    Epoch 742/1000
    13/13 [==============================] - 0s 2ms/step - loss: 0.3786
    Epoch 743/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3409
    Epoch 744/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3318
    Epoch 745/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3281
    Epoch 746/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3304
    Epoch 747/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3277
    Epoch 748/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3441
    Epoch 749/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3797
    Epoch 750/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3511
    Epoch 751/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3599
    Epoch 752/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4169
    Epoch 753/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4063
    Epoch 754/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3516
    Epoch 755/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3407
    Epoch 756/1000
    13/13 [==============================] - 0s 2ms/step - loss: 0.3493
    Epoch 757/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3608
    Epoch 758/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3780
    Epoch 759/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3424
    Epoch 760/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3436
    Epoch 761/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3541
    Epoch 762/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3457
    Epoch 763/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3317
    Epoch 764/1000
    13/13 [==============================] - 0s 4ms/step - loss: 0.3496
    Epoch 765/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3551
    Epoch 766/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3396
    Epoch 767/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3339
    Epoch 768/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3589
    Epoch 769/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3521
    Epoch 770/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3301
    Epoch 771/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3454
    Epoch 772/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3471
    Epoch 773/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3825
    Epoch 774/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3659
    Epoch 775/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3377
    Epoch 776/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3882
    Epoch 777/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3705
    Epoch 778/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3279
    Epoch 779/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3339
    Epoch 780/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3435
    Epoch 781/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3393
    Epoch 782/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3259
    Epoch 783/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3296
    Epoch 784/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3298
    Epoch 785/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3286
    Epoch 786/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3392
    Epoch 787/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3368
    Epoch 788/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3307
    Epoch 789/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3382
    Epoch 790/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3355
    Epoch 791/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3734
    Epoch 792/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3761
    Epoch 793/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3444
    Epoch 794/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3632
    Epoch 795/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3406
    Epoch 796/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3788
    Epoch 797/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3315
    Epoch 798/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3506
    Epoch 799/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3608
    Epoch 800/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3491
    Epoch 801/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3315
    Epoch 802/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3287
    Epoch 803/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3276
    Epoch 804/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3280
    Epoch 805/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3504
    Epoch 806/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3500
    Epoch 807/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3403
    Epoch 808/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3552
    Epoch 809/1000
    13/13 [==============================] - 0s 4ms/step - loss: 0.3773
    Epoch 810/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3458
    Epoch 811/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3324
    Epoch 812/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3241
    Epoch 813/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3331
    Epoch 814/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3376
    Epoch 815/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3443
    Epoch 816/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3452
    Epoch 817/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3625
    Epoch 818/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3543
    Epoch 819/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3300
    Epoch 820/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3694
    Epoch 821/1000
    13/13 [==============================] - 0s 4ms/step - loss: 0.3836
    Epoch 822/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3472
    Epoch 823/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3578
    Epoch 824/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3510
    Epoch 825/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3420
    Epoch 826/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3308
    Epoch 827/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3247
    Epoch 828/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3456
    Epoch 829/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3698
    Epoch 830/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4228
    Epoch 831/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3441
    Epoch 832/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3515
    Epoch 833/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3434
    Epoch 834/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3518
    Epoch 835/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3238
    Epoch 836/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3339
    Epoch 837/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3339
    Epoch 838/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3434
    Epoch 839/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3268
    Epoch 840/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3740
    Epoch 841/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3566
    Epoch 842/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3545
    Epoch 843/1000
    13/13 [==============================] - 0s 2ms/step - loss: 0.3543
    Epoch 844/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3347
    Epoch 845/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3272
    Epoch 846/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3351
    Epoch 847/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3570
    Epoch 848/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3441
    Epoch 849/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3220
    Epoch 850/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3376
    Epoch 851/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3364
    Epoch 852/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3501
    Epoch 853/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3658
    Epoch 854/1000
    13/13 [==============================] - 0s 4ms/step - loss: 0.3400
    Epoch 855/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3381
    Epoch 856/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3374
    Epoch 857/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3421
    Epoch 858/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3686
    Epoch 859/1000
    13/13 [==============================] - 0s 2ms/step - loss: 0.3783
    Epoch 860/1000
    13/13 [==============================] - 0s 2ms/step - loss: 0.3459
    Epoch 861/1000
    13/13 [==============================] - 0s 2ms/step - loss: 0.3653
    Epoch 862/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3272
    Epoch 863/1000
    13/13 [==============================] - 0s 2ms/step - loss: 0.3222
    Epoch 864/1000
    13/13 [==============================] - 0s 2ms/step - loss: 0.3736
    Epoch 865/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3834
    Epoch 866/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3725
    Epoch 867/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3334
    Epoch 868/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3360
    Epoch 869/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3430
    Epoch 870/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3601
    Epoch 871/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3625
    Epoch 872/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3410
    Epoch 873/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3373
    Epoch 874/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3479
    Epoch 875/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3524
    Epoch 876/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3360
    Epoch 877/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3316
    Epoch 878/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3564
    Epoch 879/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3425
    Epoch 880/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3270
    Epoch 881/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3594
    Epoch 882/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3598
    Epoch 883/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.4354
    Epoch 884/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3778
    Epoch 885/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3704
    Epoch 886/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3419
    Epoch 887/1000
    13/13 [==============================] - 0s 4ms/step - loss: 0.3491
    Epoch 888/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3509
    Epoch 889/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3373
    Epoch 890/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3713
    Epoch 891/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3285
    Epoch 892/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3294
    Epoch 893/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3340
    Epoch 894/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3266
    Epoch 895/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3464
    Epoch 896/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3392
    Epoch 897/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3304
    Epoch 898/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3448
    Epoch 899/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3721
    Epoch 900/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3583
    Epoch 901/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3743
    Epoch 902/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3616
    Epoch 903/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3491
    Epoch 904/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3283
    Epoch 905/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3386
    Epoch 906/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3571
    Epoch 907/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3552
    Epoch 908/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3694
    Epoch 909/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4247
    Epoch 910/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3797
    Epoch 911/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3910
    Epoch 912/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3706
    Epoch 913/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3323
    Epoch 914/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3561
    Epoch 915/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3473
    Epoch 916/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3535
    Epoch 917/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3453
    Epoch 918/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3378
    Epoch 919/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3582
    Epoch 920/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3751
    Epoch 921/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3452
    Epoch 922/1000
    13/13 [==============================] - 0s 2ms/step - loss: 0.3507
    Epoch 923/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3225
    Epoch 924/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3479
    Epoch 925/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3356
    Epoch 926/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3285
    Epoch 927/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3434
    Epoch 928/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3272
    Epoch 929/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3504
    Epoch 930/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3919
    Epoch 931/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4201
    Epoch 932/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3934
    Epoch 933/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3428
    Epoch 934/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3645
    Epoch 935/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3348
    Epoch 936/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3342
    Epoch 937/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3461
    Epoch 938/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3503
    Epoch 939/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3471
    Epoch 940/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3407
    Epoch 941/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3188
    Epoch 942/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3240
    Epoch 943/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3440
    Epoch 944/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3599
    Epoch 945/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3812
    Epoch 946/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3393
    Epoch 947/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3357
    Epoch 948/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3297
    Epoch 949/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3231
    Epoch 950/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3178
    Epoch 951/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3111
    Epoch 952/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3343
    Epoch 953/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3389
    Epoch 954/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3572
    Epoch 955/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3215
    Epoch 956/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3439
    Epoch 957/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3319
    Epoch 958/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3322
    Epoch 959/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3159
    Epoch 960/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3218
    Epoch 961/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3287
    Epoch 962/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3196
    Epoch 963/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3408
    Epoch 964/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3208
    Epoch 965/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3241
    Epoch 966/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3396
    Epoch 967/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3292
    Epoch 968/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3362
    Epoch 969/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3865
    Epoch 970/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3795
    Epoch 971/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3494
    Epoch 972/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3260
    Epoch 973/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3279
    Epoch 974/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3238
    Epoch 975/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3419
    Epoch 976/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3488
    Epoch 977/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3278
    Epoch 978/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3219
    Epoch 979/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3267
    Epoch 980/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3458
    Epoch 981/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3263
    Epoch 982/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3288
    Epoch 983/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3174
    Epoch 984/1000
    13/13 [==============================] - 0s 2ms/step - loss: 0.3339
    Epoch 985/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3361
    Epoch 986/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3253
    Epoch 987/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3248
    Epoch 988/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3199
    Epoch 989/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3323
    Epoch 990/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3463
    Epoch 991/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3422
    Epoch 992/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3354
    Epoch 993/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3225
    Epoch 994/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3282
    Epoch 995/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3532
    Epoch 996/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3445
    Epoch 997/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3738
    Epoch 998/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3308
    Epoch 999/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3505
    Epoch 1000/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3514





    <keras.callbacks.History at 0x7f291409a650>




```python
# BEGIN UNIT TEST
model_r.summary()

model_r_test(model_r, classes, X_train.shape[1]) 
# END UNIT TEST
```

    Model: "sequential"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     dense_5 (Dense)             (None, 120)               360       
                                                                     
     dense_6 (Dense)             (None, 40)                4840      
                                                                     
     dense_7 (Dense)             (None, 6)                 246       
                                                                     
    =================================================================
    Total params: 5,446
    Trainable params: 5,446
    Non-trainable params: 0
    _________________________________________________________________
    ddd
    [92mAll tests passed!


<details>
  <summary><font size="3" color="darkgreen"><b>Click for hints</b></font></summary>
    
Summary should match this (layer instance names may increment )
```
Model: "ComplexRegularized"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
L1 (Dense)                   (None, 120)               360       
_________________________________________________________________
L2 (Dense)                   (None, 40)                4840      
_________________________________________________________________
L3 (Dense)                   (None, 6)                 246       
=================================================================
Total params: 5,446
Trainable params: 5,446
Non-trainable params: 0
_________________________________________________________________
```
  <details>
  <summary><font size="3" color="darkgreen"><b>Click for more hints</b></font></summary>
  
```python
tf.random.set_seed(1234)
model_r = Sequential(
    [
        Dense(120, activation = 'relu', kernel_regularizer=tf.keras.regularizers.l2(0.1), name="L1"), 
        Dense(40, activation = 'relu', kernel_regularizer=tf.keras.regularizers.l2(0.1), name="L2"),  
        Dense(classes, activation = 'linear', name="L3")  
    ], name="ComplexRegularized"
)
model_r.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
    optimizer=tf.keras.optimizers.Adam(0.01),                             
)

model_r.fit(
    X_train,y_train,
    epochs=1000
)                                   
``` 


```python
#make a model for plotting routines to call
model_predict_r = lambda Xl: np.argmax(tf.nn.softmax(model_r.predict(Xl)).numpy(),axis=1)
 
plt_nn(model_predict_r, X_train,y_train, classes, X_cv, y_cv, suptitle="Regularized")
```


    Canvas(toolbar=Toolbar(toolitems=[('Home', 'Reset original view', 'home', 'home'), ('Back', 'Back to previous …


The results look very similar to the 'ideal' model. Let's check classification error.


```python
training_cerr_reg = eval_cat_err(y_train, model_predict_r(X_train))
cv_cerr_reg = eval_cat_err(y_cv, model_predict_r(X_cv))
test_cerr_reg = eval_cat_err(y_test, model_predict_r(X_test))
print(f"categorization error, training, regularized: {training_cerr_reg:0.3f}, simple model, {training_cerr_simple:0.3f}, complex model: {training_cerr_complex:0.3f}" )
print(f"categorization error, cv,       regularized: {cv_cerr_reg:0.3f}, simple model, {cv_cerr_simple:0.3f}, complex model: {cv_cerr_complex:0.3f}" )
```

    categorization error, training, regularized: 0.072, simple model, 0.062, complex model: 0.003
    categorization error, cv,       regularized: 0.066, simple model, 0.087, complex model: 0.122


The simple model is a bit better in the training set than the regularized model but it worse in the cross validation set.

<a name="7"></a>
## 7 - Iterate to find optimal regularization value
As you did in linear regression, you can try many regularization values. This code takes several minutes to run. If you have time, you can run it and check the results. If not, you have completed the graded parts of the assignment!


```python
tf.random.set_seed(1234)
lambdas = [0.0, 0.001, 0.01, 0.05, 0.1, 0.2, 0.3]
models=[None] * len(lambdas)
for i in range(len(lambdas)):
    lambda_ = lambdas[i]
    models[i] =  Sequential(
        [
            Dense(120, activation = 'relu', kernel_regularizer=tf.keras.regularizers.l2(lambda_)),
            Dense(40, activation = 'relu', kernel_regularizer=tf.keras.regularizers.l2(lambda_)),
            Dense(classes, activation = 'linear')
        ]
    )
    models[i].compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(0.01),
    )

    models[i].fit(
        X_train,y_train,
        epochs=1000
    )
    print(f"Finished lambda = {lambda_}")

```

    Epoch 1/1000
    13/13 [==============================] - 0s 1ms/step - loss: 1.1106
    Epoch 2/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4281
    Epoch 3/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3345
    Epoch 4/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2896
    Epoch 5/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2867
    Epoch 6/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2918
    Epoch 7/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2497
    Epoch 8/1000
    13/13 [==============================] - 0s 4ms/step - loss: 0.2298
    Epoch 9/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2307
    Epoch 10/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2071
    Epoch 11/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2115
    Epoch 12/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2070
    Epoch 13/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2366
    Epoch 14/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2261
    Epoch 15/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2224
    Epoch 16/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2055
    Epoch 17/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.2044
    Epoch 18/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2006
    Epoch 19/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2168
    Epoch 20/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2047
    Epoch 21/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2237
    Epoch 22/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.2497
    Epoch 23/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2113
    Epoch 24/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2025
    Epoch 25/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2107
    Epoch 26/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.2000
    Epoch 27/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1935
    Epoch 28/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1963
    Epoch 29/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2188
    Epoch 30/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2424
    Epoch 31/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1969
    Epoch 32/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1950
    Epoch 33/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1904
    Epoch 34/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2173
    Epoch 35/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.2074
    Epoch 36/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1768
    Epoch 37/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1794
    Epoch 38/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1733
    Epoch 39/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1955
    Epoch 40/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1870
    Epoch 41/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2128
    Epoch 42/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1987
    Epoch 43/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1895
    Epoch 44/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.2073
    Epoch 45/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2148
    Epoch 46/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1774
    Epoch 47/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1886
    Epoch 48/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1763
    Epoch 49/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1769
    Epoch 50/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1763
    Epoch 51/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2020
    Epoch 52/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1889
    Epoch 53/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.2035
    Epoch 54/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1761
    Epoch 55/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1838
    Epoch 56/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1774
    Epoch 57/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1953
    Epoch 58/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1882
    Epoch 59/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1860
    Epoch 60/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1919
    Epoch 61/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1848
    Epoch 62/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1630
    Epoch 63/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1616
    Epoch 64/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2008
    Epoch 65/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1936
    Epoch 66/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1824
    Epoch 67/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2092
    Epoch 68/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2287
    Epoch 69/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1877
    Epoch 70/1000
    13/13 [==============================] - 0s 4ms/step - loss: 0.1716
    Epoch 71/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1917
    Epoch 72/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1703
    Epoch 73/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1750
    Epoch 74/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1836
    Epoch 75/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1696
    Epoch 76/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1542
    Epoch 77/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1715
    Epoch 78/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1545
    Epoch 79/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1593
    Epoch 80/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1844
    Epoch 81/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1881
    Epoch 82/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1696
    Epoch 83/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1614
    Epoch 84/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1762
    Epoch 85/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1779
    Epoch 86/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1658
    Epoch 87/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1614
    Epoch 88/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1639
    Epoch 89/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1629
    Epoch 90/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1475
    Epoch 91/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1452
    Epoch 92/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1473
    Epoch 93/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1490
    Epoch 94/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1650
    Epoch 95/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1706
    Epoch 96/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1704
    Epoch 97/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1764
    Epoch 98/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1855
    Epoch 99/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1685
    Epoch 100/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1569
    Epoch 101/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1645
    Epoch 102/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1737
    Epoch 103/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1935
    Epoch 104/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1600
    Epoch 105/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1483
    Epoch 106/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1555
    Epoch 107/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1678
    Epoch 108/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1435
    Epoch 109/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1419
    Epoch 110/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1494
    Epoch 111/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1538
    Epoch 112/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1682
    Epoch 113/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1687
    Epoch 114/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1436
    Epoch 115/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1366
    Epoch 116/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1485
    Epoch 117/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1400
    Epoch 118/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1357
    Epoch 119/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1444
    Epoch 120/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1403
    Epoch 121/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1465
    Epoch 122/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1549
    Epoch 123/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1402
    Epoch 124/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1337
    Epoch 125/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1422
    Epoch 126/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1560
    Epoch 127/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1319
    Epoch 128/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1389
    Epoch 129/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1404
    Epoch 130/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1299
    Epoch 131/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1247
    Epoch 132/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1244
    Epoch 133/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1260
    Epoch 134/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1158
    Epoch 135/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1343
    Epoch 136/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1306
    Epoch 137/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1294
    Epoch 138/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1297
    Epoch 139/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1342
    Epoch 140/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1255
    Epoch 141/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1232
    Epoch 142/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1199
    Epoch 143/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1192
    Epoch 144/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1192
    Epoch 145/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1342
    Epoch 146/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1477
    Epoch 147/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1780
    Epoch 148/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1673
    Epoch 149/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1402
    Epoch 150/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1292
    Epoch 151/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1296
    Epoch 152/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1221
    Epoch 153/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1300
    Epoch 154/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1316
    Epoch 155/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1274
    Epoch 156/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1192
    Epoch 157/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1266
    Epoch 158/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1185
    Epoch 159/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1197
    Epoch 160/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1148
    Epoch 161/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1137
    Epoch 162/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1427
    Epoch 163/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1420
    Epoch 164/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1327
    Epoch 165/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1276
    Epoch 166/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1099
    Epoch 167/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1205
    Epoch 168/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1307
    Epoch 169/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1476
    Epoch 170/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1673
    Epoch 171/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1349
    Epoch 172/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1183
    Epoch 173/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1225
    Epoch 174/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1276
    Epoch 175/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1029
    Epoch 176/1000
    13/13 [==============================] - 0s 2ms/step - loss: 0.1134
    Epoch 177/1000
    13/13 [==============================] - 0s 2ms/step - loss: 0.1081
    Epoch 178/1000
    13/13 [==============================] - 0s 2ms/step - loss: 0.1245
    Epoch 179/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1346
    Epoch 180/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1233
    Epoch 181/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1113
    Epoch 182/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1040
    Epoch 183/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1155
    Epoch 184/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1049
    Epoch 185/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1111
    Epoch 186/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1079
    Epoch 187/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1021
    Epoch 188/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1048
    Epoch 189/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0971
    Epoch 190/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0985
    Epoch 191/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1026
    Epoch 192/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1111
    Epoch 193/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0991
    Epoch 194/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0890
    Epoch 195/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0880
    Epoch 196/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1006
    Epoch 197/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0974
    Epoch 198/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1141
    Epoch 199/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1423
    Epoch 200/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1381
    Epoch 201/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1105
    Epoch 202/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1005
    Epoch 203/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0846
    Epoch 204/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1125
    Epoch 205/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1129
    Epoch 206/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1219
    Epoch 207/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1161
    Epoch 208/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1137
    Epoch 209/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1178
    Epoch 210/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1017
    Epoch 211/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1051
    Epoch 212/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1014
    Epoch 213/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1096
    Epoch 214/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1087
    Epoch 215/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1047
    Epoch 216/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1044
    Epoch 217/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1044
    Epoch 218/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1006
    Epoch 219/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1093
    Epoch 220/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1041
    Epoch 221/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0956
    Epoch 222/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1109
    Epoch 223/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1041
    Epoch 224/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1000
    Epoch 225/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0968
    Epoch 226/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0951
    Epoch 227/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1092
    Epoch 228/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1041
    Epoch 229/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1032
    Epoch 230/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1153
    Epoch 231/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1237
    Epoch 232/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0978
    Epoch 233/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1074
    Epoch 234/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1059
    Epoch 235/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1122
    Epoch 236/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0974
    Epoch 237/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0879
    Epoch 238/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0913
    Epoch 239/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0831
    Epoch 240/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0752
    Epoch 241/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0733
    Epoch 242/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0886
    Epoch 243/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0837
    Epoch 244/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0866
    Epoch 245/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0933
    Epoch 246/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0976
    Epoch 247/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1150
    Epoch 248/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0904
    Epoch 249/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1073
    Epoch 250/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1296
    Epoch 251/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1022
    Epoch 252/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0987
    Epoch 253/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0846
    Epoch 254/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0813
    Epoch 255/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0924
    Epoch 256/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0799
    Epoch 257/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0947
    Epoch 258/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0956
    Epoch 259/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0788
    Epoch 260/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1018
    Epoch 261/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0942
    Epoch 262/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0780
    Epoch 263/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0821
    Epoch 264/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0795
    Epoch 265/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0924
    Epoch 266/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0948
    Epoch 267/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0767
    Epoch 268/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0720
    Epoch 269/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0742
    Epoch 270/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0747
    Epoch 271/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0726
    Epoch 272/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0984
    Epoch 273/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1074
    Epoch 274/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0836
    Epoch 275/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0783
    Epoch 276/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0799
    Epoch 277/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1225
    Epoch 278/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1017
    Epoch 279/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0990
    Epoch 280/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1014
    Epoch 281/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0808
    Epoch 282/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0798
    Epoch 283/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0847
    Epoch 284/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0755
    Epoch 285/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0631
    Epoch 286/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0651
    Epoch 287/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0602
    Epoch 288/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0733
    Epoch 289/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0659
    Epoch 290/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0682
    Epoch 291/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0745
    Epoch 292/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0848
    Epoch 293/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0701
    Epoch 294/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0828
    Epoch 295/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0741
    Epoch 296/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0890
    Epoch 297/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0800
    Epoch 298/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0803
    Epoch 299/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0765
    Epoch 300/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0733
    Epoch 301/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0544
    Epoch 302/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0718
    Epoch 303/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0877
    Epoch 304/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0687
    Epoch 305/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0671
    Epoch 306/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0575
    Epoch 307/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0773
    Epoch 308/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0779
    Epoch 309/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0696
    Epoch 310/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0883
    Epoch 311/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0880
    Epoch 312/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0707
    Epoch 313/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0603
    Epoch 314/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0772
    Epoch 315/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0660
    Epoch 316/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0586
    Epoch 317/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0618
    Epoch 318/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0588
    Epoch 319/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0674
    Epoch 320/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0598
    Epoch 321/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0670
    Epoch 322/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0970
    Epoch 323/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1366
    Epoch 324/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1148
    Epoch 325/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0837
    Epoch 326/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0749
    Epoch 327/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0746
    Epoch 328/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0698
    Epoch 329/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0691
    Epoch 330/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0541
    Epoch 331/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0558
    Epoch 332/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0653
    Epoch 333/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0593
    Epoch 334/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0606
    Epoch 335/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0696
    Epoch 336/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0713
    Epoch 337/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0628
    Epoch 338/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0752
    Epoch 339/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0723
    Epoch 340/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0647
    Epoch 341/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0688
    Epoch 342/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0793
    Epoch 343/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0595
    Epoch 344/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0528
    Epoch 345/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0552
    Epoch 346/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0534
    Epoch 347/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0471
    Epoch 348/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0491
    Epoch 349/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0524
    Epoch 350/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0696
    Epoch 351/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0690
    Epoch 352/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0864
    Epoch 353/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0999
    Epoch 354/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1094
    Epoch 355/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1189
    Epoch 356/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1059
    Epoch 357/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0655
    Epoch 358/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0652
    Epoch 359/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0544
    Epoch 360/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0545
    Epoch 361/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0549
    Epoch 362/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0581
    Epoch 363/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0506
    Epoch 364/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0579
    Epoch 365/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0583
    Epoch 366/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0607
    Epoch 367/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0428
    Epoch 368/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0495
    Epoch 369/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0721
    Epoch 370/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0817
    Epoch 371/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0588
    Epoch 372/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0516
    Epoch 373/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0526
    Epoch 374/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0463
    Epoch 375/1000
    13/13 [==============================] - 0s 2ms/step - loss: 0.0447
    Epoch 376/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0441
    Epoch 377/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0422
    Epoch 378/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0391
    Epoch 379/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0343
    Epoch 380/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0461
    Epoch 381/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0442
    Epoch 382/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0496
    Epoch 383/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0509
    Epoch 384/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0479
    Epoch 385/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0520
    Epoch 386/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0391
    Epoch 387/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0394
    Epoch 388/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0510
    Epoch 389/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0525
    Epoch 390/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0666
    Epoch 391/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0490
    Epoch 392/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0551
    Epoch 393/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0689
    Epoch 394/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0663
    Epoch 395/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0844
    Epoch 396/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0704
    Epoch 397/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0700
    Epoch 398/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0591
    Epoch 399/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0586
    Epoch 400/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0628
    Epoch 401/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1717
    Epoch 402/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1648
    Epoch 403/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1616
    Epoch 404/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1326
    Epoch 405/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1367
    Epoch 406/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1098
    Epoch 407/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1122
    Epoch 408/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1798
    Epoch 409/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1268
    Epoch 410/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1123
    Epoch 411/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0720
    Epoch 412/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0774
    Epoch 413/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0661
    Epoch 414/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0720
    Epoch 415/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0580
    Epoch 416/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0572
    Epoch 417/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0586
    Epoch 418/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0546
    Epoch 419/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0573
    Epoch 420/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0721
    Epoch 421/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0658
    Epoch 422/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0686
    Epoch 423/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0491
    Epoch 424/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0647
    Epoch 425/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0465
    Epoch 426/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0435
    Epoch 427/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0362
    Epoch 428/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0411
    Epoch 429/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0374
    Epoch 430/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0412
    Epoch 431/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0391
    Epoch 432/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0412
    Epoch 433/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0479
    Epoch 434/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0436
    Epoch 435/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0482
    Epoch 436/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0420
    Epoch 437/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0347
    Epoch 438/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0390
    Epoch 439/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0328
    Epoch 440/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0371
    Epoch 441/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0334
    Epoch 442/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0348
    Epoch 443/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0370
    Epoch 444/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0408
    Epoch 445/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0329
    Epoch 446/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0318
    Epoch 447/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0391
    Epoch 448/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0408
    Epoch 449/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0346
    Epoch 450/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0340
    Epoch 451/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0332
    Epoch 452/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0325
    Epoch 453/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0406
    Epoch 454/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0394
    Epoch 455/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0584
    Epoch 456/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0440
    Epoch 457/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0412
    Epoch 458/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0468
    Epoch 459/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0373
    Epoch 460/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0329
    Epoch 461/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0390
    Epoch 462/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0284
    Epoch 463/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0310
    Epoch 464/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0348
    Epoch 465/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0302
    Epoch 466/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0348
    Epoch 467/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0350
    Epoch 468/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0347
    Epoch 469/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0305
    Epoch 470/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0369
    Epoch 471/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0436
    Epoch 472/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0543
    Epoch 473/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0477
    Epoch 474/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0630
    Epoch 475/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1523
    Epoch 476/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3248
    Epoch 477/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1600
    Epoch 478/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1623
    Epoch 479/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1206
    Epoch 480/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0955
    Epoch 481/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1595
    Epoch 482/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1626
    Epoch 483/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1170
    Epoch 484/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1481
    Epoch 485/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0686
    Epoch 486/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0590
    Epoch 487/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0651
    Epoch 488/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0575
    Epoch 489/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0593
    Epoch 490/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0539
    Epoch 491/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0451
    Epoch 492/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0436
    Epoch 493/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0484
    Epoch 494/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0639
    Epoch 495/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0497
    Epoch 496/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0787
    Epoch 497/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0805
    Epoch 498/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0639
    Epoch 499/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0504
    Epoch 500/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0478
    Epoch 501/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0466
    Epoch 502/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0419
    Epoch 503/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0365
    Epoch 504/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0352
    Epoch 505/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0368
    Epoch 506/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0337
    Epoch 507/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0375
    Epoch 508/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0317
    Epoch 509/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0318
    Epoch 510/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0364
    Epoch 511/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0337
    Epoch 512/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0290
    Epoch 513/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0317
    Epoch 514/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0320
    Epoch 515/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0271
    Epoch 516/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0343
    Epoch 517/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0308
    Epoch 518/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0388
    Epoch 519/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0444
    Epoch 520/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0381
    Epoch 521/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0356
    Epoch 522/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0324
    Epoch 523/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0292
    Epoch 524/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0308
    Epoch 525/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0308
    Epoch 526/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0365
    Epoch 527/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0351
    Epoch 528/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0305
    Epoch 529/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0320
    Epoch 530/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0351
    Epoch 531/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0290
    Epoch 532/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0329
    Epoch 533/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0387
    Epoch 534/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0431
    Epoch 535/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0414
    Epoch 536/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0318
    Epoch 537/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0285
    Epoch 538/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0278
    Epoch 539/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0274
    Epoch 540/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0338
    Epoch 541/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0262
    Epoch 542/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0283
    Epoch 543/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0265
    Epoch 544/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0267
    Epoch 545/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0278
    Epoch 546/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0256
    Epoch 547/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0302
    Epoch 548/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0323
    Epoch 549/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0262
    Epoch 550/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0288
    Epoch 551/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0283
    Epoch 552/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0315
    Epoch 553/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0411
    Epoch 554/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0376
    Epoch 555/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0346
    Epoch 556/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0296
    Epoch 557/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0307
    Epoch 558/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0270
    Epoch 559/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0268
    Epoch 560/1000
    13/13 [==============================] - 0s 2ms/step - loss: 0.0303
    Epoch 561/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0251
    Epoch 562/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0267
    Epoch 563/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0249
    Epoch 564/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0265
    Epoch 565/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0297
    Epoch 566/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0338
    Epoch 567/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0432
    Epoch 568/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0483
    Epoch 569/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1205
    Epoch 570/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1063
    Epoch 571/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1035
    Epoch 572/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1415
    Epoch 573/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1534
    Epoch 574/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1474
    Epoch 575/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0772
    Epoch 576/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0691
    Epoch 577/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0770
    Epoch 578/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0637
    Epoch 579/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0528
    Epoch 580/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0371
    Epoch 581/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0356
    Epoch 582/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0431
    Epoch 583/1000
    13/13 [==============================] - 0s 2ms/step - loss: 0.0300
    Epoch 584/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0309
    Epoch 585/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0307
    Epoch 586/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0321
    Epoch 587/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0266
    Epoch 588/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0274
    Epoch 589/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0276
    Epoch 590/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0267
    Epoch 591/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0305
    Epoch 592/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0278
    Epoch 593/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0343
    Epoch 594/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0259
    Epoch 595/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0259
    Epoch 596/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0258
    Epoch 597/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0262
    Epoch 598/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0254
    Epoch 599/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0251
    Epoch 600/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0241
    Epoch 601/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0269
    Epoch 602/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0287
    Epoch 603/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0257
    Epoch 604/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0254
    Epoch 605/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0232
    Epoch 606/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0281
    Epoch 607/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0247
    Epoch 608/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0254
    Epoch 609/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0237
    Epoch 610/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0253
    Epoch 611/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0256
    Epoch 612/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0235
    Epoch 613/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0290
    Epoch 614/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0236
    Epoch 615/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0249
    Epoch 616/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0253
    Epoch 617/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0231
    Epoch 618/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0241
    Epoch 619/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0253
    Epoch 620/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0290
    Epoch 621/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0456
    Epoch 622/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0647
    Epoch 623/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1078
    Epoch 624/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1180
    Epoch 625/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0837
    Epoch 626/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0510
    Epoch 627/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0333
    Epoch 628/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0327
    Epoch 629/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0389
    Epoch 630/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0347
    Epoch 631/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0342
    Epoch 632/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0272
    Epoch 633/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0240
    Epoch 634/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0235
    Epoch 635/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0243
    Epoch 636/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0225
    Epoch 637/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0222
    Epoch 638/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0223
    Epoch 639/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0215
    Epoch 640/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0247
    Epoch 641/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0248
    Epoch 642/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0257
    Epoch 643/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0213
    Epoch 644/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0277
    Epoch 645/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0266
    Epoch 646/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0320
    Epoch 647/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0269
    Epoch 648/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0357
    Epoch 649/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0321
    Epoch 650/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0255
    Epoch 651/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0287
    Epoch 652/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0251
    Epoch 653/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0242
    Epoch 654/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0239
    Epoch 655/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0218
    Epoch 656/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0227
    Epoch 657/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0247
    Epoch 658/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0265
    Epoch 659/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0257
    Epoch 660/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0233
    Epoch 661/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0246
    Epoch 662/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0313
    Epoch 663/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0238
    Epoch 664/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0277
    Epoch 665/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0205
    Epoch 666/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0238
    Epoch 667/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0249
    Epoch 668/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0441
    Epoch 669/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0441
    Epoch 670/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0305
    Epoch 671/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0323
    Epoch 672/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0356
    Epoch 673/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0670
    Epoch 674/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1732
    Epoch 675/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0889
    Epoch 676/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1098
    Epoch 677/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0468
    Epoch 678/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0532
    Epoch 679/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0577
    Epoch 680/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0880
    Epoch 681/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1123
    Epoch 682/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1581
    Epoch 683/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1343
    Epoch 684/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1065
    Epoch 685/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1236
    Epoch 686/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1184
    Epoch 687/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1218
    Epoch 688/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1673
    Epoch 689/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1437
    Epoch 690/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0897
    Epoch 691/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0665
    Epoch 692/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0579
    Epoch 693/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0563
    Epoch 694/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0425
    Epoch 695/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0441
    Epoch 696/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0411
    Epoch 697/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0429
    Epoch 698/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0347
    Epoch 699/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0367
    Epoch 700/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0311
    Epoch 701/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0333
    Epoch 702/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0308
    Epoch 703/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0287
    Epoch 704/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0297
    Epoch 705/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0282
    Epoch 706/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0263
    Epoch 707/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0286
    Epoch 708/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0275
    Epoch 709/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0274
    Epoch 710/1000
    13/13 [==============================] - 0s 2ms/step - loss: 0.0252
    Epoch 711/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0277
    Epoch 712/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0261
    Epoch 713/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0311
    Epoch 714/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0265
    Epoch 715/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0281
    Epoch 716/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0275
    Epoch 717/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0264
    Epoch 718/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0240
    Epoch 719/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0234
    Epoch 720/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0284
    Epoch 721/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0311
    Epoch 722/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0244
    Epoch 723/1000
    13/13 [==============================] - 0s 2ms/step - loss: 0.0249
    Epoch 724/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0269
    Epoch 725/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0224
    Epoch 726/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0238
    Epoch 727/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0234
    Epoch 728/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0223
    Epoch 729/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0220
    Epoch 730/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0268
    Epoch 731/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0363
    Epoch 732/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0300
    Epoch 733/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0208
    Epoch 734/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0254
    Epoch 735/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0264
    Epoch 736/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0230
    Epoch 737/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0224
    Epoch 738/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0270
    Epoch 739/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0257
    Epoch 740/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0228
    Epoch 741/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0249
    Epoch 742/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0241
    Epoch 743/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0210
    Epoch 744/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0216
    Epoch 745/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0208
    Epoch 746/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0227
    Epoch 747/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0193
    Epoch 748/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0241
    Epoch 749/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0217
    Epoch 750/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0248
    Epoch 751/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0203
    Epoch 752/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0194
    Epoch 753/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0252
    Epoch 754/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0203
    Epoch 755/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0206
    Epoch 756/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0192
    Epoch 757/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0213
    Epoch 758/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0206
    Epoch 759/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0247
    Epoch 760/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0227
    Epoch 761/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0204
    Epoch 762/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0219
    Epoch 763/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0266
    Epoch 764/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0699
    Epoch 765/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0436
    Epoch 766/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0451
    Epoch 767/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1029
    Epoch 768/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1082
    Epoch 769/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0924
    Epoch 770/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0936
    Epoch 771/1000
    13/13 [==============================] - 0s 2ms/step - loss: 0.0690
    Epoch 772/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0589
    Epoch 773/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0519
    Epoch 774/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0714
    Epoch 775/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1015
    Epoch 776/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0932
    Epoch 777/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1891
    Epoch 778/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1356
    Epoch 779/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1081
    Epoch 780/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0973
    Epoch 781/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0768
    Epoch 782/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0761
    Epoch 783/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1075
    Epoch 784/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0789
    Epoch 785/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0467
    Epoch 786/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0394
    Epoch 787/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0360
    Epoch 788/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0324
    Epoch 789/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0329
    Epoch 790/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0291
    Epoch 791/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0283
    Epoch 792/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0291
    Epoch 793/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0261
    Epoch 794/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0294
    Epoch 795/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0250
    Epoch 796/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0292
    Epoch 797/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0286
    Epoch 798/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0271
    Epoch 799/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0307
    Epoch 800/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0298
    Epoch 801/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0371
    Epoch 802/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0259
    Epoch 803/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0274
    Epoch 804/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0266
    Epoch 805/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0260
    Epoch 806/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0254
    Epoch 807/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0258
    Epoch 808/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0252
    Epoch 809/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0280
    Epoch 810/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0249
    Epoch 811/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0255
    Epoch 812/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0259
    Epoch 813/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0310
    Epoch 814/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0258
    Epoch 815/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0246
    Epoch 816/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0263
    Epoch 817/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0328
    Epoch 818/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0247
    Epoch 819/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0250
    Epoch 820/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0258
    Epoch 821/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0252
    Epoch 822/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0256
    Epoch 823/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0299
    Epoch 824/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0312
    Epoch 825/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0243
    Epoch 826/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0263
    Epoch 827/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0247
    Epoch 828/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0233
    Epoch 829/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0246
    Epoch 830/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0262
    Epoch 831/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0259
    Epoch 832/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0238
    Epoch 833/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0221
    Epoch 834/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0240
    Epoch 835/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0248
    Epoch 836/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0253
    Epoch 837/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0340
    Epoch 838/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0229
    Epoch 839/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0294
    Epoch 840/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0286
    Epoch 841/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0268
    Epoch 842/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0283
    Epoch 843/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0271
    Epoch 844/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0247
    Epoch 845/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0235
    Epoch 846/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0300
    Epoch 847/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0246
    Epoch 848/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0244
    Epoch 849/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0219
    Epoch 850/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0258
    Epoch 851/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0244
    Epoch 852/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0257
    Epoch 853/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0220
    Epoch 854/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0221
    Epoch 855/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0256
    Epoch 856/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0211
    Epoch 857/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0227
    Epoch 858/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0252
    Epoch 859/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0224
    Epoch 860/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0214
    Epoch 861/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0204
    Epoch 862/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0228
    Epoch 863/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0206
    Epoch 864/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0198
    Epoch 865/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0200
    Epoch 866/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0273
    Epoch 867/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0271
    Epoch 868/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0217
    Epoch 869/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0231
    Epoch 870/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0325
    Epoch 871/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0354
    Epoch 872/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0321
    Epoch 873/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0216
    Epoch 874/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0201
    Epoch 875/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0218
    Epoch 876/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0217
    Epoch 877/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0275
    Epoch 878/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0305
    Epoch 879/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0440
    Epoch 880/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0466
    Epoch 881/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0729
    Epoch 882/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0460
    Epoch 883/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0439
    Epoch 884/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0811
    Epoch 885/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0291
    Epoch 886/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0309
    Epoch 887/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0289
    Epoch 888/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0294
    Epoch 889/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0283
    Epoch 890/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0240
    Epoch 891/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0232
    Epoch 892/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0225
    Epoch 893/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0196
    Epoch 894/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0218
    Epoch 895/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0189
    Epoch 896/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0221
    Epoch 897/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0204
    Epoch 898/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0200
    Epoch 899/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0208
    Epoch 900/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0205
    Epoch 901/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0199
    Epoch 902/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0298
    Epoch 903/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0185
    Epoch 904/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0290
    Epoch 905/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0272
    Epoch 906/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0237
    Epoch 907/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0190
    Epoch 908/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0210
    Epoch 909/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0189
    Epoch 910/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0199
    Epoch 911/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0688
    Epoch 912/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1337
    Epoch 913/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1883
    Epoch 914/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2096
    Epoch 915/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1323
    Epoch 916/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0795
    Epoch 917/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1167
    Epoch 918/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0621
    Epoch 919/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0929
    Epoch 920/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0352
    Epoch 921/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0303
    Epoch 922/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0287
    Epoch 923/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0457
    Epoch 924/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0712
    Epoch 925/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0553
    Epoch 926/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0385
    Epoch 927/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0311
    Epoch 928/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0394
    Epoch 929/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0261
    Epoch 930/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0346
    Epoch 931/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0332
    Epoch 932/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0322
    Epoch 933/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0311
    Epoch 934/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0493
    Epoch 935/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0289
    Epoch 936/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0325
    Epoch 937/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0255
    Epoch 938/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0210
    Epoch 939/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0235
    Epoch 940/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0259
    Epoch 941/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0371
    Epoch 942/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0300
    Epoch 943/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0265
    Epoch 944/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0327
    Epoch 945/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0367
    Epoch 946/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0307
    Epoch 947/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0376
    Epoch 948/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0375
    Epoch 949/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0350
    Epoch 950/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0284
    Epoch 951/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0293
    Epoch 952/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0374
    Epoch 953/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0353
    Epoch 954/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0395
    Epoch 955/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0405
    Epoch 956/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0432
    Epoch 957/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0234
    Epoch 958/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0266
    Epoch 959/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0213
    Epoch 960/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0200
    Epoch 961/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0203
    Epoch 962/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0190
    Epoch 963/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0239
    Epoch 964/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0240
    Epoch 965/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0261
    Epoch 966/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0197
    Epoch 967/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0206
    Epoch 968/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0188
    Epoch 969/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0200
    Epoch 970/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0169
    Epoch 971/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0161
    Epoch 972/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0176
    Epoch 973/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0218
    Epoch 974/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0161
    Epoch 975/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0203
    Epoch 976/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0384
    Epoch 977/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0292
    Epoch 978/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0234
    Epoch 979/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0522
    Epoch 980/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0851
    Epoch 981/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0541
    Epoch 982/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0380
    Epoch 983/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0328
    Epoch 984/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0276
    Epoch 985/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0227
    Epoch 986/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0235
    Epoch 987/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0287
    Epoch 988/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0170
    Epoch 989/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0166
    Epoch 990/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0175
    Epoch 991/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0149
    Epoch 992/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0152
    Epoch 993/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0153
    Epoch 994/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0142
    Epoch 995/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0199
    Epoch 996/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0231
    Epoch 997/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0199
    Epoch 998/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.0188
    Epoch 999/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0155
    Epoch 1000/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.0172
    Finished lambda = 0.0
    Epoch 1/1000
    13/13 [==============================] - 0s 1ms/step - loss: 1.1055
    Epoch 2/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4858
    Epoch 3/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4067
    Epoch 4/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3608
    Epoch 5/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3565
    Epoch 6/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3595
    Epoch 7/1000
    13/13 [==============================] - 0s 4ms/step - loss: 0.3211
    Epoch 8/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3000
    Epoch 9/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2910
    Epoch 10/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2648
    Epoch 11/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.2734
    Epoch 12/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2646
    Epoch 13/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2929
    Epoch 14/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2762
    Epoch 15/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3013
    Epoch 16/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2616
    Epoch 17/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2628
    Epoch 18/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2574
    Epoch 19/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2740
    Epoch 20/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2536
    Epoch 21/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2622
    Epoch 22/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2747
    Epoch 23/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2742
    Epoch 24/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.2539
    Epoch 25/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2712
    Epoch 26/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2506
    Epoch 27/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2506
    Epoch 28/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.2504
    Epoch 29/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2647
    Epoch 30/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2773
    Epoch 31/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2587
    Epoch 32/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2579
    Epoch 33/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.2446
    Epoch 34/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2647
    Epoch 35/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2664
    Epoch 36/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2432
    Epoch 37/1000
    13/13 [==============================] - 0s 4ms/step - loss: 0.2508
    Epoch 38/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2304
    Epoch 39/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2398
    Epoch 40/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2355
    Epoch 41/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.2703
    Epoch 42/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2665
    Epoch 43/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2429
    Epoch 44/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2581
    Epoch 45/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2825
    Epoch 46/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.2437
    Epoch 47/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2321
    Epoch 48/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2325
    Epoch 49/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2283
    Epoch 50/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.2255
    Epoch 51/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2451
    Epoch 52/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2366
    Epoch 53/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2477
    Epoch 54/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.2280
    Epoch 55/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2741
    Epoch 56/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2435
    Epoch 57/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2698
    Epoch 58/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2489
    Epoch 59/1000
    13/13 [==============================] - 0s 4ms/step - loss: 0.2588
    Epoch 60/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2569
    Epoch 61/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2475
    Epoch 62/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2257
    Epoch 63/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.2267
    Epoch 64/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2697
    Epoch 65/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2643
    Epoch 66/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2571
    Epoch 67/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.2815
    Epoch 68/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2878
    Epoch 69/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2394
    Epoch 70/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2338
    Epoch 71/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2546
    Epoch 72/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.2465
    Epoch 73/1000
    13/13 [==============================] - 0s 2ms/step - loss: 0.2550
    Epoch 74/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2502
    Epoch 75/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2468
    Epoch 76/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.2304
    Epoch 77/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2368
    Epoch 78/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2341
    Epoch 79/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2314
    Epoch 80/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.2368
    Epoch 81/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2401
    Epoch 82/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2478
    Epoch 83/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2346
    Epoch 84/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.2324
    Epoch 85/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2536
    Epoch 86/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2255
    Epoch 87/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2297
    Epoch 88/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.2306
    Epoch 89/1000
    13/13 [==============================] - 0s 2ms/step - loss: 0.2300
    Epoch 90/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2262
    Epoch 91/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2189
    Epoch 92/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2184
    Epoch 93/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.2201
    Epoch 94/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2176
    Epoch 95/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2427
    Epoch 96/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2451
    Epoch 97/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.2428
    Epoch 98/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2501
    Epoch 99/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2412
    Epoch 100/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2254
    Epoch 101/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.2411
    Epoch 102/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2359
    Epoch 103/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2533
    Epoch 104/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2353
    Epoch 105/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.2218
    Epoch 106/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2232
    Epoch 107/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2330
    Epoch 108/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2145
    Epoch 109/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2194
    Epoch 110/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.2264
    Epoch 111/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2220
    Epoch 112/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2372
    Epoch 113/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2271
    Epoch 114/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.2141
    Epoch 115/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2125
    Epoch 116/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2254
    Epoch 117/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2180
    Epoch 118/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.2147
    Epoch 119/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2193
    Epoch 120/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2160
    Epoch 121/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2220
    Epoch 122/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2230
    Epoch 123/1000
    13/13 [==============================] - 0s 2ms/step - loss: 0.2187
    Epoch 124/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2099
    Epoch 125/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2094
    Epoch 126/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2328
    Epoch 127/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2216
    Epoch 128/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2138
    Epoch 129/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2163
    Epoch 130/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2168
    Epoch 131/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.2217
    Epoch 132/1000
    13/13 [==============================] - 0s 2ms/step - loss: 0.2144
    Epoch 133/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2035
    Epoch 134/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2080
    Epoch 135/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.2124
    Epoch 136/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2100
    Epoch 137/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2188
    Epoch 138/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2184
    Epoch 139/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.2081
    Epoch 140/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2205
    Epoch 141/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2095
    Epoch 142/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2068
    Epoch 143/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2077
    Epoch 144/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2204
    Epoch 145/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2201
    Epoch 146/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2146
    Epoch 147/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2133
    Epoch 148/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.2092
    Epoch 149/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2115
    Epoch 150/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2234
    Epoch 151/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2182
    Epoch 152/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.2191
    Epoch 153/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2164
    Epoch 154/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2110
    Epoch 155/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2173
    Epoch 156/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.2131
    Epoch 157/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2189
    Epoch 158/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2196
    Epoch 159/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2014
    Epoch 160/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2044
    Epoch 161/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2024
    Epoch 162/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2071
    Epoch 163/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2171
    Epoch 164/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2202
    Epoch 165/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.2135
    Epoch 166/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2072
    Epoch 167/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2135
    Epoch 168/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2180
    Epoch 169/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.2160
    Epoch 170/1000
    13/13 [==============================] - 0s 2ms/step - loss: 0.2288
    Epoch 171/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2118
    Epoch 172/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2039
    Epoch 173/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.2185
    Epoch 174/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2109
    Epoch 175/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1978
    Epoch 176/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2058
    Epoch 177/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.2035
    Epoch 178/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2134
    Epoch 179/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2120
    Epoch 180/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2032
    Epoch 181/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.2208
    Epoch 182/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2110
    Epoch 183/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2122
    Epoch 184/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2098
    Epoch 185/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2084
    Epoch 186/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1989
    Epoch 187/1000
    13/13 [==============================] - 0s 2ms/step - loss: 0.2045
    Epoch 188/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2013
    Epoch 189/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2033
    Epoch 190/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2129
    Epoch 191/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2110
    Epoch 192/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2040
    Epoch 193/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2081
    Epoch 194/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.2046
    Epoch 195/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1934
    Epoch 196/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1943
    Epoch 197/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2082
    Epoch 198/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.2047
    Epoch 199/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2199
    Epoch 200/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2055
    Epoch 201/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1974
    Epoch 202/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1992
    Epoch 203/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1944
    Epoch 204/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2274
    Epoch 205/1000
    13/13 [==============================] - 0s 2ms/step - loss: 0.1976
    Epoch 206/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1989
    Epoch 207/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2099
    Epoch 208/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2115
    Epoch 209/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1963
    Epoch 210/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2017
    Epoch 211/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2062
    Epoch 212/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2089
    Epoch 213/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2148
    Epoch 214/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2068
    Epoch 215/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.2078
    Epoch 216/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2014
    Epoch 217/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2152
    Epoch 218/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2124
    Epoch 219/1000
    13/13 [==============================] - 0s 4ms/step - loss: 0.2030
    Epoch 220/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2046
    Epoch 221/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1955
    Epoch 222/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1952
    Epoch 223/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.2066
    Epoch 224/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2206
    Epoch 225/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2010
    Epoch 226/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1978
    Epoch 227/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1886
    Epoch 228/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1925
    Epoch 229/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1953
    Epoch 230/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2075
    Epoch 231/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.2373
    Epoch 232/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2167
    Epoch 233/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2053
    Epoch 234/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1968
    Epoch 235/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.2008
    Epoch 236/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1953
    Epoch 237/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1950
    Epoch 238/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2228
    Epoch 239/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2118
    Epoch 240/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2069
    Epoch 241/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2012
    Epoch 242/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2153
    Epoch 243/1000
    13/13 [==============================] - 0s 2ms/step - loss: 0.2126
    Epoch 244/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2101
    Epoch 245/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1979
    Epoch 246/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1987
    Epoch 247/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1946
    Epoch 248/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1889
    Epoch 249/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1984
    Epoch 250/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1975
    Epoch 251/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1919
    Epoch 252/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1904
    Epoch 253/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1942
    Epoch 254/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2016
    Epoch 255/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1996
    Epoch 256/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1887
    Epoch 257/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2110
    Epoch 258/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2040
    Epoch 259/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1890
    Epoch 260/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1960
    Epoch 261/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.2038
    Epoch 262/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1948
    Epoch 263/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1931
    Epoch 264/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1913
    Epoch 265/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1912
    Epoch 266/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1895
    Epoch 267/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1990
    Epoch 268/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1895
    Epoch 269/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1909
    Epoch 270/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1946
    Epoch 271/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1935
    Epoch 272/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1962
    Epoch 273/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.2024
    Epoch 274/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1985
    Epoch 275/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2075
    Epoch 276/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1942
    Epoch 277/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1964
    Epoch 278/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1922
    Epoch 279/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2000
    Epoch 280/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1983
    Epoch 281/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1969
    Epoch 282/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1901
    Epoch 283/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1988
    Epoch 284/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1857
    Epoch 285/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1860
    Epoch 286/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1946
    Epoch 287/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1907
    Epoch 288/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2126
    Epoch 289/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2023
    Epoch 290/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1985
    Epoch 291/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1901
    Epoch 292/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1820
    Epoch 293/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1869
    Epoch 294/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1866
    Epoch 295/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1950
    Epoch 296/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1952
    Epoch 297/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1884
    Epoch 298/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.2042
    Epoch 299/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1900
    Epoch 300/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1985
    Epoch 301/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2013
    Epoch 302/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.2040
    Epoch 303/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2127
    Epoch 304/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1954
    Epoch 305/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1994
    Epoch 306/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1881
    Epoch 307/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1973
    Epoch 308/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1940
    Epoch 309/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1895
    Epoch 310/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1879
    Epoch 311/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1832
    Epoch 312/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1879
    Epoch 313/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1920
    Epoch 314/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1844
    Epoch 315/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1927
    Epoch 316/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1871
    Epoch 317/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1866
    Epoch 318/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2143
    Epoch 319/1000
    13/13 [==============================] - 0s 4ms/step - loss: 0.1956
    Epoch 320/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1846
    Epoch 321/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1823
    Epoch 322/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1892
    Epoch 323/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.2019
    Epoch 324/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1810
    Epoch 325/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1885
    Epoch 326/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1797
    Epoch 327/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1900
    Epoch 328/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1975
    Epoch 329/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1947
    Epoch 330/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1871
    Epoch 331/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1923
    Epoch 332/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1922
    Epoch 333/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1962
    Epoch 334/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2092
    Epoch 335/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2044
    Epoch 336/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1936
    Epoch 337/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1886
    Epoch 338/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1911
    Epoch 339/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1960
    Epoch 340/1000
    13/13 [==============================] - 0s 4ms/step - loss: 0.1932
    Epoch 341/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1838
    Epoch 342/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1832
    Epoch 343/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1799
    Epoch 344/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1793
    Epoch 345/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1950
    Epoch 346/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1947
    Epoch 347/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1916
    Epoch 348/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1930
    Epoch 349/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1804
    Epoch 350/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1765
    Epoch 351/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1839
    Epoch 352/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1919
    Epoch 353/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1982
    Epoch 354/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1934
    Epoch 355/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1957
    Epoch 356/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1822
    Epoch 357/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1815
    Epoch 358/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1859
    Epoch 359/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1802
    Epoch 360/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1887
    Epoch 361/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1839
    Epoch 362/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.2091
    Epoch 363/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1962
    Epoch 364/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1910
    Epoch 365/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1972
    Epoch 366/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1994
    Epoch 367/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1840
    Epoch 368/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1756
    Epoch 369/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1775
    Epoch 370/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1890
    Epoch 371/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1755
    Epoch 372/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1778
    Epoch 373/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1861
    Epoch 374/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1799
    Epoch 375/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1866
    Epoch 376/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1800
    Epoch 377/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1793
    Epoch 378/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1850
    Epoch 379/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1849
    Epoch 380/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1833
    Epoch 381/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1820
    Epoch 382/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1913
    Epoch 383/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.2015
    Epoch 384/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1958
    Epoch 385/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1810
    Epoch 386/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1816
    Epoch 387/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1793
    Epoch 388/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1775
    Epoch 389/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1880
    Epoch 390/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1937
    Epoch 391/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1957
    Epoch 392/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1833
    Epoch 393/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1794
    Epoch 394/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1742
    Epoch 395/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1827
    Epoch 396/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1841
    Epoch 397/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1877
    Epoch 398/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1852
    Epoch 399/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1804
    Epoch 400/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1793
    Epoch 401/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1811
    Epoch 402/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1814
    Epoch 403/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1797
    Epoch 404/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1781
    Epoch 405/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1703
    Epoch 406/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1769
    Epoch 407/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1819
    Epoch 408/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1797
    Epoch 409/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1801
    Epoch 410/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1771
    Epoch 411/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1905
    Epoch 412/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1850
    Epoch 413/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1878
    Epoch 414/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1764
    Epoch 415/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1702
    Epoch 416/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1813
    Epoch 417/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1852
    Epoch 418/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1756
    Epoch 419/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1730
    Epoch 420/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1734
    Epoch 421/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1806
    Epoch 422/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1750
    Epoch 423/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1810
    Epoch 424/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1855
    Epoch 425/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1778
    Epoch 426/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1800
    Epoch 427/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1719
    Epoch 428/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1764
    Epoch 429/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1678
    Epoch 430/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1733
    Epoch 431/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1739
    Epoch 432/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1725
    Epoch 433/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1822
    Epoch 434/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1725
    Epoch 435/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1819
    Epoch 436/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1952
    Epoch 437/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1776
    Epoch 438/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1788
    Epoch 439/1000
    13/13 [==============================] - 0s 4ms/step - loss: 0.1701
    Epoch 440/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1785
    Epoch 441/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1721
    Epoch 442/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1750
    Epoch 443/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1800
    Epoch 444/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1697
    Epoch 445/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1742
    Epoch 446/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1905
    Epoch 447/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1883
    Epoch 448/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1722
    Epoch 449/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1719
    Epoch 450/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1731
    Epoch 451/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1819
    Epoch 452/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1721
    Epoch 453/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1752
    Epoch 454/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1742
    Epoch 455/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1744
    Epoch 456/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1711
    Epoch 457/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1748
    Epoch 458/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1788
    Epoch 459/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1799
    Epoch 460/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1711
    Epoch 461/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1777
    Epoch 462/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1691
    Epoch 463/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1777
    Epoch 464/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1802
    Epoch 465/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1750
    Epoch 466/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1755
    Epoch 467/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1792
    Epoch 468/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1744
    Epoch 469/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1654
    Epoch 470/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1720
    Epoch 471/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1833
    Epoch 472/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1737
    Epoch 473/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1689
    Epoch 474/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1731
    Epoch 475/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1811
    Epoch 476/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1756
    Epoch 477/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1820
    Epoch 478/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1972
    Epoch 479/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1826
    Epoch 480/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1786
    Epoch 481/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1727
    Epoch 482/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1761
    Epoch 483/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1784
    Epoch 484/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1865
    Epoch 485/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1801
    Epoch 486/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1695
    Epoch 487/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1741
    Epoch 488/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1731
    Epoch 489/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1732
    Epoch 490/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1808
    Epoch 491/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1713
    Epoch 492/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1770
    Epoch 493/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1698
    Epoch 494/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1808
    Epoch 495/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1732
    Epoch 496/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1892
    Epoch 497/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1761
    Epoch 498/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1763
    Epoch 499/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1737
    Epoch 500/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1731
    Epoch 501/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1714
    Epoch 502/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1789
    Epoch 503/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1904
    Epoch 504/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1732
    Epoch 505/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1770
    Epoch 506/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1708
    Epoch 507/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1765
    Epoch 508/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1627
    Epoch 509/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1732
    Epoch 510/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1714
    Epoch 511/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1680
    Epoch 512/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1712
    Epoch 513/1000
    13/13 [==============================] - 0s 4ms/step - loss: 0.1709
    Epoch 514/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1726
    Epoch 515/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1670
    Epoch 516/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1681
    Epoch 517/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1617
    Epoch 518/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1680
    Epoch 519/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1839
    Epoch 520/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1735
    Epoch 521/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1882
    Epoch 522/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1784
    Epoch 523/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1759
    Epoch 524/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1657
    Epoch 525/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1669
    Epoch 526/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1650
    Epoch 527/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1754
    Epoch 528/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1749
    Epoch 529/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1827
    Epoch 530/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1671
    Epoch 531/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1650
    Epoch 532/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1651
    Epoch 533/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1686
    Epoch 534/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1726
    Epoch 535/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1681
    Epoch 536/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1750
    Epoch 537/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1724
    Epoch 538/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1626
    Epoch 539/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1664
    Epoch 540/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1730
    Epoch 541/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1869
    Epoch 542/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1679
    Epoch 543/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1635
    Epoch 544/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1683
    Epoch 545/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1704
    Epoch 546/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1692
    Epoch 547/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1742
    Epoch 548/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1743
    Epoch 549/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1815
    Epoch 550/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1871
    Epoch 551/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1805
    Epoch 552/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1800
    Epoch 553/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1729
    Epoch 554/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1697
    Epoch 555/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1675
    Epoch 556/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1669
    Epoch 557/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1669
    Epoch 558/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1689
    Epoch 559/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1717
    Epoch 560/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1732
    Epoch 561/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1760
    Epoch 562/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1676
    Epoch 563/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1701
    Epoch 564/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1608
    Epoch 565/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1627
    Epoch 566/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1692
    Epoch 567/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1792
    Epoch 568/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1907
    Epoch 569/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1717
    Epoch 570/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1712
    Epoch 571/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1690
    Epoch 572/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1726
    Epoch 573/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1668
    Epoch 574/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1646
    Epoch 575/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1633
    Epoch 576/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1623
    Epoch 577/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1674
    Epoch 578/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1770
    Epoch 579/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1626
    Epoch 580/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1734
    Epoch 581/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1641
    Epoch 582/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1642
    Epoch 583/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1660
    Epoch 584/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1715
    Epoch 585/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1646
    Epoch 586/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1651
    Epoch 587/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1687
    Epoch 588/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1636
    Epoch 589/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1671
    Epoch 590/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1664
    Epoch 591/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1848
    Epoch 592/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1959
    Epoch 593/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1758
    Epoch 594/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1660
    Epoch 595/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1688
    Epoch 596/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1687
    Epoch 597/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1634
    Epoch 598/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1754
    Epoch 599/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1679
    Epoch 600/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1763
    Epoch 601/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1678
    Epoch 602/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1666
    Epoch 603/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1664
    Epoch 604/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1667
    Epoch 605/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1640
    Epoch 606/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1624
    Epoch 607/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1660
    Epoch 608/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1686
    Epoch 609/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1678
    Epoch 610/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1728
    Epoch 611/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1634
    Epoch 612/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1657
    Epoch 613/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1728
    Epoch 614/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1716
    Epoch 615/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1674
    Epoch 616/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1601
    Epoch 617/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1609
    Epoch 618/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1690
    Epoch 619/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1643
    Epoch 620/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1705
    Epoch 621/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1785
    Epoch 622/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1632
    Epoch 623/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1720
    Epoch 624/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1634
    Epoch 625/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1655
    Epoch 626/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1685
    Epoch 627/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1638
    Epoch 628/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1630
    Epoch 629/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1606
    Epoch 630/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1633
    Epoch 631/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1651
    Epoch 632/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1616
    Epoch 633/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1572
    Epoch 634/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1576
    Epoch 635/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1616
    Epoch 636/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1724
    Epoch 637/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1782
    Epoch 638/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1818
    Epoch 639/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1664
    Epoch 640/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1669
    Epoch 641/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1659
    Epoch 642/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1644
    Epoch 643/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1588
    Epoch 644/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1608
    Epoch 645/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1689
    Epoch 646/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1672
    Epoch 647/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1631
    Epoch 648/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1641
    Epoch 649/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1597
    Epoch 650/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1691
    Epoch 651/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1637
    Epoch 652/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1590
    Epoch 653/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1591
    Epoch 654/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1654
    Epoch 655/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1628
    Epoch 656/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1575
    Epoch 657/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1627
    Epoch 658/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1637
    Epoch 659/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1640
    Epoch 660/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1630
    Epoch 661/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1638
    Epoch 662/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1606
    Epoch 663/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1622
    Epoch 664/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1666
    Epoch 665/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1570
    Epoch 666/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1608
    Epoch 667/1000
    13/13 [==============================] - 0s 2ms/step - loss: 0.1652
    Epoch 668/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1617
    Epoch 669/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1643
    Epoch 670/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1612
    Epoch 671/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1640
    Epoch 672/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1693
    Epoch 673/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1591
    Epoch 674/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1650
    Epoch 675/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1610
    Epoch 676/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1566
    Epoch 677/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1623
    Epoch 678/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1564
    Epoch 679/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1675
    Epoch 680/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1742
    Epoch 681/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1790
    Epoch 682/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1779
    Epoch 683/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1753
    Epoch 684/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1692
    Epoch 685/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1788
    Epoch 686/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1874
    Epoch 687/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1708
    Epoch 688/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1683
    Epoch 689/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1634
    Epoch 690/1000
    13/13 [==============================] - 0s 4ms/step - loss: 0.1617
    Epoch 691/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1640
    Epoch 692/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1608
    Epoch 693/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1657
    Epoch 694/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1631
    Epoch 695/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1630
    Epoch 696/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1552
    Epoch 697/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1622
    Epoch 698/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1611
    Epoch 699/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1574
    Epoch 700/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1555
    Epoch 701/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1579
    Epoch 702/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1627
    Epoch 703/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1560
    Epoch 704/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1624
    Epoch 705/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1583
    Epoch 706/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1563
    Epoch 707/1000
    13/13 [==============================] - 0s 4ms/step - loss: 0.1612
    Epoch 708/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1604
    Epoch 709/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1750
    Epoch 710/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1753
    Epoch 711/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1708
    Epoch 712/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1668
    Epoch 713/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1671
    Epoch 714/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1543
    Epoch 715/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1748
    Epoch 716/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1604
    Epoch 717/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1651
    Epoch 718/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1606
    Epoch 719/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1602
    Epoch 720/1000
    13/13 [==============================] - 0s 4ms/step - loss: 0.1585
    Epoch 721/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1671
    Epoch 722/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1553
    Epoch 723/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1588
    Epoch 724/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1534
    Epoch 725/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1654
    Epoch 726/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1566
    Epoch 727/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1590
    Epoch 728/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1607
    Epoch 729/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1564
    Epoch 730/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1612
    Epoch 731/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1658
    Epoch 732/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1772
    Epoch 733/1000
    13/13 [==============================] - 0s 4ms/step - loss: 0.1629
    Epoch 734/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1657
    Epoch 735/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1574
    Epoch 736/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1603
    Epoch 737/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1598
    Epoch 738/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1714
    Epoch 739/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1832
    Epoch 740/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1698
    Epoch 741/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1638
    Epoch 742/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1561
    Epoch 743/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1575
    Epoch 744/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1536
    Epoch 745/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1615
    Epoch 746/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1653
    Epoch 747/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1647
    Epoch 748/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1587
    Epoch 749/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1592
    Epoch 750/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1620
    Epoch 751/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1577
    Epoch 752/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1612
    Epoch 753/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1570
    Epoch 754/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1626
    Epoch 755/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1578
    Epoch 756/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1551
    Epoch 757/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1586
    Epoch 758/1000
    13/13 [==============================] - 0s 4ms/step - loss: 0.1532
    Epoch 759/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1573
    Epoch 760/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1603
    Epoch 761/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1582
    Epoch 762/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1551
    Epoch 763/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1670
    Epoch 764/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1811
    Epoch 765/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1565
    Epoch 766/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1609
    Epoch 767/1000
    13/13 [==============================] - 0s 2ms/step - loss: 0.1624
    Epoch 768/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1588
    Epoch 769/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1516
    Epoch 770/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1537
    Epoch 771/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1596
    Epoch 772/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1531
    Epoch 773/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1655
    Epoch 774/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1720
    Epoch 775/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1744
    Epoch 776/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1563
    Epoch 777/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1590
    Epoch 778/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1543
    Epoch 779/1000
    13/13 [==============================] - 0s 4ms/step - loss: 0.1536
    Epoch 780/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1526
    Epoch 781/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1564
    Epoch 782/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1557
    Epoch 783/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1491
    Epoch 784/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1597
    Epoch 785/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1565
    Epoch 786/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1683
    Epoch 787/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1617
    Epoch 788/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1532
    Epoch 789/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1497
    Epoch 790/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1545
    Epoch 791/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1577
    Epoch 792/1000
    13/13 [==============================] - 0s 4ms/step - loss: 0.1562
    Epoch 793/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1554
    Epoch 794/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1573
    Epoch 795/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1554
    Epoch 796/1000
    13/13 [==============================] - 0s 4ms/step - loss: 0.1542
    Epoch 797/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1676
    Epoch 798/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1574
    Epoch 799/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1515
    Epoch 800/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1504
    Epoch 801/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1536
    Epoch 802/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1482
    Epoch 803/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1533
    Epoch 804/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1660
    Epoch 805/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1520
    Epoch 806/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1538
    Epoch 807/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1547
    Epoch 808/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1503
    Epoch 809/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1535
    Epoch 810/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1703
    Epoch 811/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1708
    Epoch 812/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1572
    Epoch 813/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1517
    Epoch 814/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1531
    Epoch 815/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1550
    Epoch 816/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1579
    Epoch 817/1000
    13/13 [==============================] - 0s 4ms/step - loss: 0.1608
    Epoch 818/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1555
    Epoch 819/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1793
    Epoch 820/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1690
    Epoch 821/1000
    13/13 [==============================] - 0s 4ms/step - loss: 0.1628
    Epoch 822/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1627
    Epoch 823/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1586
    Epoch 824/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1757
    Epoch 825/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1684
    Epoch 826/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1590
    Epoch 827/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1590
    Epoch 828/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1637
    Epoch 829/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1608
    Epoch 830/1000
    13/13 [==============================] - 0s 2ms/step - loss: 0.1513
    Epoch 831/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1630
    Epoch 832/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1555
    Epoch 833/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1585
    Epoch 834/1000
    13/13 [==============================] - 0s 2ms/step - loss: 0.1480
    Epoch 835/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1504
    Epoch 836/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1528
    Epoch 837/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1586
    Epoch 838/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1555
    Epoch 839/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1596
    Epoch 840/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1552
    Epoch 841/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1612
    Epoch 842/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1690
    Epoch 843/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1602
    Epoch 844/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1541
    Epoch 845/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1550
    Epoch 846/1000
    13/13 [==============================] - 0s 4ms/step - loss: 0.1606
    Epoch 847/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1592
    Epoch 848/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1574
    Epoch 849/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1516
    Epoch 850/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1509
    Epoch 851/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1519
    Epoch 852/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1667
    Epoch 853/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1508
    Epoch 854/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1507
    Epoch 855/1000
    13/13 [==============================] - 0s 2ms/step - loss: 0.1633
    Epoch 856/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1582
    Epoch 857/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1661
    Epoch 858/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1703
    Epoch 859/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1607
    Epoch 860/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1603
    Epoch 861/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1549
    Epoch 862/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1584
    Epoch 863/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1518
    Epoch 864/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1543
    Epoch 865/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1512
    Epoch 866/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1637
    Epoch 867/1000
    13/13 [==============================] - 0s 2ms/step - loss: 0.1567
    Epoch 868/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1579
    Epoch 869/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1508
    Epoch 870/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1573
    Epoch 871/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1512
    Epoch 872/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1483
    Epoch 873/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1507
    Epoch 874/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1528
    Epoch 875/1000
    13/13 [==============================] - 0s 4ms/step - loss: 0.1563
    Epoch 876/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1574
    Epoch 877/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1532
    Epoch 878/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1611
    Epoch 879/1000
    13/13 [==============================] - 0s 4ms/step - loss: 0.1620
    Epoch 880/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1547
    Epoch 881/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1618
    Epoch 882/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1671
    Epoch 883/1000
    13/13 [==============================] - 0s 4ms/step - loss: 0.1656
    Epoch 884/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1590
    Epoch 885/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1662
    Epoch 886/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1552
    Epoch 887/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1557
    Epoch 888/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1586
    Epoch 889/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1580
    Epoch 890/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1534
    Epoch 891/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1542
    Epoch 892/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1603
    Epoch 893/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1542
    Epoch 894/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1605
    Epoch 895/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1594
    Epoch 896/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1528
    Epoch 897/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1592
    Epoch 898/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1519
    Epoch 899/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1682
    Epoch 900/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1645
    Epoch 901/1000
    13/13 [==============================] - 0s 2ms/step - loss: 0.1666
    Epoch 902/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1746
    Epoch 903/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1785
    Epoch 904/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1790
    Epoch 905/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1830
    Epoch 906/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1629
    Epoch 907/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1628
    Epoch 908/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1514
    Epoch 909/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1558
    Epoch 910/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1518
    Epoch 911/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1519
    Epoch 912/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1464
    Epoch 913/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1500
    Epoch 914/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1474
    Epoch 915/1000
    13/13 [==============================] - 0s 4ms/step - loss: 0.1558
    Epoch 916/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1530
    Epoch 917/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1539
    Epoch 918/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1493
    Epoch 919/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1500
    Epoch 920/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1471
    Epoch 921/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1483
    Epoch 922/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1545
    Epoch 923/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1526
    Epoch 924/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1526
    Epoch 925/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1487
    Epoch 926/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1503
    Epoch 927/1000
    13/13 [==============================] - 0s 4ms/step - loss: 0.1488
    Epoch 928/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1499
    Epoch 929/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1495
    Epoch 930/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1526
    Epoch 931/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1513
    Epoch 932/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1522
    Epoch 933/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1556
    Epoch 934/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1561
    Epoch 935/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1499
    Epoch 936/1000
    13/13 [==============================] - 0s 2ms/step - loss: 0.1436
    Epoch 937/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1455
    Epoch 938/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1599
    Epoch 939/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1516
    Epoch 940/1000
    13/13 [==============================] - 0s 4ms/step - loss: 0.1553
    Epoch 941/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1578
    Epoch 942/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1452
    Epoch 943/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1505
    Epoch 944/1000
    13/13 [==============================] - 0s 2ms/step - loss: 0.1507
    Epoch 945/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1521
    Epoch 946/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1551
    Epoch 947/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1554
    Epoch 948/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1675
    Epoch 949/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1505
    Epoch 950/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1504
    Epoch 951/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1501
    Epoch 952/1000
    13/13 [==============================] - 0s 4ms/step - loss: 0.1580
    Epoch 953/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1586
    Epoch 954/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1537
    Epoch 955/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1565
    Epoch 956/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1492
    Epoch 957/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1532
    Epoch 958/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1590
    Epoch 959/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1550
    Epoch 960/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1483
    Epoch 961/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1471
    Epoch 962/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1516
    Epoch 963/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1588
    Epoch 964/1000
    13/13 [==============================] - 0s 4ms/step - loss: 0.1493
    Epoch 965/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1506
    Epoch 966/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1461
    Epoch 967/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1539
    Epoch 968/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1624
    Epoch 969/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1565
    Epoch 970/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1548
    Epoch 971/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1566
    Epoch 972/1000
    13/13 [==============================] - 0s 4ms/step - loss: 0.1498
    Epoch 973/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1529
    Epoch 974/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1591
    Epoch 975/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1545
    Epoch 976/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1460
    Epoch 977/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1581
    Epoch 978/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1575
    Epoch 979/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1494
    Epoch 980/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1558
    Epoch 981/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1530
    Epoch 982/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1445
    Epoch 983/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1529
    Epoch 984/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.1450
    Epoch 985/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1494
    Epoch 986/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1509
    Epoch 987/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1414
    Epoch 988/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1559
    Epoch 989/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1550
    Epoch 990/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1503
    Epoch 991/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1496
    Epoch 992/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1508
    Epoch 993/1000
    13/13 [==============================] - 0s 2ms/step - loss: 0.1506
    Epoch 994/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1482
    Epoch 995/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1551
    Epoch 996/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1615
    Epoch 997/1000
    13/13 [==============================] - 0s 4ms/step - loss: 0.1736
    Epoch 998/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1754
    Epoch 999/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1714
    Epoch 1000/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.1538
    Finished lambda = 0.001
    Epoch 1/1000
    13/13 [==============================] - 0s 1ms/step - loss: 1.4887
    Epoch 2/1000
    13/13 [==============================] - 0s 2ms/step - loss: 0.7947
    Epoch 3/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.6159
    Epoch 4/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.5300
    Epoch 5/1000
    13/13 [==============================] - 0s 4ms/step - loss: 0.4991
    Epoch 6/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4763
    Epoch 7/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4761
    Epoch 8/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4651
    Epoch 9/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.4366
    Epoch 10/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4063
    Epoch 11/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4032
    Epoch 12/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4115
    Epoch 13/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.4421
    Epoch 14/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4607
    Epoch 15/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4457
    Epoch 16/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4180
    Epoch 17/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3981
    Epoch 18/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3570
    Epoch 19/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3768
    Epoch 20/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3689
    Epoch 21/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3560
    Epoch 22/1000
    13/13 [==============================] - 0s 3ms/step - loss: 0.3717
    Epoch 23/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3560
    Epoch 24/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.3582
    Epoch 25/1000
    13/13 [==============================] - 0s 1ms/step - loss: 0.4461
    Epoch 26/1000
    13/13 [==============================] - 0s 4ms/step - loss: 0.4211



```python
plot_iterate(lambdas, models, X_train, y_train, X_cv, y_cv)
```

As regularization is increased, the performance of the model on the training and cross-validation data sets converge. For this data set and model, lambda > 0.01 seems to be a reasonable choice.

<a name="7.1"></a>
### 7.1 Test
Let's try our optimized models on the test set and compare them to 'ideal' performance. 


```python
plt_compare(X_test,y_test, classes, model_predict_s, model_predict_r, centers)
```

Our test set is small and seems to have a number of outliers so classification error is high. However, the performance of our optimized models is comparable to ideal performance.

## Congratulations! 
You have become familiar with important tools to apply when evaluating your machine learning models. Namely:  
* splitting data into trained and untrained sets allows you to differentiate between underfitting and overfitting
* creating three data sets, Training, Cross-Validation and Test allows you to
    * train your parameters $W,B$ with the training set
    * tune model parameters such as complexity, regularization and number of examples with the cross-validation set
    * evaluate your 'real world' performance using the test set.
* comparing training vs cross-validation performance provides insight into a model's propensity towards overfitting (high variance) or underfitting (high bias)


```python

```
