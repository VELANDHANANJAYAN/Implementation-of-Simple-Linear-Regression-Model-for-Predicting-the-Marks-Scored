# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.import the needed packages.

2.Assigning hours to x and scores to y.

3.Plot the scatter plot.

4.Use mse,rmse,mae formula to find the values.

## Program:
```python
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Velan D
RegisterNumber:  212222040176
*/
# IMPORT REQUIRED PACKAGE
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt
dataset=pd.read_csv('student_scores.csv')
print(dataset)
# READ CSV FILES
dataset=pd.read_csv('student_scores.csv')
print(dataset.head())
print(dataset.tail())
# COMPARE DATASET
x=dataset.iloc[:,:-1].values
print(x)
y=dataset.iloc[:,1].values
print(y)
# PRINT PREDICTED VALUE
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)
y_pred = reg.predict(x_test)
print(y_pred)
print(y_test)
# GRAPH PLOT FOR TRAINING SET
plt.scatter(x_train,y_train,color='purple')
plt.plot(x_train,reg.predict(x_train),color='black')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
# GRAPH PLOT FOR TESTING SET
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,reg.predict(x_train),color='black')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
# PRINT THE ERROR
mse=mean_absolute_error(y_test,y_pred)
print('Mean Square Error = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('Mean Absolute Error = ',mae)
rmse=np.sqrt(mse)
print("Root Mean Square Error = ",rmse)

```

## Output:
To read csv file

![image](https://github.com/VELANDHANANJAYAN/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119405038/ee006794-dc96-45cd-a28a-c2315a95a78e)


To Read Head and Tail Files

![image](https://github.com/VELANDHANANJAYAN/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119405038/061d03ef-8710-45b5-9254-0cb47782b673)


Compare Dataset

![image](https://github.com/VELANDHANANJAYAN/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119405038/42aabedd-71aa-4080-bada-daf37bd96340)


Predicted Value

![image](https://github.com/VELANDHANANJAYAN/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119405038/16c1170e-988e-43fe-92fa-b89a0db72c57)


Graph For Training Set

![image](https://github.com/VELANDHANANJAYAN/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119405038/39f9b67d-10f3-461e-9df7-5a764362241c)


Graph For Testing Set

![image](https://github.com/VELANDHANANJAYAN/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119405038/667e1567-7931-419d-a93c-deb5c70e1c54)


Error

![image](https://github.com/VELANDHANANJAYAN/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119405038/3ad71463-1722-48b5-a86f-343020b5b7a0)




## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
