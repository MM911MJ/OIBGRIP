Python 3.11.1 (tags/v3.11.1:a7a450f, Dec  6 2022, 19:58:39) [MSC v.1934 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
#Iris Flower Classification
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
iris=pd.read_csv("C:\\Users\\madak\\OneDrive\\Desktop\\Iris.csv")
print(iris.head())
   Id  SepalLengthCm  SepalWidthCm  PetalLengthCm  PetalWidthCm      Species
0   1            5.1           3.5            1.4           0.2  Iris-setosa
1   2            4.9           3.0            1.4           0.2  Iris-setosa
2   3            4.7           3.2            1.3           0.2  Iris-setosa
3   4            4.6           3.1            1.5           0.2  Iris-setosa
4   5            5.0           3.6            1.4           0.2  Iris-setosa
print(iris.describe())
               Id  SepalLengthCm  SepalWidthCm  PetalLengthCm  PetalWidthCm
count  150.000000     150.000000    150.000000     150.000000    150.000000
mean    75.500000       5.843333      3.054000       3.758667      1.198667
std     43.445368       0.828066      0.433594       1.764420      0.763161
min      1.000000       4.300000      2.000000       1.000000      0.100000
25%     38.250000       5.100000      2.800000       1.600000      0.300000
50%     75.500000       5.800000      3.000000       4.350000      1.300000
75%    112.750000       6.400000      3.300000       5.100000      1.800000
max    150.000000       7.900000      4.400000       6.900000      2.500000
print("Target Labels",iris["Species"].unique())
Target Labels ['Iris-setosa' 'Iris-versicolor' 'Iris-virginica']

#Plotting the dataset
import plotly.express as px
fig=px.scatter(iris,x="SepalWidthCm",y="SepalLengthCm",color="Species")
fig.show()

#performing classification
from sklearn.model_selection import train_test_split
X = iris.drop(['Species'], axis=1)
X = X.to_numpy()[:, (2,3)]
y = iris['Species']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.5, random_state=42)
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X_train,y_train)
LogisticRegression()

#training predictions
training_prediction = log_reg.predict(X_train)
training_prediction
array(['Iris-versicolor', 'Iris-virginica', 'Iris-versicolor',
       'Iris-setosa', 'Iris-versicolor', 'Iris-virginica', 'Iris-setosa',
       'Iris-setosa', 'Iris-versicolor', 'Iris-virginica', 'Iris-setosa',
       'Iris-virginica', 'Iris-setosa', 'Iris-setosa', 'Iris-versicolor',
       'Iris-versicolor', 'Iris-virginica', 'Iris-virginica',
       'Iris-virginica', 'Iris-virginica', 'Iris-versicolor',
       'Iris-setosa', 'Iris-setosa', 'Iris-virginica', 'Iris-virginica',
       'Iris-setosa', 'Iris-setosa', 'Iris-setosa', 'Iris-versicolor',
       'Iris-virginica', 'Iris-setosa', 'Iris-virginica',
       'Iris-virginica', 'Iris-setosa', 'Iris-versicolor',
       'Iris-versicolor', 'Iris-versicolor', 'Iris-versicolor',
       'Iris-virginica', 'Iris-setosa', 'Iris-virginica',
       'Iris-versicolor', 'Iris-virginica', 'Iris-versicolor',
       'Iris-versicolor', 'Iris-versicolor', 'Iris-setosa',
       'Iris-versicolor', 'Iris-versicolor', 'Iris-setosa',
       'Iris-versicolor', 'Iris-virginica', 'Iris-virginica',
       'Iris-setosa', 'Iris-versicolor', 'Iris-virginica',
       'Iris-virginica', 'Iris-setosa', 'Iris-virginica', 'Iris-setosa',
       'Iris-virginica', 'Iris-virginica', 'Iris-virginica',
       'Iris-versicolor', 'Iris-virginica', 'Iris-versicolor',
       'Iris-versicolor', 'Iris-versicolor', 'Iris-virginica',
       'Iris-setosa', 'Iris-versicolor', 'Iris-versicolor', 'Iris-setosa',
       'Iris-versicolor', 'Iris-virginica'], dtype=object)

#test predictions
test_prediction = log_reg.predict(X_test)
test_prediction
array(['Iris-versicolor', 'Iris-setosa', 'Iris-virginica',
       'Iris-versicolor', 'Iris-versicolor', 'Iris-setosa',
       'Iris-versicolor', 'Iris-virginica', 'Iris-versicolor',
       'Iris-versicolor', 'Iris-virginica', 'Iris-setosa', 'Iris-setosa',
       'Iris-setosa', 'Iris-setosa', 'Iris-versicolor', 'Iris-virginica',
       'Iris-versicolor', 'Iris-versicolor', 'Iris-virginica',
       'Iris-setosa', 'Iris-virginica', 'Iris-setosa', 'Iris-virginica',
       'Iris-virginica', 'Iris-virginica', 'Iris-virginica',
       'Iris-virginica', 'Iris-setosa', 'Iris-setosa', 'Iris-setosa',
       'Iris-setosa', 'Iris-versicolor', 'Iris-setosa', 'Iris-setosa',
       'Iris-virginica', 'Iris-versicolor', 'Iris-setosa', 'Iris-setosa',
       'Iris-setosa', 'Iris-virginica', 'Iris-versicolor',
       'Iris-versicolor', 'Iris-setosa', 'Iris-setosa', 'Iris-versicolor',
       'Iris-virginica', 'Iris-virginica', 'Iris-versicolor',
       'Iris-virginica', 'Iris-versicolor', 'Iris-virginica',
       'Iris-versicolor', 'Iris-setosa', 'Iris-virginica',
       'Iris-versicolor', 'Iris-setosa', 'Iris-setosa', 'Iris-setosa',
       'Iris-versicolor', 'Iris-versicolor', 'Iris-setosa', 'Iris-setosa',
       'Iris-setosa', 'Iris-versicolor', 'Iris-setosa', 'Iris-versicolor',
       'Iris-virginica', 'Iris-setosa', 'Iris-versicolor',
       'Iris-virginica', 'Iris-setosa', 'Iris-versicolor',
       'Iris-virginica', 'Iris-versicolor'], dtype=object)
>>> 
>>> #performance in training
>>> from sklearn import metrics
>>> print("Precision, Recall, Confusion matrix, in training\n")
Precision, Recall, Confusion matrix, in training

>>> print(metrics.classification_report(y_train, training_prediction, digits=3))
                 precision    recall  f1-score   support

    Iris-setosa      1.000     1.000     1.000        21
Iris-versicolor      0.889     0.889     0.889        27
 Iris-virginica      0.889     0.889     0.889        27

       accuracy                          0.920        75
      macro avg      0.926     0.926     0.926        75
   weighted avg      0.920     0.920     0.920        75

>>> print(metrics.confusion_matrix(y_train, training_prediction))
[[21  0  0]
 [ 0 24  3]
 [ 0  3 24]]
>>> 
>>> #performance in testing
>>> print("Precision, Recall, Confusion matrix, in testing\n")
Precision, Recall, Confusion matrix, in testing

>>> print(metrics.classification_report(y_test, test_prediction, digits=3))
                 precision    recall  f1-score   support

    Iris-setosa      1.000     1.000     1.000        29
Iris-versicolor      0.920     1.000     0.958        23
 Iris-virginica      1.000     0.913     0.955        23

       accuracy                          0.973        75
      macro avg      0.973     0.971     0.971        75
   weighted avg      0.975     0.973     0.973        75

>>> print(metrics.confusion_matrix(y_test, test_prediction))
[[29  0  0]
 [ 0 23  0]
 [ 0  2 21]]
