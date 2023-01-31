Python 3.11.1 (tags/v3.11.1:a7a450f, Dec  6 2022, 19:58:39) [MSC v.1934 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>> import pandas as pd
>>> import numpy as np
>>> import matplotlib.pyplot as plt
>>> import seaborn as sns
>>> from sklearn.model_selection import train_test_split
>>> from sklearn.linear_model import LinearRegression
>>> data = pd.read_csv("https://raw.githubusercontent.com/amankharwal/Website-data/master/advertising.csv")
>>> print(data.head())
      TV  Radio  Newspaper  Sales
0  230.1   37.8       69.2   22.1
1   44.5   39.3       45.1   10.4
2   17.2   45.9       69.3   12.0
3  151.5   41.3       58.5   16.5
4  180.8   10.8       58.4   17.9
>>> print(data.isnull().sum())
TV           0
Radio        0
Newspaper    0
Sales        0
dtype: int64
>>> sns.set(style="whitegrid")
>>> plt.figure(figsize=(12, 10))
<Figure size 1200x1000 with 0 Axes>
>>> sns.heatmap(data.corr())
<AxesSubplot: >
>>> plt.show()
>>> 
>>> x = np.array(data.drop(["Sales"], 1))

Warning (from warnings module):
  File "<pyshell#14>", line 1
FutureWarning: In a future version of pandas all arguments of DataFrame.drop except for the argument 'labels' will be keyword-only.
>>> y = np.array(data["Sales"])
>>> xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)
>>> model = LinearRegression()
>>> model.fit(xtrain, ytrain)
LinearRegression()
>>> ypred = model.predict(xtest)
>>> data = pd.DataFrame(data={"Predicted Sales": ypred.flatten()})
>>> print(data)
    Predicted Sales
0         17.034772
1         20.409740
2         23.723989
3          9.272785
4         21.682719
5         12.569402
6         21.081195
7          8.690350
8         17.237013
9         16.666575
10         8.923965
11         8.481734
12        18.207512
13         8.067507
14        12.645510
15        14.931628
16         8.128146
17        17.898766
18        11.008806
19        20.478328
20        20.806318
21        12.598833
22        10.905183
23        22.388548
24         9.417961
25         7.925067
26        20.839085
27        13.815209
28        10.770809
29         7.926825
30        15.959474
31        10.634909
32        20.802920
33        10.434342
34        21.578475
35        21.183645
36        12.128218
37        22.809533
38        12.609928
39         6.464413
