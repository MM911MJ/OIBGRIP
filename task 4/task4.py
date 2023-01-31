Python 3.11.1 (tags/v3.11.1:a7a450f, Dec  6 2022, 19:58:39) [MSC v.1934 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>> import numpy as np
>>> import pandas as pd
>>> from sklearn.feature_extraction.text import CountVectorizer
>>> from sklearn.model_selection import train_test_split
>>> from sklearn.naive_bayes import MultinomialNB
>>> data=pd.read_csv("https://raw.githubusercontent.com/amankharwal/SMS-Spam-Detection/master/spam.csv", encoding= 'latin-1')
>>> data.head( )
  class  ... Unnamed: 4
0   ham  ...        NaN
1   ham  ...        NaN
2  spam  ...        NaN
3   ham  ...        NaN
4   ham  ...        NaN

[5 rows x 5 columns]
>>> data.isnull().sum()
class            0
message          0
Unnamed: 2    5522
Unnamed: 3    5560
Unnamed: 4    5566
dtype: int64
>>> data = data[['class','message']]
>>> data.head()
  class                                            message
0   ham  Go until jurong point, crazy.. Available only ...
1   ham                      Ok lar... Joking wif u oni...
2  spam  Free entry in 2 a wkly comp to win FA Cup fina...
3   ham  U dun say so early hor... U c already then say...
4   ham  Nah I don't think he goes to usf, he lives aro...
>>> x=np.array(data['message'])
>>> y=np.array(data['class'])
>>> cv=CountVectorizer()
>>> X=cv.fit_transform(x)
>>> Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,test_size=0.3)
>>> clf = MultinomialNB()
>>> clf.fit(X,y)
MultinomialNB()
>>> sample=input("Enter a message:")
Enter a message:you won $9 cash prize
>>> dt=cv.transform([sample]).toarray()
>>> print(clf.predict(dt))
['spam']
