# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
1. Import the necessary packages using import statement.
2. Read the given csv file and print the number of contents to be displayed.
3. Split the dataset using train_test_split.
4. Calculate Y_Pred and accuracy.
5. Display the result. 

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Subanu. K 
RegisterNumber:  212219040152
*/
import pandas as pd
data=pd.read_csv("spam.csv",encoding='latin-1')
data.head()
data.info()
data.isnull().sum()
x=data["EmailText"].values
y=data["Label"].values
from sklearn.model_selection import train_test_split 
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```

## Output:
## Data.head()
![image](https://user-images.githubusercontent.com/87663343/174468698-d2fb8a07-aaf6-498c-9ed3-6eaaecba83c5.png)
## Data.info()
![image](https://user-images.githubusercontent.com/87663343/174468720-76768b71-db8c-4073-ae35-c6ed57a28fc6.png)
## Data.isnull().sum():
![image](https://user-images.githubusercontent.com/87663343/174468740-b9157b06-5c00-456e-9063-67cae56b129f.png)
## Y_pred:
![image](https://user-images.githubusercontent.com/87663343/174468759-0a9dafb0-ce36-4b81-98a0-570bf7e0a598.png)
## Accuracy:
![image](https://user-images.githubusercontent.com/87663343/174468773-a6c5f52d-0ceb-4d20-8bc6-c3226e5da7e9.png)





## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
