from django.shortcuts import render
import sklearn
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split              
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score
from datetime import date




# Create your views here.
def home(request):
    return render(request,'index.html')

def main(request):
    
    return render(request,'main.html')


def result(request):

    val1 = float(request.GET['n1'])
    val2 = float(request.GET['n2'])
    val3 = float(request.GET['n3'])
    val4 = float(request.GET['n4'])
    val5 = float(request.GET['n5'])
    val6 = float(request.GET['n6'])
    val7 = float(request.GET['n7'])
    val8 = float(request.GET['n8'])
    val9 = str(request.GET['n9'])
    data = pd.read_csv(".\diabetes.csv")
    x = data.drop('Outcome',axis=1)
    y = data['Outcome']
    
    

    X_train, X_test, Y_train, Y_test = train_test_split(x,y, test_size=0.3, random_state=101)

    from sklearn.impute import SimpleImputer
    fill_values = SimpleImputer(missing_values=np.nan,strategy='mean')
    X_train = fill_values.fit_transform(X_train)
    X_test = fill_values.fit_transform(X_test)

    

    from sklearn.ensemble import RandomForestClassifier

    random_forest = RandomForestClassifier(random_state=10)
    random_forest.fit(X_train, Y_train)

    

    
    
    predection1 = random_forest.predict([[val1, val2, val3, val4, val5, val6, val7, val8]])
    from sklearn import metrics
    
    from sklearn.svm import SVC
    svm_model = SVC()
    svm_model.fit(X_train, Y_train)
    predection2 = svm_model.predict([[val1, val2, val3, val4, val5, val6, val7, val8]])


    from sklearn.tree import DecisionTreeClassifier
    dtree = DecisionTreeClassifier()
    dtree.fit(X_train,Y_train)
    predection3 = dtree.predict([[val1, val2, val3, val4, val5, val6, val7, val8]])



    from sklearn.linear_model import LogisticRegression
    logistic = LogisticRegression()
    logistic.fit(X_train,Y_train)
    predection4 = logistic.predict([[val1, val2, val3, val4, val5, val6, val7, val8]])




    result1 = ''
    result2=''
    result3=''
    result4=''


    if predection1 == [1]:
        result1 = 'POSITIVE'
    else:
        result1 = 'NEGATIVE'


    if predection2 == [1]:
        result2 = 'POSITIVE'
    else:
        result2 = 'NEGATIVE'


    if predection3 == [1]:
        result3 = 'POSITIVE'
    else:
        result3 = 'NEGATIVE'

    if predection4 == [1]:
        result4 = 'POSITIVE'
    else:
        result4 = 'NEGATIVE'

    
    today = date.today()


    return render(request,'result.html',{'result0':result1,'result1':result2,'result2':result3,'result3':result4,'Name':val9,'date':today,'age':int(val8),'Pregnancies':int(val1),'Glucose':val2,'BloodPressure':val3,'SkinThickness':val4,'Insulin':val5,'BMI':val6,'DiabetesPedigreeFunction':val7})





    

