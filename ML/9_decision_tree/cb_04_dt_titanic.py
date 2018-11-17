#coding:utf-8
"""
------------------------------------------------
@File Name    : cb_04_dt_titanic
@Function     : 
@Author       : Minux
@Date         : 2018/11/17
@Revised Date : 2018/11/17
------------------------------------------------
"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

titanic = pd.read_csv('titanic.csv')
# print(titanic.head())

def titanic_prediction():
    inputs = titanic.drop(['PassengerId', 'Survived', 'Name', 'Ticket'], axis=1)
    inputs['Cabin'].fillna('Unkown', inplace=True)
    inputs['Embarked'].fillna('U', inplace=True)
    inputs['Age'].fillna(-1, inplace=True)

    y_true = titanic['Survived']

    le_Sex = LabelEncoder()
    le_Cabin = LabelEncoder()
    le_Embarked = LabelEncoder()

    inputs['Sex_n'] = le_Sex.fit_transform(inputs['Sex'])
    inputs['Cabin_n'] = le_Sex.fit_transform(inputs['Cabin'])
    inputs['Embarked_n'] = le_Embarked.fit_transform(inputs['Embarked'])

    inputs.drop(['Sex', 'Cabin', 'Embarked'], axis=1, inplace=True)

    # print(inputs)

    dt_model = DecisionTreeClassifier()
    dt_model.fit(inputs, y_true)
    res = dt_model.score(inputs, y_true)
    print(res)

if __name__ == '__main__':
    titanic_prediction()