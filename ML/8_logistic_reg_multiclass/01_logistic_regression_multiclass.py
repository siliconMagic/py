#coding:utf-8
"""
------------------------------------------------
@File Name    : 01_logistic_regression_multiclass
@Function     : 
@Author       : Minux
@Date         : 2018/9/22
@Revised Date : 2018/9/22
------------------------------------------------
"""
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
from sklearn.datasets import load_digits

digits = load_digits()

# check what digits contains
'''
the dataset is made up of 1797 8x8 images
'''
# print(dir(digits))
# print(digits.data[0])

def show_image_in_0_9():
    for i in range(10):
        plt.imshow(digits.images[i],cmap='gray')
        # plt.imshow(digits.data[0].reshape(8,8), cmap='gray')
        plt.show()


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import seaborn as sn


def call_logistic_regression():
    X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    res = model.score(X_test, y_test)
    print(res)
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    plt.figure(figsize=(10,7))
    sn.heatmap(cm, annot=True)
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    plt.show()


'''
confusion matrix 
'''

'''
Exercise: about iris
'''
from sklearn.datasets import load_iris
iris = load_iris()

def logistic_regression_for_iris_dataset():
    X_train,X_test,y_train,y_test = train_test_split(iris.data, iris.target ,test_size=0.2)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    res = model.score(X_test, y_test)
    print(res)
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10,7))
    sn.heatmap(cm, annot=True)
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    plt.savefig(r'./heatmap_of_iris.png')
    plt.show()


if __name__ == '__main__':
    # call_logistic_regression()
    logistic_regression_for_iris_dataset()




