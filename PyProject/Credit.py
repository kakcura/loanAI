import itertools
from builtins import float
from SVM import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from  sklearn.preprocessing import StandardScaler
from KNN import  KNN
from NaiveBayes import fit, getPredictions, getAccuracy
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math
from IPython.core.interactiveshell import InteractiveShell
import warnings
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report,accuracy_score
from sklearn.metrics import classification_report
from sklearn import svm
warnings.filterwarnings('ignore') # to supress seaborn warnings
pd.options.display.max_columns = None
pd.options.display.max_columns = None


def fix_header(data):
    # fix header and make the first row as header
    new_header = data.iloc[0]
    data = data[1:]
    data.columns = new_header
    data.rename(columns={'default payment next month': 'DEFAULTER'}, inplace=True)
    return data


def main():
    #data_set = pd.read_csv('creditdata.csv', index_col=0)
    data_set = pd.read_csv('creditSmall.csv', index_col=0)

    data_set = fix_header(data_set)

    data_set.EDUCATION[data_set.EDUCATION == '0'] = '4'
    data_set.EDUCATION[data_set.EDUCATION == '5'] = '4'
    data_set.EDUCATION[data_set.EDUCATION == '6'] = '4'
    data_set.MARRIAGE[data_set.MARRIAGE == '0'] = '3'
    data_set = data_set.astype(float)
    print(data_set.DEFAULTER.mean()*100)

    data_set['BILL_PAY_RATIO1'] = (data_set['BILL_AMT1'] - data_set['PAY_AMT1']) / data_set['LIMIT_BAL']
    data_set['BILL_PAY_RATIO2'] = (data_set['BILL_AMT2'] - data_set['PAY_AMT2']) / data_set['LIMIT_BAL']
    data_set['BILL_PAY_RATIO3'] = (data_set['BILL_AMT3'] - data_set['PAY_AMT3']) / data_set['LIMIT_BAL']
    data_set['BILL_PAY_RATIO4'] = (data_set['BILL_AMT4'] - data_set['PAY_AMT4']) / data_set['LIMIT_BAL']
    data_set['BILL_PAY_RATIO5'] = (data_set['BILL_AMT5'] - data_set['PAY_AMT5']) / data_set['LIMIT_BAL']
    data_set['BILL_PAY_RATIO6'] = (data_set['BILL_AMT6'] - data_set['PAY_AMT6']) / data_set['LIMIT_BAL']

    x = data_set.drop(['DEFAULTER'], axis=1)
    y = data_set.DEFAULTER

    # rescale the metrics to the same mean and standard deviation
    scaler = preprocessing.StandardScaler()
    x = scaler.fit(x).transform(x)

    # Further divide the train data into train test split 70% & 30% respectively
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, stratify=y, random_state=2)



    # creating classifier

    # classifier = neighbors.KNeighborsClassifier(n_neighbors=3)
    # classifier.fit(X_train, Y_train)
    # accuracy = classifier.score(X_test, Y_test)
    # print("Accuracy :",accuracy)

    #myaccuracy= KNN(x_train,y_train,y_test,x_test,3)
    #print(myaccuracy)


    # Predicition using Naive Bayes
    # model = fit(X_train, Y_train)
    # predictions = getPredictions(model, X_test)
    # accuracies = getAccuracy(Y_test, predictions)
    # print('Accuracy: ', accuracies)
    #
    # gaussNb = GaussianNB()
    # gaussNb.fit(X_train, Y_train)
    # print(gaussNb)
    # y_expect = Y_test
    # y_pred = gaussNb.predict(X_test)
    # print(accuracy_score(y_expect, y_pred))
    #
    # classification = classification_report(Y_test, predictions)
    # print(classification)
    # label = [0, 1]
    # cmatrix = confusion_matrix(Y_test, predictions, label)
    # print(cmatrix)
    #
    # classification1 = classification_report(Y_test, y_pred)
    # print(classification1)
    # label1 = [0, 1]
    # cmatrix1 = confusion_matrix(Y_test, y_pred, label1)
    # print(cmatrix1)
    # plot_confusion_matrix(cmatrix, label)
    # plot_confusion_matrix(cmatrix1, label1, title="2")


    clf = SVM()
    y_svmTrain = np.where(y_train == 0, -1, 1)
    clf.fit(x_train, y_svmTrain)

    y_predict = clf.predict(x_test)
    correct = np.sum(y_predict == y_test)
    print("%d out of %d predictions correct" % (correct, len(y_predict)))
    print()
    print(y_test)

    plot_margin(x_train[y_train == 1], x_train[y_train == -1], clf)
    print(x_test)






# for SVM
# model= svm.SVC(kernel='linear',C=1,gamma=1)
# model.fit(X_train,Y_train)
# model.score(X_train,Y_train)
# predicted=model.predict(X_test)
# print(predicted)
def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=False):
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()


if __name__ == '__main__':
    main()
