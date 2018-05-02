from rocComp import *
import numpy as np 
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier

data=np.genfromtxt('./Feat/total_o_50.csv', delimiter=',')
Y=np.asarray([int(y) for y in data[:,0].tolist()])
X=data[:,1:]

#Preparing Data
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3,stratify=Y)

make_my_comparision_roc(GaussianNB(),X_train,y_train,X_test,y_test,"GaussianNB")
make_my_comparision_roc(SVC(kernel='linear',random_state=0,probability=True,max_iter=200),X_train,y_train,X_test,y_test,"Linear SVM")
make_my_comparision_roc(SVC(random_state=0,kernel='rbf',C=10,tol=0.0001,degree=4,probability=True,max_iter=200),X_train,y_train,X_test,y_test,"RBF SVM")
make_my_comparision_roc(RandomForestClassifier(max_depth=15, random_state=0,n_estimators=20),X_train,y_train,X_test,y_test,"Random Forest")
make_my_comparision_roc(MLPClassifier((100,100,50),activation='relu',solver='lbfgs',learning_rate_init=0.00001),X_train,y_train,X_test,y_test,"Neural Net")
make_my_comparision_roc(AdaBoostClassifier(),X_train,y_train,X_test,y_test,"AdaBoost")
make_my_comparision_roc(BaggingClassifier(),X_train,y_train,X_test,y_test,"Bagging")
make_my_comparision_roc(KNeighborsClassifier(n_neighbors=2),X_train,y_train,X_test,y_test,"Knn")