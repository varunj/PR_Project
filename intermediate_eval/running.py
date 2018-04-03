import numpy as np 
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA,KernelPCA
import collections
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import NMF
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
# datatrain=np.genfromtxt('train.csv',delimiter=',')
# datatest=np.genfromtxt('test.csv',delimiter=',')
# X_train=datatrain[1:,1:]
# y_train=datatrain[1:,0]
# X_test=datatest[1:,1:]
# y_test=datatest[1:,0]
data=np.genfromtxt('./Feat/total_o_50.csv', delimiter=',')
Y=np.asarray([int(y) for y in data[:,0].tolist()])
X=data[:,1:]
#Preparing Data
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
clf = LinearDiscriminantAnalysis()
X_train=clf.fit_transform(X_train,y_train)
X_test=clf.transform(X_test)
# pca = PCA(n_components=20)
# X_train=pca.fit_transform(X_train)
# X_test=pca.transform(X_test)
# kpca=KernelPCA(kernel='rbf')
# X_train=kpca.fit_transform(X_train)
# X_test=kpca.transform(X_test)

# print()
# print(collections.Counter(y_train.tolist()))
# print(collections.Counter(y_test.tolist()))
#Gaussian Naive Bayes
clf=GaussianNB()
clf=clf.fit(X_train,y_train)
print(clf.score(X_test,y_test))

clf = LinearSVC(random_state=0)
clf=clf.fit(X_train,y_train)
print(clf.score(X_test,y_test))

clf = SVC(random_state=0,kernel='rbf',C=10,tol=0.0001,degree=4)
clf=clf.fit(X_train,y_train)
print(clf.score(X_test,y_test))

clf = RandomForestClassifier(max_depth=15, random_state=0,n_estimators=20)
clf=clf.fit(X_train,y_train)
print(clf.score(X_test,y_test))

clf=MLPClassifier((100,100,50),activation='relu',solver='lbfgs',learning_rate_init=0.00001)
clf=clf.fit(X_train,y_train)
print(clf.score(X_test,y_test))

clf=AdaBoostClassifier()
clf=clf.fit(X_train,y_train)
print(clf.score(X_test,y_test))

clf=BaggingClassifier()
clf=clf.fit(X_train,y_train)
print(clf.score(X_test,y_test))

clf=KNeighborsClassifier(n_neighbors=2)
clf=clf.fit(X_train,y_train)
print(clf.score(X_test,y_test))
