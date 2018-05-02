import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import itertools
from sklearn.preprocessing import label_binarize

def do_LDA(model,X_train,Y_train,X_test,Y_test):
	clf = LinearDiscriminantAnalysis()
	X_train=clf.fit_transform(X_train,Y_train)
	X_test=clf.transform(X_test)
	clf=model
	clf=clf.fit(X_train,Y_train)
	scores=clf.predict_proba(X_test)
	print("LDA")
	print(clf.score(X_test,Y_test))
	trueLabelsBin = label_binarize(Y_test, classes=list(set(Y_test)))
	print(trueLabelsBin.ravel())
	fpr,tpr,rf=roc_curve(trueLabelsBin.ravel(),scores.ravel())
	return fpr,tpr

def do_PCA(model,X_train,Y_train,X_test,Y_test):
	pca = PCA(n_components=0.9)
	X_train=pca.fit_transform(X_train)
	X_test=pca.transform(X_test)
	clf=model
	clf=clf.fit(X_train,Y_train)
	scores=clf.predict_proba(X_test)
	print("PCA")
	print(clf.score(X_test,Y_test))
	trueLabelsBin = label_binarize(Y_test, classes=list(set(Y_test)))
	fpr,tpr,rf=roc_curve(trueLabelsBin.ravel(),scores.ravel())
	return fpr,tpr

def do_Normal(model,X_train,Y_train,X_test,Y_test):
	clf=model
	clf=clf.fit(X_train,Y_train)
	scores=clf.predict_proba(X_test)
	print("Normal")
	print(clf.score(X_test,Y_test))
	trueLabelsBin = label_binarize(Y_test, classes=list(set(Y_test)))
	fpr,tpr,rf=roc_curve(trueLabelsBin.ravel(),scores.ravel())
	return fpr,tpr


def make_my_comparision_roc(model,X_train,Y_train,X_test,Y_test,model_name):
	fpr1,tpr1=do_Normal(model,X_train,Y_train,X_test,Y_test)
	fpr2,tpr2=do_PCA(model,X_train,Y_train,X_test,Y_test)
	fpr3,tpr3=do_LDA(model,X_train,Y_train,X_test,Y_test)
	plt.plot(fpr1,tpr1,label="Normal")
	plt.plot(fpr2,tpr2,label="PCA")
	plt.plot(fpr3,tpr3,label="LDA")
	plt.legend()
	plt.xlabel("False Positive Rate")
	plt.ylabel("True Positive Rate")
	plt.title(model_name)
	plt.savefig(model_name+"_roc_comp.png")
	plt.show()

