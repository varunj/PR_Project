import numpy as np
np.random.seed(123)
np.set_printoptions(threshold=np.nan)
import os
import pickle
from pprint import pprint
from time import time
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, roc_auc_score, precision_score
import glob
from sklearn.preprocessing import label_binarize
import sklearn


def calcMetrics(trueLabels, predLabels, scores, filename):
	# cmc curve
	genuineArr, imposterArr = [], []
	trueLabels = [x-1 for x in trueLabels]

	for x in range(0,len(trueLabels)):
		genuineArr.append(scores[x][trueLabels[x]])
		imposterArr.append(np.concatenate((scores[x][:trueLabels[x]], scores[x][trueLabels[x]+1:]), axis=0))
	genuineArr = np.asarray(genuineArr).flatten()
	imposterArr = np.asarray(imposterArr).flatten()
	p, x = np.histogram(genuineArr, bins=int(len(scores)/10))
	pMatch = p/max(p)
	x = x[:-1] + (x[1] - x[0])/2
	plt.plot(x, pMatch, label='genuine')
	p2, x2 = np.histogram(imposterArr, bins=int(len(scores)/10))
	pImposter = p2/max(p2)
	x2 = x2[:-1] + (x2[1] - x2[0])/2
	plt.plot(x2, pImposter, label='imposter')
	plt.legend()
	plt.title('Match Score Distribution')
	plt.xlabel('Score')
	plt.ylabel('P(Match at Score)')
	plt.tight_layout()
	splitt = filename.split('_')
	plt.savefig(splitt[0] + '_' + splitt[1] + '_matchscore.png')
	plt.close()


	# roc, tpr, fpr
	trueLabelsBin = label_binarize(trueLabels, classes=list(set(trueLabels)))
	tpr, fpr, roc_auc = {}, {}, {}
	for i in range(22-1):
		fpr[i], tpr[i], _ = roc_curve(trueLabelsBin[:, i], scores[:, i])
		roc_auc[i] = auc(fpr[i], tpr[i])


	fpr["micro"], tpr["micro"], thresholds = roc_curve(trueLabelsBin.ravel(), scores.ravel())
	roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])	
	plt.plot(fpr["micro"], tpr["micro"])
	plt.title('ROC')
	plt.xlabel('FPR')
	plt.ylabel('TPR')
	plt.savefig(splitt[0] + '_' + splitt[1] + '_roc.png')
	plt.close()


	print('prec, acc, tpr, fpr : ', precision_score(trueLabels, predLabels, average='weighted'), \
		accuracy_score(trueLabels, predLabels), np.average(tpr["micro"]), np.average(fpr["micro"]))

	# EER: point where TPR FPR meet 135deg line OR  the common value when the false acceptance rate (FAR) and false rejection rate (FRR) are equal
	fnr = 1 - tpr["micro"]
	eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr["micro"])))]
	eer = min(fpr["micro"][np.nanargmin(np.absolute((fnr - fpr["micro"])))], fnr[np.nanargmin(np.absolute((fnr - fpr["micro"])))])
	print('equal err rate 		: ', eer)

	# HTER: 1- 0.5(TP / (TP + FN) + TN / (TN + FP)) OR (FAR[index of EER] + FRR[index of EER])/2
	# hter = 1 - 0.5*( tp/(tp+fn) + tn/(tn+fp) )
	hter2 = sum([fpr["micro"][np.nanargmin(np.absolute((fnr - fpr["micro"])))], fnr[np.nanargmin(np.absolute((fnr - fpr["micro"])))]])/2
	print('half total err rate : ', hter2)
