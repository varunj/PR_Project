import matplotlib as mpl
mpl.use('Agg')
import numpy as np
np.random.seed(123)
import glob, os
import pandas as pd
from scipy.spatial.distance import euclidean
from keras.layers.core import Dropout, Activation, Flatten
import matplotlib.pyplot as plt
from math import sqrt
from pprint import pprint
import pdb
np.set_printoptions(threshold=np.nan)
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint, Callback
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from collections import Counter
from sklearn import preprocessing
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from conf_matrix import plot_confusion_matrix

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)

CLASS_NAMES = ['high arm wave', 'horizontal arm wave', 'hammer', 'hand catch', 'forward punch', 'high throw', 'draw X', 'draw tick', 'draw circle', 'hand clap', 'twohand wave', 'side boxing', 'bend', 'forward kick', 'side kick', 'jogging', 'tennis swing', 'tennis serve', 'golf swing', 'pickup-throw']


data = [[] for x in range(0,565)]
labels = []
i = 0
for fileName in glob.glob("MSRAction3D/MSRAction3DSkeletonReal3D_800/*.txt"):
	file = open(fileName, 'r+')
	temp = []
	for eachLine in file.readlines():
		temp.append([float(x) for x in eachLine.strip().split(' ')])

	temp2 = []
	for x in range(0,len(temp)-19,20):
		temp2.append(np.array([temp[xx] for xx in range(x,x+20)]).flatten())

	data[i] = temp2
	labels.append(fileName.split('_')[1])
	i = i + 1

data = np.asarray(data)
le = preprocessing.LabelEncoder()
le.fit(labels)
labels = le.transform(labels)
labels = to_categorical(labels)

dataTrain, dataTest, labelsTrain, labelsTest = train_test_split(data, labels, stratify=labels, test_size=0.3, random_state=42)
print('shape dataTrain, dataTest, labelsTrain, labelsTest: ', dataTrain.shape, dataTest.shape, labelsTrain.shape, labelsTest.shape)


model = Sequential()  
model.add(LSTM(40, input_dim=60, input_length=40, return_sequences=True))
model.add(Flatten())
model.add(Dense(20, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# print(model.summary())
# plot_model(model, to_file='model.png')
# print (model.get_config())
#model.layers[0].w.get_value()
#output_layer=model.layers[0].get_output()
#SVG(model_to_dot(model).create(prog='dot', format='svg'))


checkpoint = ModelCheckpoint("model.hdf5", monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

history = model.fit(dataTrain, labelsTrain, epochs=30, batch_size=10, verbose=1, callbacks=callbacks_list,validation_split=0.30)

#model.save('my_model20.h5')
#model.save_weights('my_model_weights20.h5')

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'])
plt.savefig('accuracy.png')
plt.close()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'])
plt.savefig('loss.png')
plt.close()



scores = model.predict(dataTest)
labelsPred = [np.argmax(x) for x in scores]
labelsTest = [np.argmax(x) for x in labelsTest]
print(scores.shape)

acc = accuracy_score(labelsTest, labelsPred)
print(acc)

# cmc curve
genuineArr, imposterArr = [], []
for x in range(0,len(labelsTest)):
	genuineArr.append(scores[x][labelsTest[x]])
	imposterArr.append(np.concatenate((scores[x][:labelsTest[x]], scores[x][labelsTest[x]+1:]), axis=0))
genuineArr = np.asarray(genuineArr).flatten()
imposterArr = np.asarray(imposterArr).flatten()
p, x = np.histogram(genuineArr, bins=int(len(scores)/30))
pMatch = p/max(p)
x = x[:-1] + (x[1] - x[0])/2
plt.plot(x, pMatch, label='genuine')
p2, x2 = np.histogram(imposterArr, bins=int(len(scores)/30))
pImposter = p2/max(p2)
x2 = x2[:-1] + (x2[1] - x2[0])/2
plt.plot(x2, pImposter, label='imposter')
plt.tight_layout()
plt.legend()
plt.title('Match Score Distribution')
plt.xlabel('Score')
plt.ylabel('P(Match at Score)')
plt.savefig('matchscore.png')
plt.close()

# roc, tpr, fpr
labelsTestBin = label_binarize(labelsTest, classes=list(set(labelsTest)))
tpr, fpr, roc_auc = {}, {}, {}
for i in range(20):
	fpr[i], tpr[i], _ = roc_curve(labelsTestBin[:, i], scores[:, i])
	roc_auc[i] = auc(fpr[i], tpr[i])
fpr["micro"], tpr["micro"], _ = roc_curve(labelsTestBin.ravel(), scores.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])	
plt.plot(fpr['micro'], tpr['micro'])
plt.plot([0, 1], [0, 1], 'k--', label='ROC Curve (AUC = %0.6f)' % (roc_auc["micro"]))
plt.tight_layout()
plt.legend()
plt.title('ROC')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.savefig('roc.png')
plt.close()


# conf matrix
cnf_matrix = confusion_matrix(labelsTest, labelsPred)
plot_confusion_matrix(cnf_matrix, classes=[x for x in range(len(CLASS_NAMES))], fileName='confMat.png')
