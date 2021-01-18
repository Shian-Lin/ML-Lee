import pandas as pd
import numpy as np
import math
from sys import argv
## read data
X = pd.read_csv("X_train", index_col = 0).to_numpy()
#X = X.loc[:, (~(X == 0)).any()]
y = pd.read_csv("Y_train", index_col = 0).to_numpy()

##calculate mean and std
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)

##normalize the data
for row in range(X.shape[0]):
	X[row] = (X[row] - X_mean) / (X_std + 1e-7)

##add bias term
X = np.concatenate((np.ones([X.shape[0],1]), X), axis = 1)

##split the data
X_train = X[: math.floor(len(X) * 0.8)] 
X_val = X[math.floor(len(X) * 0.8):]

y_train = y[:math.floor(len(y) * 0.8)]
y_val = y[math.floor(len(y) * 0.8):]

print("shape of training data:", X_train.shape)
print("data to train:", X_train)
input("Press Enter to continue...")

def sigmoid(x):
	return 1 / (1 + np.exp(-x))
def computeCost(x, y, theta):
	eps = 1e-7
	
	return np.sum(-(y * np.log(sigmoid(np.dot(x, theta)) + eps) + (1 - y) * np.log(1- sigmoid(np.dot(x, theta))+ eps))) / x.shape[0]

theta = np.zeros([X_train.shape[1], 1])
J = []
lr = 0.004
num_iter = 1000
eps = 1e-7
beta1 = 0.9
beta2 = 0.99
mt = np.zeros([X_train.shape[1], 1])
vt = np.zeros([X_train.shape[1], 1])
check = int(argv[1])
filename = argv[2]
if check:
	theta = np.load(filename)
	
	predict = np.dot(X_val, theta)
	predict[predict>=.5] = 1
	predict[predict<.5] = 0
	print("cost of validation:", computeCost(X_val, y_val, theta))
	print("accuracy:", (predict == y_val).sum() / X_val.shape[0])

	##predict testing data
	X_test = pd.read_csv("X_test", index_col = 0).to_numpy()
	for row in range(X_test.shape[0]):
		X_test[row] = (X_test[row] - X_mean) / (X_std + 1e-7)
	X_test = np.concatenate((np.ones([X_test.shape[0],1]), X_test), axis = 1)
	predict = np.dot(X_test, theta)
	predict[predict>=.5] = 1
	predict[predict<.5] = 0
	output = pd.DataFrame([x for x in range(X_test.shape[0])], columns = ["id"])
	output['label'] = predict
	output['label'] = output['label'].astype(int)
	output.to_csv('predict.csv', index = False)
	#X_test = X_test.loc[:, (~(X_test == 0)).any()].to_numpy()


else:
	for cnt in range(num_iter):
		hx = y_train - sigmoid(np.dot(X_train, theta))
		grad = np.sum(-X_train * hx / X_train.shape[0], axis = 0).reshape(-1,1)
		#momentum
		mt = beta1 * mt + (1 - beta1) * grad
		vt = beta2 * vt + (1 - beta2) * (grad ** 2)
		mhat = mt / (1 - beta1 ** num_iter)
		vhat = vt / (1 - beta2 ** num_iter)
		theta = theta - lr * mhat / (np.sqrt(vhat) + eps)
		J.append(computeCost(X_train, y_train, theta))

		print("Cost:", computeCost(X_train, y_train, theta))
	predict = np.dot(X_train, theta)
	predict[predict>=.5] = 1
	predict[predict<.5] = 0
	print("accuracy:", (predict == y_train).sum() / X_train.shape[0])
	np.save(filename, theta)
	











