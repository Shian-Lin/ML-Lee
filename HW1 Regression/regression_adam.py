import pandas as pd
import numpy as np
import math

def featTransform(x):
	x = np.concatenate((x, x[:,9].reshape(-1,1) **2), axis = 1)
	#x = np.delete(x, 15, 1)
	x = np.concatenate((x, x**2), axis = 1)
	x = np.concatenate((x, x[:, :x.shape[1] - 1] * x[:, 1:x.shape[1]]), axis = 1)
	x = np.concatenate((x, x**3), axis = 1)
	x = np.concatenate((x, x**4), axis = 1)
	return x
def computeCost(x, y, theta):
	return np.sum((np.dot(x , theta) - y)**2) / x.shape[0]
#training
def fitAdam(x_train, y_train,theta , lr=0.001, num_iter=10000, beta1=0.1, beta2=0.7):
	eps = 1e-7
	mt = np.zeros([x_train.shape[1], 1])
	vt = np.zeros([x_train.shape[1], 1])
	for cnt in range(num_iter):

		hx = np.matmul(x_train, theta)
		grad = (2* np.matmul(-x_train.T, y_train - hx)) / x_train.shape[0]
		
		#momentum
		mt = beta1 * mt + (1 - beta1) * grad
		vt = beta2 * vt + (1 - beta2) * (grad ** 2)
		mhat = mt / (1 - beta1 ** num_iter)
		vhat = vt / (1 - beta2 ** num_iter)

		#update weights
		theta = theta - lr * mhat / (np.sqrt(vhat) + eps)

		J.append(computeCost(x_train, y_train, theta))

		print("Cost:", computeCost(x_train, y_train, theta))

	return theta

def SGD(x_train, y_train, theta ,sample = 5000, lr=1e-3, num_iter=10000, beta1=0.9, beta2=0.99):
	eps = 1e-7
	mt = np.zeros([x_train.shape[1], 1])
	vt = np.zeros([x_train.shape[1], 1])
	
	for cnt in range(num_iter):
		picked = np.random.choice(x_train.shape[0], sample)
		
		x = x_train[picked]
		y = y_train[picked]
		
		hx = np.matmul(x, theta)
		
		grad = 2 * np.matmul(-x.T, (y-hx)) / sample
		#print('grad ', grad)
		"""
		#momentum
		mt = beta1 * mt + (1 - beta1) * grad
		vt = beta2 * vt + (1 - beta2) * (grad ** 2)
		mhat = mt / (1 - beta1 ** num_iter)
		vhat = vt / (1 - beta2 ** num_iter)
		
		#update weights
		theta = theta - lr * mhat / (np.sqrt(vhat) + eps)
		"""
		theta = theta - lr * grad

		J.append(computeCost(x, y, theta))

		print("Cost:", computeCost(x, y, theta))

	return theta
#load data
df = pd.read_csv('train.csv', encoding='big5')

#drop unused column
rawData = df.iloc[:, 3:]

#replace NR 
rawData = rawData.replace('NR', 0)


#convert to np
rawData = rawData.to_numpy()

#split data into 12 portions (1 month/unit)
monthData = {}
for mth in range(12):
	buff = np.empty([18, 480]) #18 features, 24 hrs * 20 days = 480
	for day in range(20):
		buff[:, 24 * day: 24 * (day + 1)] = rawData[18 * (day + mth * 20): 18 * (day + mth * 20 + 1), :]
	monthData[mth] = buff

#make training data and label

x = np.empty([12 * 471, 18 * 9], dtype = float)
y = np.empty([12* 471, 1], dtype = float)

for mth in range(12):
	for data in range(471):
		x[mth * 471 + data,:] = monthData[mth][:,data: data+9].reshape(1,-1)
		y[mth * 471 + data,0] = monthData[mth][9, data + 9]


#add more features
x = featTransform(x)


#normalize features
x_mean = np.mean(x, axis=0)
x_std = np.std(x, axis=0)
print("mean:", x_mean)
print("std:", x_std)
input("Press Enter to continue...")

for ins in range(x.shape[0]):
	x[ins] = (x[ins] - x_mean) / x_std




#add bias term
x = np.concatenate((np.ones([x.shape[0],1]), x), axis = 1)
print(x.shape)
np.random.shuffle(x)
x_train = x[: math.floor(len(x) * 0.8)] 
x_val = x[math.floor(len(x) * 0.8):]

y_train = y[:math.floor(len(y) * 0.8)]
y_val = y[math.floor(len(y) * 0.8):]


print("shape of whole data:", x.shape)
print("data to train:", x_train)
input("Press Enter to continue...")
#suffle the data(optional)

#start training

dim = x.shape[1]
num_train = x.shape[0]
theta = np.zeros([dim, 1])
J = [] #history of costs
#J.append(computeCost(x_train, theta, y_train))
#print("Initial Cost: {}".format(J[0]))

#params for training

plotCurve = False
if plotCurve:
	import matplotlib.pyplot as plt
	train_cost = []
	val_cost = []
	num_iter = 2000
	
	for cnt in range(1,num_iter):
		
		tmp_xTrain = x_train[:cnt]
		tmp_yTrain = y_train[:cnt]

		theta = fitAdam(tmp_xTrain, tmp_yTrain, theta)
		
		cost = computeCost(tmp_xTrain, tmp_yTrain, theta)
		train_cost.append(cost)
		valCost = computeCost(x_val, y_val, theta)
		val_cost.append(valCost)

		print("Round {}, val loss: {}".format(cnt, valCost))
	print("start plotting learning curve...")
	xAxis = [x for x in range(num_iter-1)]

	plt.plot(xAxis, train_cost, color = 'blue', label="Train")
	plt.plot(xAxis, val_cost, color = 'green', label="Validation")
	plt.xlabel('Size of training data')
	plt.ylabel('Loss rate')
	plt.legend()
	plt.show()
else:
	eigvals = np.linalg.eigvals(np.matmul(x_train.T, x_train))
	print(eigvals)
	print(abs(np.max(eigvals)))

	theta = fitAdam(x_train, y_train, theta)

	print("cost of validation:", computeCost(x_val, y_val, theta))
	
	#clean testing data
	df = pd.read_csv('test.csv',header = None)

	#drop unused column
	rawData = df.iloc[:, 2:]

	#replace NR 
	rawData = rawData.replace('NR', 0)
	

	#convert to np
	rawData = rawData.to_numpy()

	#split data into 12 portions (1 month/unit)
	testData = np.empty([rawData.shape[0] // 18, 18 * 9], dtype=float)
	for id in range(rawData.shape[0] // 18):
		testData[id] = rawData[id * 18: (id + 1) * 18, :].reshape(1,-1)

	testData = featTransform(testData)
	#testData = np.delete(testData, 15, 1)
	#testData = testData[:, 1:]
	#testData = np.concatenate((testData, testData**2), axis = 1)
	#testData = np.concatenate((testData, testData[:, :testData.shape[1] - 1] * testData[:, 1:testData.shape[1]]), axis = 1)
	#testData = np.concatenate((testData, testData**3), axis = 1)
	#x = np.concatenate((x, x**4), axis = 1)
	#make training data and label


	#normalize features

	for ins in range(testData.shape[0]):
		testData[ins] = (testData[ins] - x_mean) / x_std


	testData = np.concatenate((np.ones([testData.shape[0],1]), testData), axis = 1)

	output = pd.DataFrame(["id_{}".format(x) for x in range(testData.shape[0])], columns = ["id"])
	output['value'] = np.dot(testData, theta)
	output.to_csv('predict.csv', index = False)
	print("output", output)



