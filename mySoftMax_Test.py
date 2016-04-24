#coding:utf-8
from numpy  import *
from scipy import optimize
import math
import scipy.io as sio
from numpy  import *
import numpy
from scipy.sparse import *

def loadMNISTImages(filename):
	f = open(filename, 'rb')
	assert f != -1, 'Could not open %s' % filename
	magic = fromfile(f, dtype='>i4', count=1)
	assert magic == 2051, 'Bad magic number in %s' % filename
	numImages = fromfile(f, dtype='>i4', count=1)
	numRows = fromfile(f, dtype='>i4', count=1)
	numCols = fromfile(f, dtype='>i4', count=1)
	images = fromfile(f, dtype='B')
	images = images.reshape(numImages, numCols, numRows)
	f.close()
	images = images.reshape(images.shape[0], images.shape[1]*images.shape[2])
	images = double(images) / 255
	return images
	
	
def loadMNISTLabels(filename):
	f = open(filename, 'rb')
	assert f != -1, 'Could not open %s' % filename
	magic = fromfile(f, dtype='>i4', count=1)
	assert magic == 2049, 'Bad magic number in %s' % filename
	numLabels = fromfile(f, dtype='>i4', count=1)
	labels = fromfile(f, dtype='B')
	assert labels.shape[0] == numLabels, 'Mismatch in label count'
	f.close()
	return labels



def computeNumericalGradient(J, theta): #计算数值梯度
	numgrad = zeros(theta.shape)
	print "numgrad:",numgrad,'theta:',theta,'J:',J
	EPSILON = 1e-04
	bases = eye(numgrad.shape[0])
	print "bases:",bases
	for i in range(numgrad.shape[0]):
		(value1, grad1) = J(theta + EPSILON*bases[:,i])
		(value2, grad2) = J(theta - EPSILON*bases[:,i])
		numgrad[i] = (value1 - value2) / (2*EPSILON)
		print "(value1, grad1):",(value1, grad1),"(value2, grad2):",(value1, grad1),"numgrad[i]",numgrad[i]
	return numgrad

def cost(thetaParam, numClasses, inputSize, lambdaParam, data, labels):
	
	thetaParam = thetaParam.reshape(numClasses, inputSize)#展开theta系数
	m = data.shape[0]
	groundTruth = csc_matrix( (ones(m),(labels,range(m))), shape=(numClasses,m) ).todense()
	cost = 0
	M = thetaParam.dot(data.T)
	M = M - amax(M, 0)
	h_data = exp(M)
	h_data = h_data / sum(h_data, 0)
	cost = -sum(multiply(groundTruth, log(h_data)))/m + lambdaParam/2 * sum(thetaParam**2)#计算代价

	thetaGrad = -((groundTruth - h_data).dot(data))/m + lambdaParam*thetaParam#梯度计算

	return (cost, squeeze(array(thetaGrad.ravel())))


def predict(thetaParam, data):	
	h_data = exp(thetaParam.dot(data.T))
	h_data = h_data / sum(h_data, 0)
	return argmax(h_data, axis=0)





def writeToTxt(list_name,file_path):
    try:
        fp = open(file_path,"w+")
        for item in list_name:
            fp.write(str(item)+"\n")
	    print "item--",item
        fp.close()
    except IOError:
        print("fail to open file")

def train_SoftMax():
	inputSize = 28 * 28		#定义输入图片大小
	numClasses = 10			#定义分类类别为10类
	lambdaParam = 1e-4		# 定义权重衰减系数（惩罚系数）
	trainData = loadMNISTImages('mnist/train-images-idx3-ubyte') #导入数据图片
	trainLabels = loadMNISTLabels('mnist/train-labels-idx1-ubyte')#导入标签
	print size(trainLabels)
	thetaParam = 0.005 * random.normal(size=numClasses * inputSize)#随机初始化theta
	options = {
			'maxiter': 100,
			'disp': True,
		}
	def softmaxCostCallback(x):
		return cost(x, numClasses, inputSize, lambdaParam, trainData, trainLabels) 
	result = optimize.minimize(softmaxCostCallback, thetaParam, method='L-BFGS-B', jac=True, options=options)
	optTheta = result.x[0:numClasses*inputSize].reshape(numClasses, inputSize)
	numpy.save("opt.npy",optTheta);

def test_SoftMax():
	testData = loadMNISTImages('mnist/t10k-images-idx3-ubyte')
	testLabels = loadMNISTLabels('mnist/t10k-labels-idx1-ubyte')
	optTheta=numpy.load("opt.npy");
	pred = predict(optTheta, testData)
	acc = mean(testLabels==pred)
	print size(testLabels)
	print('Accuracy: %0.3f%%\n' % (acc * 100))
	

if __name__ == "__main__":
	#train_SoftMax();
	test_SoftMax();
	
