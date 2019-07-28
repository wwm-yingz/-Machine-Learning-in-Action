#机器学习实战
#chap2 K-临近算法
#wwm-yingz

from numpy import *
import operator #运算符模块
import  os

def createDataSet():#创建数据集和标签
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels
    pass

#2-1 k-近邻算法
#(输入向量,样本集,样本标签,k值
def classify0(inX, dataSet, labels,k):
    #距离计算 (欧式距离)
    dataSetSize = dataSet.shape[0]#样本集数量
    diffMat = tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    # 选距离最小的k个点
    sortedDistIndicies = distances.argsort() #argsort返回值是从小到大的索引
    classCount={}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]] #找最接近的前k个样本集的label出来
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1#计数 空值返回0
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True) #python3中 iteritems->items
    #排序 iteritems是迭代器 迭代器形如[[key1,value1],[key2,value2],...,[keyn,valuen]]  itemgetter(1)代表获取对象第一个域的值,也就是value的值,其实也就是key出现的次数
    return sortedClassCount[0][0] #第一个0代表出现次数最多的那组,第二个0代表是key,也就是label(1代表出现次数)


#约会问题
#txt转np.array格式
def file2matrix(filename):
    fr = open(filename) #文件IO
    numberOfLines = len(fr.readlines())         #get the number of lines in the file
    returnMat = zeros((numberOfLines,3))        #prepare matrix to return
    classLabelVector = []                       #prepare labels return
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip() #删掉回车
        listFromLine = line.split('\t')  #变list
        returnMat[index,:] = listFromLine[0:3] #一共有4列 前3列是特征,第4列是label
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat,classLabelVector  #样本集  样本标记

#归一化处理,将各个特征归一,避免因为数值大小不同导致权重不同.  转[0,1]之间
def autoNorm(dataSet):
    minVals = dataSet.min(0)#每一列的最小值,0代表的是维度索引
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m,1))
    normDataSet = normDataSet/tile(ranges, (m,1))   #element wise divide
    return normDataSet, ranges, minVals

#约会网站的分类器测试
def datingClassTest():
    hoRatio = 0.10      #用于测试分类器正确率的样本数据比例  因为k-邻近算法不需要训练
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')       #读取数据
    normMat, ranges, minVals = autoNorm(datingDataMat) #归一化
    m = normMat.shape[0] #总行数
    numTestVecs = int(m*hoRatio)  #用于测试的样本个数
    errorCount = 0.0 #错误计数
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print ("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))#测试结果,真实结果
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print( "the total error rate is: %f" % (errorCount/float(numTestVecs)))
    print (errorCount)



# 手写数字识别
#img图像转array    图像已转为32*32的01黑白像素图,将其转为1*1024的向量,即可用之前的分类器处理,不用重写classify0
def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

#测试函数
def handwritingClassTest():
    #获取目录内容
    hwLabels = []
    trainingFileList = os.listdir('trainingDigits')           #load the training set
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    #从文件名获取数字
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)
    testFileList = os.listdir('testDigits')        #iterate through the test set
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print ("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
        if (classifierResult != classNumStr): errorCount += 1.0
    print ("\nthe total number of errors is: %d" % errorCount)
    print ("\nthe total error rate is: %f" % (errorCount/float(mTest)))